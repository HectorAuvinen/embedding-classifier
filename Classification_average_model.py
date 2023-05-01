import torch
import numpy as np

from torch import nn

OOV_TOKEN = "[oov]"

class ClassificationAverageModel(torch.nn.Module):
    """
    Class for a classification model that averages word embeddings and then uses a linear layer to classify
    """
    def __init__(self, embeddings, document_dim, num_classes):
        super().__init__()
        
        # weight must be cloned for this to be differentiable
        self.embeddings = torch.nn.Parameter(embeddings.double().weight.clone())

        # linear layer that maps from document_dim to num_classes
        self.linear = torch.nn.Linear(document_dim, num_classes)

        # softmax layer that maps from num_classes to num_classes
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        """
        forward pass of the model.
        input is a batch of documents, which are lists of word ids (batch_size, max_document_length)
        embeddings is a lookup table for the word embeddings (vocabulary_size, embedding_dim)
        output is a batch of class probabilities (batch_size, num_classes)
        """

        # initialize matrix for output   
        y = torch.zeros((x.shape[0], self.embeddings.shape[1]),dtype=torch.float).to(self.device)
    
        # iterate over samples in batch
        for i, sample in enumerate(x):

            # get the embeddings of the sample (max_document_length, embedding_dim)
            out = self.embeddings[sample]

            # calculate the mean, save it in the output y
            y[i] = torch.mean(out, axis=0)
            
        # get output batch through linear layer and then do softmax
        y = self.linear(y)
        y = self.softmax(y)

        return y
    
    def predict(self, x):
        """ predict the class of a batch of documents"""

        # do forward pass
        prodbabilities = self.forward(x)

        # take the class with highest probability
        y = torch.argmax(prodbabilities, axis=1)
        
        return y
    


class NLP_Dictionary():
    """ class for handling the index mapping between dictionary and embeddings, and resizing the data """
    def __init__(self, dictionary, oov_token):
        self.dictionary = dictionary
        self.oov_index = dictionary.loc[dictionary["index"] == oov_token].index.item()


    def word_to_id(self, word : str):
        """return the index of a word"""
        try:
            return self.dictionary.loc[self.dictionary["index"] == word].index.item()
        except:
            return self.oov_index


    def words_to_ids(self, words: list):
        """turn a list of words into a list of ids."""
        # transfor to ids.
        ids = [self.word_to_id(word) for word in words]
        return ids


    def resize(self, words, length, padding_token):
        """resize any list to <length>, pad if too small."""
        
        # safe
        words = words[:length]

        # if the sample is too small, add OOV tokens in the end.
        if len(words) < length:
            words += [padding_token] * (length - len(words))

        assert len(words) == length, "length does not fit the maximum_document_length"

        return words




def prepare_batching_data(
    dictionary: NLP_Dictionary,
    data, # Pandas series of lists of words
    labels, # Pandas series of labels
    max_document_length = 50,
    padding_token=OOV_TOKEN
    ):
    """ Map list of words to list of ids, resize each sample to max_document_length and transform to tensors. """

    labels = torch.tensor(list(labels))

    data = data.apply(
        lambda x : dictionary.resize(
            words = dictionary.words_to_ids(x), 
            length = max_document_length, 
            padding_token = dictionary.word_to_id(padding_token)
            )
        )

    # transform into a tensor of shape (batch_size, max_document_length)
    data = torch.tensor(list(data))

    return data, labels


def batch_generator(
    data,
    labels,
    batch_size
    ):
    """Pyhton Generator that yields batches until there are no more samples."""
    
    n = 0
    while True:
        x = data[n*batch_size:(n+1)*batch_size]
        
        if len(x) == 0:
            break

        y = labels[n*batch_size:(n+1)*batch_size]


        # increment batch counter
        n += 1
        yield n, x, y



@torch.no_grad()
def evaluate(network: nn.Module, data, criterion) -> list:
    """ evaluate the network on the given data using the given criterion"""
    network.eval()
    device = network.device
    results = []

    for i, x, y in data:

        # send data,labels to device
        x, y = x.to(device), y.to(device)

        # do forward pass
        logits = network(x)

        #calculate criterion (loss/accuracy)
        results.append(criterion(logits, y).item())

    return torch.tensor(results)

@torch.enable_grad()
def update(network: torch.nn.Module, data, loss: nn.Module,
           opt: torch.optim.Optimizer) -> list:
    """ update the network on the given data using the given loss and optimizer"""
    network.train()
    device = network.device
    errs = []

    for i, x, y in data:

        # send data,labels to device
        x, y = x.to(device), y.to(device)

        # do forward pass
        logits = network(x)

        # calculate and store loss
        err = loss(logits, y)
        errs.append(err.item())

        # update weights
        opt.zero_grad()
        err.backward()
        opt.step()

    return torch.tensor(errs)

@torch.no_grad()
def accuracy(logits, y):
    """ Compute the accuracy of a prediction. """
    correct = logits.argmax(dim=1) == y
    return torch.mean(correct.float(), dim=0)



def create_embedding_matrix(embeddings_dict,dictionary):
    """create an embedding matrix that maps words to indices. 
    If word is not in embeddings, sample from uniform distribution in range of embedding values"""

    # get bounds for dictionary
    max_val = np.max(np.array(list(embeddings_dict.values())).ravel())
    min_val = np.min(np.array(list(embeddings_dict.values())).ravel())

    # get word vector shape
    wv_shape = list(embeddings_dict.values())[0].shape
    print(f"Dimensionality of the Embeddings: {wv_shape}")

    # define the weight matrix for embeddings
    matrix_len = len(dictionary)

    # (n words in dictionary, dimensionality of embeddings)
    weights_matrix = np.zeros((matrix_len, wv_shape[0]))

    #for word in dictionary["index"].keys():
    for i, word in enumerate(dictionary["index"]):
        # get embedding for word
        try:
            weights_matrix[i] = embeddings_dict[word]
        
        # if not found create a new embedding
        except KeyError:
            weights_matrix[i] = np.random.uniform(min_val,max_val,wv_shape)
            print(f"random vector created for {word}")

    # define embedding matrix
    weights_matrix = torch.tensor(weights_matrix)
    embedding_matrix = torch.nn.Embedding.from_pretrained(weights_matrix)

    return embedding_matrix