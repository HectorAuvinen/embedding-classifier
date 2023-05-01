import torch

class ClassificationAverageModel(torch.nn.Module):
    """
    lorem ipsum
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