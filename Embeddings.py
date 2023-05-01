import numpy as np
import pandas as pd


def get_glove_embeddings(feature_dims = 50):
    """ returns a dictionary of glove embeddings """
    
    # create a dictionary for the embeddings
    embeddings_dict = {}

    # Define glove path
    glove_path = f"glove.6B.{feature_dims}d.txt"

    # read in glove embeddings
    with open(glove_path, 'r',encoding="utf-8") as f:
        
        for line in f:
            # split the string containing word and vector
            values = line.split()

            # get the word
            word = values[0]

            # get the vector
            vector = np.asarray(values[1:], "float32")

            # insert into dictionary
            embeddings_dict[word] = vector
            
    return embeddings_dict




def knn(source,embedding,k = 10,distance = "cosine"):
    """
    Get k most similar words to source word
    
    Args:
    source: source word for which to find k most similar words
    embedding: word embedding (GloVe)
    k: number of neighbors
    distance: distance metric determining similarity (cosine,dot)
    

    Returns:
    Pandas dataframe containing k most similar words and their proximity
    
    """
    # the source word is counted as a neighbor so we need k+1 to get k neighbors
    k += 1

    # source word
    source_word = embedding[source]
        
    # embedding of all words
    all_words = np.array(list(embedding.values()))
        
    # array of all word vectors
    arr = np.array(list(embedding))
        
    # calculate similarity according to distance metric
    if distance == "cosine":
        similarity = (source_word @ all_words.T) / (np.linalg.norm(source_word) * np.linalg.norm(all_words,axis=1))
        
    if distance == "dot":
        similarity = (source_word @ all_words.T)
    
    # get the indices of k most similar words
    top = np.argsort(similarity)[::-1][:k]
        
    # get the k most similar words
    tops = arr[top]
        
    # create a df for the word and its similarity score
    table = pd.DataFrame(columns = ["word","similarity"])

    # go over words in k most similar words
    for index,word in enumerate(tops):

        # add new row to df with word & similarity score
        table.loc[-1] = [word,similarity[top[index]]]
            
        # add an index & sort by index (to be able to keep adding new rows)
        table.index = table.index + 1
        table = table.sort_index()    
        

    
    return table