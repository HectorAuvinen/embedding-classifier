import string
import numpy as np
import pandas as pd

import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def remove_punctuations(text):
    """ remove punctuation from text. Additional punctionations added since not included in string.punctuation"""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '').replace('“', '').replace('”', '').replace('’', '').replace('–', '').replace('•','').replace('‘','')
    return text

def to_lower(text):
    """ transform text to lowercase"""
    return text.lower()

def remove_numbers(text):
    """ remove numbers from text"""    
    return ''.join([i for i in text if not i.isdigit()])

def tokenize(text):
    """ tokenize text """
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    return tokens


def token_counts(data, oov_token = "[oov]",thresh = 100):
    """
    Get token counts for a dataset, including OOV tokens specified by threshold
    """
    # get counts of tokens by unpacking all tokens from all rows
    dictionary = data["tokens"].explode().value_counts()

    # get total number of OOV tokens (sum over count in [word, count])
    n_oov = dictionary[dictionary.values < thresh].values.sum()
    
    # create entry in dictionary for OOV token
    dictionary[oov_token] = n_oov

    # drop original token of all OOV tokens
    dictionary = dictionary[dictionary.values >= thresh]

    # reset the word index
    dictionary = dictionary.reset_index()
    
    return dictionary
# 

def embeddings_to_words(embeddings_dict,dictionary):
    """create a dictionary containing words from the dataset and their embeddings. 
    If word is not in embeddings, sample from uniform distribution in range of embedding values"""

    # get bounds for dictionary
    max_val = np.max(np.array(list(embeddings_dict.values())).ravel())
    min_val = np.min(np.array(list(embeddings_dict.values())).ravel())

    # get word vector shape
    wv_shape = list(embeddings_dict.values())[0].shape

    # create dictionary for word embedding pairs
    dictionary_embeddings = {}

    for word in dictionary["index"]:
        # get embedding for word
        try:
            dictionary_embeddings[word] = embeddings_dict[word]
        
        # if not found create a new embedding
        except KeyError:
            dictionary_embeddings[word] = np.random.uniform(min_val,max_val,wv_shape)
            print(f"random vector created for {word}")

    return dictionary_embeddings
