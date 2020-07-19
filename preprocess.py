# Utilities
import datetime
import csv
import os
import pickle
import time
import re 
import string

# Common Data Science Library
import pandas as pd
import numpy as np

# nltk library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# normalization 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from config import TARGETS,PREPROCESSED_TRAINING_DATA,PREPROCESSED_TEST_DATA ,ORIGINAL

nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_training_data():
    """ Method that preprocess all data in data/ directory and creates a training_data.pkl """
    # Get training data
    dfs = [] # create a list of data frames
    files = os.listdir("data") 
    for filename in files:
        path = "data/" + filename
        x,y = preprocessing(path)
        dfs.append(pd.DataFrame(list(zip(x,y)), columns=["Features", "Class"]))
    pd.concat(dfs).to_pickle(PREPROCESSED_TRAINING_DATA)

def preprocess_test_data():
    """ Method that prepares test data from CharlesDickensTwoTales_orig.txt """
    x, y = preprocessing(ORIGINAL)  
    pd.DataFrame(list(zip(x,y)), columns=["Features", "Class"]).to_pickle(PREPROCESSED_TEST_DATA) 

def preprocessing(filename):
    """ 
    Method that takes a file name as input and outputs a preprocessed dataset with two columns:
    1- features of the target word.
    2- sense of the target word "a|the", e.g. 0 for "a" and 1 for "the"
    """
    with open(filename, "r", encoding="utf8") as f:
        # get text as a single string by replacing all \n characters with a space
        text = f.read().replace("\n", " ").strip()
    
    # Split into a list of sentences, we assumed here that 
    sentences = sent_tokenize(text)

    # Clean each sentences
    sentences = [clean_sentence(sentence) for sentence in sentences]

    # remove all sentences where a or the are not present
    sentences = [sentence for sentence in sentences if "a/DT" in sentence or "the/DT" in sentence]
        
    # extract features
    x, y = extract_features(sentences)

    # replace all appearances of a/DT or the/DT in the feature vector to a|the
    x = [re.sub(r'\b(a/DT|the/DT)\b', "a|the", feature_vector) for feature_vector in x]
    
    return x, y
    
def clean_sentence(sentence):
    """ Method that cleans a single sentence """
    # Remove punctuations, put all words to lower case and tokenize each sentence
    word_tokens = word_tokenize(re.sub("[!\"#$%&'()\*\+,-./:;<=>\?@\[\]^_`{\|}~]", " ", sentence).lower())
     
    # POS each word in each list, .pos_tag returns a list of tuples (word, tag), this line concatenate them into word/tag, e.g. shop/NN  
    word_tokens = ["/".join(tup) for tup in nltk.pos_tag(word_tokens)]

    return word_tokens

def extract_features(sentences):
    """ Given a list of sentences, return a feature vector and a target vector for each training instance """
    x = []
    y = []
    collocations = list(range(-2,3)) # create a list of indeces for creating collocations 
    collocations.remove(0)
    for tokens in sentences:
        for counter, token in enumerate(tokens):
            if token == TARGETS[0]: 
                collocation = ""
                for number in collocations: 
                  collocation += get_index(tokens, counter+number) + " "
                x.append(collocation.strip())
                y.append(0) # 0 stands for a/DT
            elif token == TARGETS[1]: 
                collocation = ""
                for number in collocations: 
                  collocation += get_index(tokens, counter+number) + " "
                x.append(collocation.strip())
                y.append(1) # 1 stands for the/DT
    return x, y
    
def get_index(tokens, index):
    """ Helper method which gets the index of a list and returns an empty string if it is out of bound """
    return tokens[index] if index < len(tokens) and index > -1 else ""

def normalize(word_tokens):
    """ Method that normalizes the word format in a list of tokens. """
    normalizer = SnowballStemmer("english")
    for counter,value in enumerate(word_tokens):
        word_tokens[counter] = normalizer.stem(value)
