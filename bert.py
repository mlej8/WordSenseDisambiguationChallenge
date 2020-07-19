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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# models
from sklearn.linear_model import LogisticRegression

# nltk library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# normalization 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Deep Learning
# import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# bert
from transformers import BertTokenizer, BertModel
from config import AMBIGUOUS, DISAMBIGUATED, TARGETS, ORIGINAL

BERT_PREPROCESS_TRAINING_DATA = "bert_preprocessed_training_data.pkl"
BERT_PREPROCESS_TEST_DATA = "bert_preprocessed_test_data.pkl"
TRAINING_CONTEXTUAL_WORD_EMBEDDINGS ="training_word_embeddings.pkl"

nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load pre-trained model with pretrained weights
print('Loading BERT model...')
pretrained_weights = 'bert-base-uncased'
model = BertModel.from_pretrained(pretrained_weights)

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

# # Get the GPU device name.
# device_name = tf.test.gpu_device_name()

# # The device name should look like the following:
# if device_name == '/device:GPU:0':
#     print('Found GPU at: {}'.format(device_name))
# else:
#     raise SystemError('GPU device not found')

# # Tell pytorch to use GPU
# import torch

# # If there's a GPU available...
# if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

def preprocess_training_data():
    """ Method that preprocess all data in data/ directory and creates a training_data.pkl """
    # Get training data
    dfs = [] # create a list of data frames
    files = os.listdir("data") 
    for filename in files:
        path = "data/" + filename
        x,y = preprocessing(path)
        dfs.append(pd.DataFrame(list(zip(x,y)), columns=["Features", "Class"]))
    pd.concat(dfs).to_pickle(BERT_PREPROCESS_TRAINING_DATA)

def preprocess_test_data():
    """ Method that prepares test data from CharlesDickensTwoTales_orig.txt """
    x, y = preprocessing(ORIGINAL)  
    pd.DataFrame(list(zip(x,y)), columns=["Features", "Class"]).to_pickle(BERT_PREPROCESS_TEST_DATA) 

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
    sentences = [sentence for sentence in sentences if "a" in sentence or "the" in sentence]
        
    # extract features
    x, y = collocation_feature_extraction(sentences)

    # replace all appearances of a or the in the feature vector to a|the
    x = [re.sub(r'\b(a|the)\b', "", sentence).strip() for sentence in x]
        
    return x, y
    
def clean_sentence(sentence):
    """ Method that cleans a single sentence """
    # Remove punctuations, put all words to lower case and tokenize each sentence
    word_tokens = word_tokenize(re.sub("[!\"#$%&'()\*\+,-./:;<=>\?@\[\]^_`{\|}~]", " ", sentence).lower())
    return word_tokens

def collocation_feature_extraction(sentences):
    """ Given a list of sentences in the form of lists of word tokens, return a feature vector and a target vector for each training instance """
    x = []
    y = []
    collocations = list(range(-2,3)) # create a list of indeces for creating collocations 
    collocations.remove(0)
    for tokens in sentences:
        for counter, token in enumerate(tokens):
            if token == "a": 
                collocation = ""
                for number in collocations: 
                  collocation += get_index(tokens, counter+number) + " "
                x.append(collocation.strip())
                y.append(0) # 0 stands for a
            elif token == "the": 
                collocation = ""
                for number in collocations: 
                  collocation += get_index(tokens, counter+number) + " "
                x.append(collocation.strip())
                y.append(1) # 1 stands for the
    return x, y

def get_index(tokens, index):
    """ Helper method which gets the index of a list and returns an empty string if it is out of bound """
    return tokens[index] if index < len(tokens) and index > -1 else ""
 
def max_length(sentences):
    """ Method that finds the maximum length of sentences in dataset. """
    max_len = 0    
    # For every sentence...
    for sent in sentences:
        if len(sent) > max_len:
            max_len = len(sent)
    return max_len   

def pad(sentences, max_len):
    """ Method that pads every sentences to the max length with 0s. """
    return np.array([sent + [0]*(max_len-len(sent)) for sent in sentences])

def encode(df_column):
    """ Method to encode input sentence into tokens using BertTokenizer. """
    return df_column.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

def get_word_embeddings(path, save_path):
    """ Method that takes a path to a dataframe and feeds it to BERT in order to get contextual word embeddings """
    # get training data
    df = pd.read_pickle(path)

    # get sentences 
    sentences = df.Features.values
    
    # get the max length of encoded tokens
    tokenized = encode(df["Features"]) # encode each sentences using BertTokenizer to have everything tokenized
    print("Number of tokenized sentences: %s" % tokenized.shape)
    max_len = max_length(tokenized) # find the max length of the tokenized sentences

    # pad all sentences to max length
    padded = pad(tokenized.values, max_len)
    print("Padded tokens shape: {}".format(padded.shape))

    # create attention mask
    attention_mask = create_attention_mask(padded)
    print("Attention masks shape: {}".format(attention_mask.shape))

    # create an array of features, e.g. contextual word embeddings from BERT
    features = []

    # split input into chunks for BERT to process (to avoid loading everything into memory once and running out of memory)    
    sections = 1000

    for counter, (padded_section, attention_mask_section) in enumerate(zip(np.array_split(padded,sections, axis=0), np.array_split(attention_mask, sections, axis=0))):
        print('======== Embedding section {} / {} ========'.format(counter + 1, sections))

        # transform input ids and attention masks into tensors
        input_ids = torch.tensor(padded_section).to(torch.int64) # convert to int64 on Windows 10 and unsqueeze shape from torch.Size([17]) to torch.Size([1, 17]) which is the input shape expected by BERT
        attention_masks = torch.tensor(attention_mask_section).to(torch.int64)
        
        # run sentences through BERT and the result of the processing will be returned into last_hidden_states 
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_masks)

        # only preserve the output corresponding to the first token of each sentence (the output corresponding to the [CLS] token contains the embedding for the entire sentence)
        feature = last_hidden_states[0][:,0,:].numpy()
        features.append(feature)

    # concatenate all features
    features = np.concatenate(features, axis=0)
    file_stream = open(file=save_path,mode="wb")
    pickle.dump(features, file_stream)
    file_stream.close()

def create_attention_mask(np_array):
    """ Helper method to create the attention masks of a numpy array. """
    return np.where(np_array != 0, 1, 0)

def disambiguate_text(predictions):
    """ Function that receives a file and disambiguate it using a list of predictions """
    # transform predictions from 0 to a and 1 to the
    predictions = ["a" if prediction == 0 else "the" for prediction in predictions]
    
    with open(AMBIGUOUS, "r") as f:
        # get text as a single string by replacing all \n characters with a space
        text = f.read().strip()

    pseudoword = "a|the"
    text = re.sub(r'\b([Aa]|[tT][Hh][Ee])\b', pseudoword, text)\
    
    # replace each one of a|the by predictions
    for prediction in predictions:
        text = text.replace(pseudoword,prediction, 1)

    # replace all occurences of a and the in the ambiguate file to a|the
    with open(DISAMBIGUATED, "w") as f: 
        f.write(text)   

def cross_validation(model, X, y, folds):
    """ Function that implements cross validation of a model. """
    # Track CV time
    start = time.time()

    # Scores
    scoring = "accuracy"
    scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)

    return "Cross validation scores: {0}\nCross validation mean {1} score: {2:.2F} (+/- {3:.2F})\nValidation time: {4}s".format(scores,scoring, scores.mean(), scores.std() * 2,time.time()-start) 


if __name__ == "__main__":
    # record program runtime
    start = time.time()
    
    # create the training and test corpus
    # preprocess_training_data()
    # preprocess_test_data()

    # using preprocessed data, create word embeddings using BERT.
    # get_word_embeddings(BERT_PREPROCESS_TRAINING_DATA, TRAINING_CONTEXTUAL_WORD_EMBEDDINGS)

    # load data 
    y = pd.read_pickle(BERT_PREPROCESS_TRAINING_DATA)["Class"].values
    X = pickle.load(open(TRAINING_CONTEXTUAL_WORD_EMBEDDINGS, "rb"))
    
    # create model
    model = LogisticRegression(C=0.5, penalty="l2", max_iter=1000, multi_class='auto', n_jobs=-1, solver='newton-cg',  dual=False, warm_start=True) 

    # cross validation folds 
    folds = 10

    # cross validate my model
    print(cross_validation(model, X=X, y=y, cv=folds))

    pickle.dump(model, open("model.pkl", "wb"))
    # # predictions = get_predictions(sentences, labels)
    # # disambiguate_text(predictions)

    print("Program runtime {}".format(time.time()-start))