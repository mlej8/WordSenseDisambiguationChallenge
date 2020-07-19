from preprocess import preprocess_test_data, preprocess_training_data
from config import PREPROCESSED_TRAINING_DATA,PREPROCESSED_TEST_DATA, DISAMBIGUATED, AMBIGUOUS
from classification import get_predictions
from validation import cross_validation, grid_search_cv

# models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import re

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

if __name__ == "__main__":
    # get training and test dataset
    # preprocess_training_data()
    # preprocess_test_data()

    # get test data
    test_df = pd.read_pickle(PREPROCESSED_TEST_DATA)
    x_test = test_df["Features"]
    y_test = test_df["Class"]

    # get training data
    train_df = pd.read_pickle(PREPROCESSED_TRAINING_DATA)
    x_train = train_df["Features"]
    y_train = train_df["Class"]

    # Models
    log_reg = LogisticRegression(C=0.5, penalty="l2", max_iter=1000, multi_class='auto', n_jobs=-1, solver='newton-cg',  dual=False, warm_start=True) 
    multi_NB = MultinomialNB(alpha=1.75)
    clf = VotingClassifier(estimators=[('lr', log_reg), ("nb", multi_NB)], voting="soft")


    # Disambiguate text using predictions from model
    predictions = get_predictions(log_reg, x_train, y_train, x_test, y_test)
    disambiguate_text(predictions)







    ### VALIDATION ### 
    # # Number of folds for Cross Validation
    # folds = 10

    # # Perform Cross-Validation to validate model 
    # print(cross_validation(model=log_reg, X=x_train, y=y_train, folds=folds))

    # # Model parameters
    # params = {
    # 'clf__C':(1, 0.5, 0.25,0.75,2)
    # }
    # params = {
    # 'clf__alpha':(1.7,1.75,1.8)
    # }

    # # Perform Grid Search CV to find the best parameters
    # best_scores, best_params, best_estimator_params = grid_search_cv(model=multi_NB,  X=x, y=y, params=params, folds=folds)
