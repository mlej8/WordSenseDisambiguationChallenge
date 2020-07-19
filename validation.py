# Sklearn
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import time
import datetime
import pickle
import csv

def cross_validation(model, X, y, folds):
    """ Function that implements cross validation of a model. """
    pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 2))),
        ('clf', model)],
        verbose=True)

    # Track CV time
    start = time.time()

    # Scores
    scoring = "accuracy"
    scores = cross_val_score(pipeline, X, y, cv=folds, scoring=scoring)

    return "Cross validation scores: {0}\nCross validation mean {1} score: {2:.2F} (+/- {3:.2F})\nValidation time: {4}s".format(scores,scoring, scores.mean(), scores.std() * 2,time.time()-start) 

def grid_search_cv(model, X, y, params, folds):
    # Pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 3))),
        ('clf', model)],
         verbose=True)

    # Use GridSearch cross validation to find the best feature extraction and hyperparameters
    gs_CV = GridSearchCV(pipeline, param_grid=params, cv=folds)
    gs_CV.fit(X, y)
    print("Performing grid search...")
    print("Pipeline: ", [name for name, _ in pipeline.steps])
    print("Best parameter (CV score={0:.3f}):".format(gs_CV.best_score_))
    print("Best parameters set: {} \nBest estimator parameters {}.".format(gs_CV.best_params_, gs_CV.best_estimator_.get_params()))

    # Write best params in csv file 
    with open(r"parameters.csv", "a") as f:
        # To write a csv_file we are using a csv_writer from csv module
        csv_writer = csv.writer(f, delimiter=",", lineterminator="\n")         
        # Write current time
        csv_writer.writerow([datetime.datetime.now()])
        score = "Cross Validation score = " + str(gs_CV.best_score_)
        csv_writer.writerow([score])        
        # Write best parameters
        for key, value in gs_CV.best_params_.items(): 
            csv_writer.writerow([key, value])   
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_{}.pkl".format(type(model).__name__), "wb"))

    return (gs_CV.best_score_,gs_CV.best_params_, gs_CV.best_estimator_.get_params())