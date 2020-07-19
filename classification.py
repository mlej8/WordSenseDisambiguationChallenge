from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def get_predictions(model, x_train, y_train, x_test, y_test): 
    """ Given a model, a feature vector and a target vector, output prediction the predictions on the training set document since the ofuscated and the original documents both generate the same feature vector/target vectors """
    # Pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 3))),
        ('clf', model)],
         verbose=True)
    pipeline.fit(x_train,y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Predicted {} out of {} correctly\nAccuracy score: {:.2f}".format(accuracy * len(predictions), len(predictions), accuracy))
    return predictions