import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.array(y_train)
    maj_class = np.bincount(y_train).argmax()
    return [maj_class]*len(X_test)