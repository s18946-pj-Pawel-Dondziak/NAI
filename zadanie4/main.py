import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
'''
Jakub Świderski s19443
Paweł Dondziak s18946

To properly run the program you need to install sklearn and pandas

This program is used to predict disease based on Decision Tree Classifier data and SVC.
'''

def read_data(col_names, feature_cols, target, csv_file):
    '''
    This function id used to load the data from  .csv data

    Parameters:
    col_names: csv file header
    feature_cols: features
    target: target to learn
    csv_file: path to csv file

    Returns: List containing train-test split of inputs.
    ''' 
    pima = pd.read_csv(csv_file, header=None, names=col_names)
    X = pima[feature_cols]
    y = pima[target]
    return train_test_split(X, y, test_size=0.3)

def calculate(clf, X_train, X_test, y_train, y_test):
    '''
    This function is used to calculate the score.

    Params:
    clf:
    X_train: The training input samples
    X_test: The input samples
    y_train: The target values
    y_test:  Ground truth (correct) labels

    Returns: The fraction of correctly classified samples
    '''
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

print("Pima Indians Diabetes")
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X_train, X_test, y_train, y_test = read_data(col_names, feature_cols, "label", "pima-indians-diabetes.csv")

print("Decision Tree Classifer Accuracy:", calculate(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
print("SVC Accuracy:", calculate(SVC(), X_train, X_test, y_train, y_test))

print("Kyphosis")
col_names = ["Kyphosis","Age","Number","Start"]
feature_cols = ["Age","Number","Start"]
X_train, X_test, y_train, y_test = read_data(col_names, feature_cols, "Kyphosis", "kyphosis.csv")

print("Decision Tree Classifer Accuracy:", calculate(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
print("SVC Accuracy:", calculate(SVC(), X_train, X_test, y_train, y_test))
