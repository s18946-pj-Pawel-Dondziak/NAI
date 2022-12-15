import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''
Authors: Paweł Dondziak s18946, Jakub Świderski s19443
To run program you need to install tensorflow, sklearn, keras, seaborn, pandas,and matplotlib packages
You'll also need python version from 3.5 to 3.8
Determining if someone has prostate cancer based on data from previous classes. 
'''

Cancer = pd.read_csv('prostate_cancer.csv')
Cancer.drop(['id'], axis=1, inplace=True)
Cancer.diagnosis_result = [1 if each ==
                           'M' else 0 for each in Cancer.diagnosis_result]

'''
split date, y =  test, x_data = train
'''
y = Cancer.diagnosis_result.values
x_data = Cancer.drop(['diagnosis_result'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x_data)

'''
split datas as train and test.
'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

method_names = []
method_scores = []

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)  # Fitting
print("Logistic Regression Classification Test Accuracy {}".format(
    log_reg.score(x_test, y_test)))
method_names.append("Logistic Reg.")
method_scores.append(log_reg.score(x_test, y_test))

'''
Confusion Matrix
'''
y_pred = log_reg.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
'''
Visualization Confusion Matrix
'''
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=50, kernel_initializer='uniform',
                   activation='relu', input_dim=x_train.shape[1]))
    classifier.add(
        Dense(units=10, kernel_initializer='uniform', activation='relu'))
    classifier.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, epochs=300)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: " + str(mean))
print("Accuracy variance: " + str(variance))
