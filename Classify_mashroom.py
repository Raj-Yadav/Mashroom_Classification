# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:43:08 2016

@author: RAJ
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import ShuffleSplit, train_test_split

data = pd.read_csv("mushrooms.csv")

#### Data Exploration  ###
print("This Dataset has {0} rows and {1} columns".format(data.shape[0],data.shape[1]))
data.isnull().sum()
data.head(1)
data.describe()
data.columns

def data_detail(data):
    for feature, col_data in data.iteritems():
        if col_data.dtype == object:
            print("{} has {}".format(feature, col_data.unique()))

data_detail(data)

y = data["class"]
X = data.drop("class", axis = 1)
X.shape

n_sample = len(data.index)
n_featurs = len(data.columns)-1
n_p = dict(data['class'].value_counts())['p']
n_e = dict(data['class'].value_counts())['e']
p_rate = float(n_p)/float(n_sample)*100


print("Total number of sample is {}".format(n_sample))
print("Total number of features is {}".format(n_featurs))
print("Number of p is {}".format(n_p))
print("Number of e is {}".format(n_e))
print("p_rate is {}".format(p_rate))

##### Data Transformation  #####
X = pd.get_dummies(X)
print("Length of transformed data columns is {}".format(len(X.columns)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

##### Defining Functions #####
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    
def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label= 'p')

def train_predict(clf, X_train, y_train, X_test, y_test):

    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    train_classifier(clf, X_train, y_train)

    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

#### Importing Classifier  ####
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

clf_A = RandomForestClassifier(random_state = 1)
clf_B = SVC(random_state = 1)
clf_C = SGDClassifier(random_state = 1)
clf_D = CalibratedClassifierCV()
clf_E = LogisticRegression(random_state = 1)


clf_list = [clf_A,clf_B,clf_C,clf_D,clf_E]
for i in clf_list:
    train_predict(i, X_train, y_train, X_test, y_test)


## Here we will go with RandomForestClassifier
importance = clf_A.feature_importances_

indices = np.argsort(importance)[::-1]
std = np.std([clf_A.feature_importances_ for tree in clf_A.estimators_],
             axis=0)

print("Feature ranking:")
for f in range(X.shape[1]):
    print("{} feature {} ({})".format(f + 1, indices[f], importance[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importance[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


features_ranking = dict()

nb_rank = 10

for f in range(X.shape[1]):
    if X.columns[indices[f]].split('_', 1 )[0] in features_ranking.keys():
        features_ranking[X.columns[indices[f]].split('_', 1 )[0]] += clf_A.feature_importances_[indices[f]]
    else:
        features_ranking[X.columns[indices[f]].split('_', 1 )[0]] = clf_A.feature_importances_[indices[f]]
features_ranking= sorted(features_ranking.items(), key=lambda features_ranking:features_ranking[1], reverse = True)


# Print the feature ranking
print('Feature ranking:')
for i in range(nb_rank):
    print('%d. feature %s (%f)' % (i+1,features_ranking[i][0], features_ranking[i][1]))
