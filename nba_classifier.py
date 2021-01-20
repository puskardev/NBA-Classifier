# Puskar Dev
# 1001516630
# Programming Assignment 2
# Date of submission -> 11/17/2020

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# store the orginial dataset column names.
original_headers = list(nba.columns.values)

#In order for the model to reduce the mistakes on bench players and players who has played less minutes, the following
#conditions are applied to original dataset and select only those records that apply.
nba = nba[(nba['MP'] > 1) & (nba['GS'] > 1)]

# target clas-variable
class_column = 'Pos'

#The orginal dataset contains attributes such as player name and team name which are not important at all. so it is removed.
#Besides those other attributes are removed as well. The following attributes are selected.
feature_columns = ['FT', 'FG%', '3P', '3PA', '3P%', '2P%','2P', 'eFG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK']

#Pandas DataFrame allows you to select columns.
#We use column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

#train-feature has 75% data of nba_feature
#test_feature has 25% data of nba_feature
train_feature, test_feature, train_class, test_class = train_test_split(nba_feature, nba_class, stratify=nba_class, train_size=0.75, test_size=0.25)

# Task 1)
# Linear support vector machines are used to build a model to fit the training data from above.
linearsvm = LinearSVC(dual=False).fit(train_feature, train_class)

#Task 2)
print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))
prediction = linearsvm.predict(test_feature)

# Task 3)
# Confusion matrix
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Task 4)
# apply 10-fold stratified cross-validation instead of 75-25% split. scores are stored in a list scores.
scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10)

#Task 5)
# Print out the accuracy of each fold in 4).
print("Cross-validation scores: {}".format(scores))

#Task 6)
# Print out the average accuracy across all the folds in 4).
print("Average cross-validation score: {:.2f}".format(scores.mean()))
