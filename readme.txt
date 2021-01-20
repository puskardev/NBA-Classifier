Submitted by : Puskar Dev
Student Id : 1001516630
Programming Assignment 2
Submitted Date : 11/17/2020

Steps performed in order to increase the accuracy are given as:

1) The orginal NBA.csv dataset contained 475 records. In order, to increase the accuracy for the model, preprocessing was done 
on the original dataset to select records where players with Games started and Minutes played are greater than 1.
(This preprocessing was done in order to reduce prediction mistakes on bench players and players who has playd less minutes.)

2) The orginal sample code contained 26 features. Based on how basketball satistics work, I removed some of the features from the
features columns. Only the following features were selected for training the model:
'FT', 'FG%', '3P', '3PA', '3P%', '2P%','2P', 'eFG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK'

3) The sample code given used K-Neighbors Classifier to build the model. However, in this implementation linear support vectore machine is used,
which gave greater accuracy compared to other classifier.
(dual=False) condiition is used while building linear SVM.

4) On 75-25% split, the test score accuracy was calculated as 0.667 in the first run.
The average cross-validation score for 10-fold stratified cross-validation was 0.67.
