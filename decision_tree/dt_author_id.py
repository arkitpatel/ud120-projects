#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=40)
t = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t, 3), "s"
t = time()
prediction = clf.predict(features_test)
print "prediction time:", round(time()-t, 3), "s"
accuracy = accuracy_score(labels_test, prediction)
print accuracy
print len(features_train[0])
#########################################################


