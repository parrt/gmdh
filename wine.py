import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import cPickle, gzip

# data from https://onlinecourses.science.psu.edu/stat857/node/223
# paper http://www.sciencedirect.com/science/article/pii/S0167923609001377
"""
Two datasets are available of which one dataset is on red wine and
have 1599 different varieties and the other is on white wine and have
4898 varieties. Only white wine data is analysed. All wines are
produced in a particular area of Portugal. Data are collected on 12
different properties of the wines one of which is Quality, based on
sensory data, and the rest are on chemical properties of the wines
including density, acidity, alcohol content etc. All chemical
properties of wines are continuous variables. Quality is an ordinal
variable with possible ranking from 1 (worst) to 10 (best). Each
variety of wine is tasted by three independent tasters and the final
rank assigned is the median rank given by the tasters.
"""
with open("Training50_winedata.csv", "rb") as csvfile:
    f = csv.reader(csvfile, dialect='excel')
    X = []
    Y = []
    i = 0
    for row in f:
        if i == 0:
            headers = row
        else:
            Y.append(int(row[len(row)-1]))
            row = [float(col) for col in row[0:-1]]
            X.append(np.array(row))
        i += 1

print headers
N = len(X)
M = len(X[0])
X = np.array(X)
print X
print Y
print N, M

clf = linear_model.Ridge(alpha=0.5)
fit = clf.fit(X,Y)
# print fit
# print fit.coef_
y_ = clf.predict(X)
# print zip(Y,y_)
print "R^2 score:", clf.score(X,Y)
num_correct = sum(int(round(a) == y) for a, y in zip(y_, Y))
print "Ridge correct", num_correct


clf = SGDClassifier(loss="log", penalty="l2", shuffle=True)
fit = clf.fit(X, Y)
# print fit
num_mis = sum([pair[0]!=pair[1] for pair in zip(clf.predict(X), Y)])
print "SGD correct", len(Y)-num_mis, "out of", len(Y)

clf = svm.SVC()
clf.fit(X, Y)
# test
predictions = [int(a) for a in clf.predict(X)]
num_correct = sum(int(a == y) for a, y in zip(predictions, Y))
print "SVM %s of %s values correct." % (num_correct, len(Y))
