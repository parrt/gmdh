import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

def getdata(filename):
    with open(filename, "rb") as csvfile:
        f = csv.reader(csvfile, dialect='excel')
        X = []
        Y = []
        i = 0
        for row in f:
            if i == 0:
                headers = row
            else:
                Y.append(int(row[len(row) - 1]))
                row = [float(col) for col in row[0:-1]]
                X.append(np.array(row))
            i += 1
    return X, Y, headers

X, Y, headers = getdata("Training50_winedata.csv")
testX, testY, headers = getdata("Test50_winedata.csv")

N = len(X)
M = len(X[0])
X = np.array(X)
print "%d exemplars, %d variables" % (N, M)

clf = linear_model.Ridge(alpha=1.5)
fit = clf.fit(X,Y)
y_ = clf.predict(X)
print "Ridge R^2 score:", clf.score(X,Y)
num_correct = sum(round(a) == y for a, y in zip(y_, Y))
print "Ridge correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "Ridge test correct", num_correct

clf = KernelRidge(alpha=0.5)
clf.fit(X,Y)
y_ = clf.predict(X)
print "KernelRidge R^2 score:", clf.score(X,Y)
num_correct = sum(round(a) == y for a, y in zip(y_, Y))
print "KernelRidge correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "KernelRidge test correct", num_correct

clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True)
clf.fit(X, Y)
y_ = clf.predict(X)
print "SGD hinge R^2 score:", clf.score(X,Y)
num_correct = sum(pair[0]==pair[1] for pair in zip(y_, Y))
print "SGD hinge correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "SGD hinge test correct", num_correct

clf = KNeighborsClassifier(11, weights='distance')
clf.fit(X, Y)
y_ = clf.predict(X)
print "KNeighbors R^2 score:", clf.score(X,Y)
num_correct = sum(pair[0]==pair[1] for pair in zip(y_, Y))
print "KNeighbors correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "KNeighbors test correct", num_correct

clf = RandomForestClassifier(n_estimators=30)
clf.fit(X, Y)
y_ = clf.predict(X)
print "RandomForest R^2 score:", clf.score(X,Y)
num_correct = sum(pair[0]==pair[1] for pair in zip(y_, Y))
print "RandomForest correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "RandomForest test correct", num_correct

clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X, Y)
y_ = clf.predict(X)
print "GradientBoosting R^2 score:", clf.score(X,Y)
num_correct = sum(pair[0]==pair[1] for pair in zip(y_, Y))
print "GradientBoosting correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "GradientBoosting test correct", num_correct
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
print zip(headers,feature_importance)

clf = svm.SVC()
clf.fit(X, Y)
y_ = clf.predict(X)
print "SVM R^2 score:", clf.score(X,Y)
num_correct = sum(a == y for a, y in zip(y_, Y))
print "SVM correct", num_correct
y_ = clf.predict(testX)
num_correct = sum(round(a) == y for a, y in zip(y_, testY))
print "SVM test correct", num_correct
