"""

A basic example of the use of ROCCH.

Class counts of covtype:
for val in np.unique(dataset.target):
    print(val, sum(dataset.target==val))

1 211840
2 283301
3 35754
4 2747
5 9493
6 17367
7 20510


"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from pycost import ROCCH


class_transform = {}
for i in (1, 3, 6):
    class_transform[i] = 1
for i in (2, 4, 5, 7):
    class_transform[i] = 2


def fetch_covertype_binary():
    covtype = fetch_covtype(shuffle=True)
    # transform to binary
    binary_target = np.array([class_transform[c] for c in covtype.target])
    covtype.target = binary_target
    return covtype


DATA_LIMIT = 100_000

def main(args):
    rocch = ROCCH()
    covtype = fetch_covertype_binary()
    dtree = DecisionTreeClassifier(min_samples_leaf=4)
    X, y = covtype.data[:DATA_LIMIT], covtype.target[:DATA_LIMIT]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    dtree.fit(X_train, y_train)
    y_scored = dtree.predict_proba(X_test)[:,0]
    (fpr, tpr, thresholds) = roc_curve(y_test, y_scored, pos_label=1)
    plt.plot(fpr, tpr, 'k')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_y_scored = gnb.predict_proba(X_test)[:,0]
    (gnb_fpr, gnb_tpr, gnb_thresholds) = roc_curve(y_test, gnb_y_scored, pos_label=1)
    plt.plot(gnb_fpr, gnb_tpr, 'r')
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )


