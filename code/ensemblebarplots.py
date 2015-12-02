from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.metrics import classification_report as cr
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
from itertools import product

m = np.zeros((6))
m[0] = 25
m[1] = 25
m[2] = 25
m[3] = 10
m[4] = 10
m[5] = 7
N = 6  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(7,5))

p1 = ax.bar(ind, np.hstack(([m[:-1],[0]])), width, color='blue')

p2 = ax.bar(ind, [0, 0, 0, 0, 0, m[-1]], width, color='green')

plt.axvline(4.65, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['SVM', 
					'Neural Network',
					'GaussianNaiveBayes',
					'DecisionTreeClassifier',
					'RandomForestClassifier',
					'EnsembleClassifier',
					],
				   rotation=40,
				   ha='right')
plt.ylim([0,25])
plt.ylabel('U2R Misclassified Samples')
plt.title('U2R Classification')
plt.tight_layout()
plt.show()




