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
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.metrics import classification_report as cr
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
from itertools import product

def encode(vector):
	le = preprocessing.LabelEncoder()
	le.fit(vector)
	vector = le.transform(vector)
	return vector

def encodeAll(mat):
	categoricalIndices = [1,2,3,-1]
	for i in categoricalIndices:
		mat[:,i] = encode(mat[:,i])
	return mat.astype(np.float)

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	true = 0
	total = 0
	z = []
	misclassified = 0
	for feature in featureVectors:
		if feature[-1] == predict(model, feature[:-1]):
			true += 1
		if feature[-1] != predict(model, feature[:-1]) and feature[-1] == 4:
			misclassified += 1
		z = z + predict(model, feature[:-1]).astype(np.int).tolist()
		total += 1
	data = featureVectors[:,-1].flatten()
	data = data.astype(np.int).tolist()
	print misclassified
	print cr(data, z )
	print "Accuracy:",
	print (true * 100) / total

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, clfs, voting='hard', weights=None):

        self.clfs = clfs
        self.named_clfs = {key:value for key,value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'\
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = []
        for clf in self.clfs:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.clfs_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])

file = open("kdd.data")
featureVectors = []
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.':
		vector[-1] = 'Normal'
	elif vector[-1] in ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.']:
		vector[-1] = 'R2L'
	elif vector[-1] in ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']:
		vector[-1] = 'U2R'
	elif vector[-1] in ['ipsweep.', 'nmap.', 'portsweep.', 'satan.']:
		vector[-1] = 'Probing'
	elif vector[-1] in ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.']:
		vector[-1] = 'DOS'
	featureVectors.append(vector)

random.seed(170)
N = 250000
M = 250000
random.shuffle(featureVectors)
mat = np.array(featureVectors[:])[:,:]
mat = encodeAll(mat)

X = mat[:N, :-1]
y = mat[:N, -1]
test = mat[N:N+M, :]
print test.shape
clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')
eclf.fit(X, y)
# eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[2,1,5])eclf.fit(X,y)
classify(eclf, test)



