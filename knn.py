import numpy as np
#import matplotlib.pyplot as plt
import random
from sklearn import svm, preprocessing

def encode(vector):
	le = preprocessing.LabelEncoder()
	le.fit(vector)
	vector = le.transform(vector)
	return vector

def encodeAll(mat):
	categoricalIndices = [1,2,3]
	for i in categoricalIndices:
		mat[:,i] = encode(mat[:,i])
	return mat.astype(np.float)

def train(X, y):
	clf = svm.SVC(kernel='linear', C = 1.0, max_iter = -1)
	clf.fit(X, y)
	print "Train Done!"
	return clf

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	true = 0
	total = 0
	for feature in featureVectors:
		if feature[-1] == predict(model, feature[:-1]):
			true += 1
		total += 1
	print "Accuracy:",
	print (true * 100) / total

file = open("kdd.data")
featureVectors = []
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.':
		vector[-1] = 0
	else:
		vector[-1] = 1
	featureVectors.append(vector)
random.shuffle(featureVectors)
mat = np.array(featureVectors)[:10000,:]
mat = encodeAll(mat)
model = train(mat[:5000, :-1], mat[:5000, -1])
classify(model, mat[5000:, :])
