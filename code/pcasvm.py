import time
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report as cr
from myPCA import reduceDimensionPCA

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
	return clf

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	true = 0
	total = 0
	z = []
	for feature in featureVectors:
		if feature[-1] == predict(model, feature[:-1]):
			true += 1
		z = z + predict(model, feature[:-1]).astype(np.int).tolist()
		total += 1
	data = featureVectors[:,-1].flatten()
	data = data.astype(np.int).tolist()
	print cr(data, z)
	print "Accuracy:",
	print (true * 100) / total

def reduceDim(mat, k):
	pca = PCA(n_components = k)
	data = pca.fit_transform(mat[:, :-1])
	classes = mat[:, -1]
	newData = np.zeros((len(mat), k + 1))
	newData[:, :-1] = data
	newData[:, -1] = classes 
	return newData

file = open("kdd.data")
featureVectors = []
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.':
		vector[-1] = 0
	else:
		vector[-1] = 1
	featureVectors.append(vector)
N = 500
random.seed(N + 1)
random.shuffle(featureVectors)
k = 11
mat = np.array(featureVectors)[:N,:]
mat = encodeAll(mat)
t0 = time.clock()
newData = reduceDimensionPCA(mat, k)
t1 = time.clock()
trainData = mat[:N/2, :]
testData = mat[N/2:, :]
model = train(trainData[:, :-1], trainData[:, -1])
t2 = time.clock()
classify(model, testData)
t3 = time.clock()

print t1-t0, " time in PCA reduction"
print t2-t1, " time in training"
print t3-t2, " time in classification"
print t3-t0, " total time"
print "------------------------------------------\n"
