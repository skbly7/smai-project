import numpy as np
import time
import random
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report as cr

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
	clf = svm.SVC(kernel='linear', C = 1.0, max_iter = 100000)
	clf.fit(X, y)
	print "Train Done!"
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
	print cr(data, z )
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
print "Simple SVM on 41 features"
N = 50000
print "Samples:",
print N
random.seed(N + 1)
random.shuffle(featureVectors)
mat = np.array(featureVectors)[:N,:]
mat = encodeAll(mat)
t0 = time.clock()
model = train(mat[:N/2, :-1], mat[:N/2, -1])
t1 = time.clock()
classify(model, mat[N/2:, :])
t2 = time.clock()

print t1-t0, "Time in Training"
print t2-t1, "Time in Classifying"
print "---------------------------------------\n"
