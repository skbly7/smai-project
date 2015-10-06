import time
import numpy as np
import matplotlib
matplotlib.get_cachedir()
import matplotlib.pyplot as plt
import random
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
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

def reduceDimFFR(mat, k):
	total_category = int(mat.max(axis=0)[-1])+1
	#overall_data = np.zeros(2,2))
	dict_feature = {}
	for i in range(total_category):
		group_by_func = mat[:, -1] == i
		cat_i_data = mat[group_by_func]
		sum_i = np.sum(cat_i_data, axis=0)
		s_d_with_class = np.divide(sum_i, len(cat_i_data))
		s_id = s_d_with_class[:-1]
		for d in range(len(s_id)):
			if d not in dict_feature:
				dict_feature[d] = []
	#		print s_id[d]
			dict_feature[d].append(s_id[d])
	variance = []
	for i in dict_feature:
		variance.append(np.var(np.array(dict_feature[i])))
	best_feature_index = np.array(variance).argsort()[:k].tolist()
	# adding back class labels to selected columns
	best_feature_index.append(-1)
	mat = mat[:,best_feature_index]
	return mat

file = open("kdd.data")
featureVectors = []
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.':
		vector[-1] = 0
	else:
		vector[-1] = 1
	featureVectors.append(vector)
N = 50000
random.seed(N + 1)
random.shuffle(featureVectors)
k = 11
mat = np.array(featureVectors)[:N,:]
mat = encodeAll(mat)
t0 = time.clock()
mat = reduceDimFFR(mat, k)
t1 = time.clock()
#newData = mat
trainData = mat[:N/2, :]
testData = mat[N/2:, :]
model = train(trainData[:, :-1], trainData[:, -1])
t2 = time.clock()
classify(model, testData)
t3 = time.clock()

print t1-t0, " time in FFR reduction"
print t2-t1, " time in training"
print t3-t2, " time in classification"
print t3-t0, " total time"
print "---------------------------------------\n"
