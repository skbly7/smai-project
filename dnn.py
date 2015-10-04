import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm, preprocessing
from sknn.mlp import Classifier, Layer
import sys
import logging
from sklearn.metrics import classification_report as cr

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

def train(X, ty):
	nn = Classifier(layers=[Layer("Sigmoid", units=5000), Layer("Sigmoid", units = 5)], learning_rate = 0.001, n_iter = 100, verbose = 1)
	nn.fit(X, ty)
	print "Train Done!"
	return nn

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	z = model.predict(featureVectors[:, :-1]).astype(np.int).reshape(-1).tolist()
	data = featureVectors[:,-1].flatten()
	data = data.astype(np.int).tolist()
	print cr(data, z, target_names=['DOS', 'Normal', 'Probing', 'R2L', 'U2R'], digits = 4)

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
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

random.shuffle(featureVectors)
limit = (50 * len(featureVectors[:50000])) / 100
print limit
mat = np.array(featureVectors[:50000])[:,:]
mat = encodeAll(mat)
model = train(mat[:limit, :-1], mat[:limit, -1])
# print model.predict(mat[1, :-1])
# print model
# print model.predict(mat[:10, :-1])
classify(model, mat[limit:, :])