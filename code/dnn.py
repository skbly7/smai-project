import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix
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
from sklearn.metrics import confusion_matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def train(X, ty):
	nn = Classifier(layers=[Layer("Sigmoid", units = 25), Layer("Sigmoid", units = 5)], learning_rate = 0.001, n_iter = 1000, verbose = 1)
	nn.fit(X, ty)
	print "Train Done!"
	return nn

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	z = model.predict(featureVectors[:, :-1]).astype(np.int).reshape(-1).tolist()
	data = featureVectors[:,-1].flatten()
	data = data.astype(np.int).tolist()
	labels = ['DOS', 'Normal', 'Probing', 'R2L', 'U2R']
	print cr(data, z, target_names=labels, digits = 4)
	cm = confusion_matrix(data, z)
	print_cm(cm, labels)

file = open("kdd.data")
featureVectors = []
normal_count = 0
r_count = 0
u_count = 0
prob_count = 0
dos_count = 0
limit = 1000
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.' and normal_count < limit:
		vector[-1] = 'Normal'
		featureVectors.append(vector)
		normal_count += 1
	elif vector[-1] in ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'] and r_count < limit:
		vector[-1] = 'R2L'
		featureVectors.append(vector)
		r_count += 1
	elif vector[-1] in ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'] and u_count < limit:
		vector[-1] = 'U2R'
		featureVectors.append(vector)
		u_count += 1
	elif vector[-1] in ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'] and prob_count < limit:
		vector[-1] = 'Probing'
		featureVectors.append(vector)
		prob_count += 1
	elif vector[-1] in ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'] and dos_count < limit:
		vector[-1] = 'DOS'
		featureVectors.append(vector)
		dos_count += 1
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

random.shuffle(featureVectors)
limit = (50 * len(featureVectors[:])) / 100
print limit
mat = np.array(featureVectors[:])[:,:]
mat = encodeAll(mat)
model = train(mat[:limit, :-1], mat[:limit, -1])
classify(model, mat[limit:, :])