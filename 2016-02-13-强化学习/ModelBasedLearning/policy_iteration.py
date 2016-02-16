# coding: utf-8
import numpy as np
from config import *

# X_count = 5
# A_count = 4

data_path = 'data'
P = np.load(data_path + '/P.npy')
R = np.load(data_path + '/R.npy')
T = 10

V = np.zeros(X_count)
Pi = np.ones((X_count, A_count)) / float(A_count)

def newValue():
	Vtmp = np.zeros(X_count)
	for i in xrange(1, T+1):
		for x_from_index in xrange(X_count):
			value = 0.0
			for a_index in range(A_count):
				for x_to_index in range(X_count):
					possiblity_action = Pi[x_from_index, a_index]
					possiblity_transfer = P[x_from_index, x_to_index, a_index]
					reward = R[x_from_index, x_to_index, a_index]
					

def newPolicy():
	pass

print V
print Pi



