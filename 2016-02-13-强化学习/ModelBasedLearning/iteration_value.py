# coding: utf-8
import numpy as np

import sys
model_path = 'model/watermelon'
sys.path.append(model_path)
from config import *
P = np.load(model_path + '/P.npy')
R = np.load(model_path + '/R.npy')

V = np.zeros(X_count)

t = 1
gamma = 0.9

def newValue():
	# print t
	Q = np.zeros((X_count, A_count))
	for x_from_index in range(X_count):
		for a_index in range(A_count):
			value = 0.0
			for x_to_index in range(X_count):
				possiblity_transfer = P[x_from_index, a_index, x_to_index]
				reward = R[x_from_index, a_index, x_to_index]
				one_value = possiblity_transfer * (reward + gamma * V[x_to_index])
				value += one_value
			Q[x_from_index, a_index] = value
	Vtmp = np.max(Q, axis=1)
	return Vtmp

# 给出两个V的相差
def valueDistance(V1, V2):
	distance = 0.0
	for i in range(V1.shape[0]):
		local_distance = abs(V1[i] - V2[i])
		if local_distance > distance:
			distance = local_distance
	return local_distance

def thePolicy():
	Q = np.zeros((X_count, A_count))
	for x_from_index in range(X_count):
		for a_index in range(A_count):
			value = 0.0
			for x_to_index in range(X_count):
				possiblity_transfer = P[x_from_index, a_index, x_to_index]
				reward = R[x_from_index, a_index, x_to_index]
				value += possiblity_transfer * (reward + gamma * V[x_to_index])
			Q[x_from_index, a_index] = value
	print 
	print 'Q: '
	print Q
	QMaxIndex = np.argmax(Q, axis=1)
	Pitmp = np.zeros((X_count, A_count))
	for i in range(X_count):
		Pitmp[i, QMaxIndex[i]] = 1.0
	return Pitmp

def purePolicy(Pi):
	return np.argmax(Pi, axis=1)

while True:
	V2 = newValue()
	value_distance = valueDistance(V, V2)
	if value_distance < 0.0001:
		Pi = thePolicy()
		clear_policy = purePolicy(Pi)
		print ''
		print 'result:'
		print clear_policy
		print V2
		break
	V = V2
	print t, value_distance
	t += 1
	if t > 1000:
		break





