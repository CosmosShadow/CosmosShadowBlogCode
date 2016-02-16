# coding: utf-8
import numpy as np
from config import *

# X_count = 5
# A_count = 4

data_path = 'data'
P = np.load(data_path + '/P.npy')
R = np.load(data_path + '/R.npy')
V = np.zeros(X_count)

t = 1

def newValue():
	print t
	Q = np.zeros((X_count, A_count))
	for x_from_index in range(X_count):
		for a_index in range(A_count):
			print ''
			value = 0.0
			for x_to_index in range(X_count):
				possiblity_transfer = P[x_from_index, x_to_index, a_index]
				reward = R[x_from_index, x_to_index, a_index]
				one_value = possiblity_transfer * (reward / float(t) + (t - 1) * V[x_to_index] / float(t))
				print t, possiblity_transfer, reward, V[x_to_index], one_value
				value += one_value
			print value
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
				possiblity_transfer = P[x_from_index, x_to_index, a_index]
				reward = R[x_from_index, x_to_index, a_index]
				value += possiblity_transfer * (reward / float(t+1) + t * V[x_to_index] / float(t+1))
			Q[x_from_index, a_index] = value
	# print Q
	QMaxIndex = np.argmax(Q, axis=1)
	Pitmp = np.zeros((X_count, A_count))
	for i in range(X_count):
		Pitmp[i, QMaxIndex[i]] = 1.0
	return Pitmp

def purePolicy(Pi):
	return np.argmax(Pi, axis=1)

while True:
	V2 = newValue()
	Pi = thePolicy()
	clear_policy = purePolicy(Pi)
	print clear_policy
	# print V2
	value_distance = valueDistance(V, V2)
	# print value_distance
	if valueDistance < 0.01:
		break
	V = V2
	t += 1
	if t > 10:
		break





