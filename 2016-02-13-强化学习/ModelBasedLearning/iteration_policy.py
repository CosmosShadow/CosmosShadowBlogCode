# coding: utf-8
import numpy as np
import sys
import Array.Softmax as Softmax

model_path = 'model/watermelon'
sys.path.append(model_path)
from config import *
P = np.load(model_path + '/P.npy')
R = np.load(model_path + '/R.npy')


T = 20
V = np.zeros(X_count)
Pi = np.ones((X_count, A_count)) / float(A_count)

def newValue():
	Vtmp = np.zeros(X_count)
	for step in xrange(1, T+1):
		Vtmptmp = np.zeros(X_count)
		for x_from_index in range(X_count):
			value = 0.0
			for a_index in range(A_count):
				for x_to_index in range(X_count):
					possiblity_action = Pi[x_from_index, a_index]
					possiblity_transfer = P[x_from_index, a_index, x_to_index]
					reward = R[x_from_index, a_index, x_to_index]
					value += possiblity_action * possiblity_transfer * (reward / float(step) + ((step - 1) / float(step)) * Vtmp[x_to_index])
			Vtmptmp[x_from_index] = value
		Vtmp = Vtmptmp
	return Vtmp

def newPolicy():
	Q = np.zeros((X_count, A_count))
	for x_from_index in range(X_count):
		for a_index in range(A_count):
			value = 0.0
			for x_to_index in range(X_count):
				possiblity_transfer = P[x_from_index, a_index, x_to_index]
				reward = R[x_from_index, a_index, x_to_index]
				value += possiblity_transfer * (reward / float(T+1) + T * V[x_to_index] / float(T+1))
			Q[x_from_index, a_index] = value
	return Softmax.softmax(Q, 1)
	# # print Q
	# QMaxIndex = np.argmax(Q, axis=1)
	# Pitmp = np.zeros((X_count, A_count))
	# for i in range(X_count):
	# 	Pitmp[i, QMaxIndex[i]] = 1.0
	# return Pitmp

def purePolicy():
	return np.argmax(Pi, axis=1)

for i in range(100):
	Piclear = purePolicy()
	V = newValue()
	Pi = newPolicy()
	# print V
	print Pi
	if np.array_equal(Piclear, purePolicy()):
		# print V
		break





