# coding: utf-8
import numpy as np
import random
from scipy import stats
from config import *

# X_count = 5
# A_count = 4

data_path = '../ModelBasedLearning/data'
P = np.load(data_path + '/P.npy')
R = np.load(data_path + '/R.npy')
# V = np.zeros(X_count)
Pi = np.ones((X_count, A_count)) / float(A_count)
Q = np.zeros((X_count, A_count))
T = 20
greedy = 0.1

# 状态 -> 动作
def action_with_state(x_state):
	possibilitys = Pi[x_state]
	K_indexs = np.arange(K_count)
	custm = stats.rv_discrete(name='custm', values=(K_indexs, possibilitys))
	a = custm.rvs()
	return a

# 状态、动作 -> 状态
def state_transfer(x_from, a):
	transfer_possibilitys = P[x_from, a]
	K_indexs = np.arange(K_count)
	custm = stats.rv_discrete(name='custm', values=(K_indexs, transfer_possibilitys))
	x_to = custm.rvs()
	return x_to

# 状态、动作、状态 -> 奖赏
def reward(x_from, a, x_to):
	return R[x_from, a, x_to]

# 生成一条轨迹
def newTrace():
	x = np.zeros(T+1)
	a = np.zeros(T+1)
	r = np.zeros(T+1)
	x[0] = random.randint(0, X_count-1)	#初始状态随机
	for i in range(T):
		a[i] = action_with_state(x[i])
		x[i+1] = state_transfer(x[i], a[i])
		r[i+1] = reward(x[i], a[i], x[i+1])
	return (x, a, r)

def updateQi(x, a, r):
	for t in range(T):
		gain_reward = np.sum(r[t+1:]) / float(r[t+1:].shape[0])
		

def newValue():
	# print t
	Q = np.zeros((X_count, A_count))
	for x_from_index in range(X_count):
		for a_index in range(A_count):
			value = 0.0
			for x_to_index in range(X_count):
				possiblity_transfer = P[x_from_index, a_index, x_to_index]
				reward = R[x_from_index, a_index, x_to_index]
				one_value = possiblity_transfer * (reward / float(t) + (t - 1) * V[x_to_index] / float(t))
				# print t, possiblity_transfer, reward, V[x_to_index], one_value
				value += one_value
			# print value
			# print ''
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
	if t > 100:
		break





