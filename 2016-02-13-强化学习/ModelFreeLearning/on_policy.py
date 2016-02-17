# coding: utf-8
import numpy as np
import random
from scipy import stats

import sys
model_path = '../ModelBasedLearning/model/watermelon'
sys.path.append(model_path)
from config import *

class MonteCarloOnPolicy(object):
	def __init__(self):
		self.X_count = X_count
		self.A_count = A_count
		self.P = np.load(model_path + '/P.npy')
		self.R = np.load(model_path + '/R.npy')
		self.Pi = np.ones((self.X_count, self.A_count)) / float(self.A_count)
		self.Q = np.zeros((self.X_count, self.A_count))
		self.Q_count = np.zeros((self.X_count, self.A_count))
		self.T = 20
		self.greedy = 0.01

	# 状态 -> 动作
	def action_with_state(self, x_state):
		possibilitys = self.Pi[x_state]
		A_indexs = np.arange(self.A_count)
		custm = stats.rv_discrete(name='custm', values=(A_indexs, possibilitys))
		a = custm.rvs()
		return a

	# 状态、动作 -> 状态
	def state_transfer(self, x_from, a):
		transfer_possibilitys = self.P[x_from, a]
		X_indexs = np.arange(self.X_count)
		custm = stats.rv_discrete(name='custm', values=(X_indexs, transfer_possibilitys))
		x_to = custm.rvs()
		return x_to

	# 状态、动作、状态 -> 奖赏
	def reward(self, x_from, a, x_to):
		return self.R[x_from, a, x_to]

	# 生成一条轨迹
	def newTrace(self):
		x = np.zeros(self.T+1)
		a = np.zeros(self.T+1)
		r = np.zeros(self.T+1)
		x[0] = random.randint(0, self.X_count-1)	#初始状态随机
		for i in range(self.T):
			a[i] = self.action_with_state(x[i])
			x[i+1] = self.state_transfer(x[i], a[i])
			r[i+1] = self.reward(x[i], a[i], x[i+1])
		return (x, a, r)

	def updateQi(self, x, a, r):
		for t in range(self.T):
			gain_reward = np.sum(r[t+1:]) / float(r[t+1:].shape[0])	#后续的平均奖赏
			self.Q[x[t], a[t]] = ((self.Q[x[t], a[t]] * self.Q_count[x[t], a[t]]) + gain_reward) / float(self.Q_count[x[t], a[t]] + 1)	#更新状态-动作值函数Q
			self.Q_count[x[t], a[t]] += 1	#更新计数

	def updatePi(self):
		self.Pi = np.ones((self.X_count, self.A_count)) * self.greedy / float(self.A_count)
		QMaxIndex = np.argmax(self.Q, axis=1)
		for i in range(self.X_count):
			self.Pi[i, QMaxIndex[i]] = self.Pi[i, QMaxIndex[i]] + 1.0 - self.greedy

	def purePolicy(self):
		return np.argmax(self.Pi, axis=1)

	def clearHistory(self):
		self.Q = np.zeros((self.X_count, self.A_count))
		self.Q_count = np.zeros((self.X_count, self.A_count))

	def test(self):
		for j in xrange(1000):
			print self.purePolicy()
			(x, a, r) = self.newTrace()
			self.updateQi(x, a, r)
			self.updatePi()
			# print Pi

if __name__ == '__main__':
	learning = MonteCarloOnPolicy()
	learning.test()





