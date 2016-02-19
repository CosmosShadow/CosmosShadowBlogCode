# coding: utf-8
import numpy as np
import random
from scipy import stats

import sys
# model_path = '../ModelBasedLearning/model/watermelon'
model_path = '../ModelBasedLearning/model/random_model'
sys.path.append(model_path)
from config import *

class MonteCarloOffPolicy(object):
	def __init__(self):
		self.X_count = X_count
		self.A_count = A_count
		self.P = np.load(model_path + '/P.npy')
		self.R = np.load(model_path + '/R.npy')
		self.Pi = np.ones((self.X_count, self.A_count)) / float(self.A_count)
		self.Pi_greedy = np.ones((self.X_count, self.A_count)) / float(self.A_count)	#策略对应的贪心策略
		self.Q = np.zeros((self.X_count, self.A_count))
		self.Q_count = np.zeros((self.X_count, self.A_count))
		self.T = 20
		self.greedy = 0.1

	# 状态 -> 动作
	def action_with_state(self, x_state):
		possibilitys = self.Pi_greedy[x_state]
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
		# 轨迹由贪心策略生成，映射到正常策略上，使用重要性采样来做
		p_radio = np.zeros(self.T)
		for i in range(self.T):
			p_radio[i] = self.Pi[x[i], a[i]] / self.Pi_greedy[x[i], a[i]]
		# # 初始的公式
		# for i in xrange(1, self.T):
		# 	r[i] *= np.prod(p_radio[i:])
		return (x, a, r, p_radio)

	def updateQi(self, x, a, r, p_radio):
		for t in range(self.T):
			# gain_reward = np.sum(r[t+1:]) / float(r[t+1:].shape[0])							#初始的公式
			# gain_reward = np.sum(r[t+1:]) * np.prod(p_radio[t:]) / float(r[t+1:].shape[0])	#周老师最新公式
			gain_reward = np.sum(r[t+1:]) * np.prod(p_radio[t+1:]) / float(r[t+1:].shape[0])	#我认为在周老师公式上起码要改动的
			self.Q[x[t], a[t]] = ((self.Q[x[t], a[t]] * self.Q_count[x[t], a[t]]) + gain_reward) / float(self.Q_count[x[t], a[t]] + 1)	#更新状态-动作值函数Q
			self.Q_count[x[t], a[t]] += 1	#更新计数

	def updatePi(self):
		self.Pi = np.zeros((self.X_count, self.A_count))
		self.Pi_greedy = np.ones((self.X_count, self.A_count)) * self.greedy / float(self.A_count)
		QMaxIndex = np.argmax(self.Q, axis=1)
		# print np.max(self.Q, axis=1)
		for i in range(self.X_count):
			self.Pi[i, QMaxIndex[i]] += 1.0
			self.Pi_greedy[i, QMaxIndex[i]] += 1.0 - self.greedy

	def purePolicy(self):
		return np.argmax(self.Pi, axis=1)

	def test(self):
		for j in xrange(5000):
			print self.purePolicy()
			# if j%20 == 0:
			# 	print self.Q
			(x, a, r, p_radio) = self.newTrace()
			self.updateQi(x, a, r, p_radio)
			self.updatePi()
			# print Pi

if __name__ == '__main__':
	learning = MonteCarloOffPolicy()
	learning.test()





