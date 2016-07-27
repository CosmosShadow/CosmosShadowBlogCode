# coding: utf-8
import numpy as np
import random
from scipy import stats

import sys
model_path = '../ModelBasedLearning/model/watermelon'
sys.path.append(model_path)
from config import *

"""TDQ
Temporal Difference Q: 时序差分学习，同策略
"""

class TDSarsa(object):
	def __init__(self):
		self.X_count = X_count
		self.A_count = A_count
		self.P = np.load(model_path + '/P.npy')
		self.R = np.load(model_path + '/R.npy')
		self.Pi = np.ones((self.X_count, self.A_count)) / float(self.A_count)
		self.Q = np.zeros((self.X_count, self.A_count))
		self.T = 20
		self.greedy = 0.01
		self.alpha = 0.1

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

	def updatePi(self):
		self.Pi = np.ones((self.X_count, self.A_count)) * self.greedy / float(self.A_count)
		QMaxIndex = np.argmax(self.Q, axis=1)
		for i in range(self.X_count):
			self.Pi[i, QMaxIndex[i]] = self.Pi[i, QMaxIndex[i]] + 1.0 - self.greedy

	def walk(self):
		x = random.randint(0, self.X_count-1)	#初始状态随机
		for i in range(2000):
			# 前进一步
			a = self.action_with_state(x)
			x_new = self.state_transfer(x, a)
			r = self.reward(x, a, x_new)
			a_new = self.action_with_state(x_new)
			# 更新状态-动作值
			self.Q[x, a] = self.Q[x, a] + self.alpha * (r + self.greedy * self.Q[x_new, a_new] - self.Q[x, a])
			# 更新策略
			self.updatePi()
			# 更新状态
			x = x_new
			# 输出
			print self.purePolicy()
			# print self.Q

	def purePolicy(self):
		return np.argmax(self.Pi, axis=1)

if __name__ == '__main__':
	learning = TDSarsa()
	learning.walk()





