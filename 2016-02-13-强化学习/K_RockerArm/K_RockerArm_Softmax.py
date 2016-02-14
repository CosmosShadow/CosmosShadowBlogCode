# coding: utf-8
from K_RockerArm_Reward import *
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class KRockerArm_Softmax(object):
	
	def __init__(self, K_count, temperature, walk_count):
		self.K_count = K_count
		self.temperature = temperature
		self.walk_count = walk_count
		self.r = 0.0
		self.r_array = np.zeros(walk_count)
		self.Q = np.zeros(K_count)
		self.count = np.zeros(K_count)

	def select_one_K(self):
		# 计算概率分布
		exp_parameter = self.Q / self.temperature
		max_exp_parameter = np.max(exp_parameter)
		normal_exp_parameter = exp_parameter - max_exp_parameter
		energy = np.exp(normal_exp_parameter)
		energy_sum = np.sum(energy)
		possibility = energy / energy_sum
		# 根据概率分布，返回随机值
		K_indexs = np.arange(self.K_count) + 1
		custm = stats.rv_discrete(name='custm', values=(K_indexs, possibility))
		return custm.rvs()

	# EE: Exploration-Exploitation 探索利用
	def EE(self):
		for i in range(self.walk_count):
			K_index = self.select_one_K()
			v = K_RockerArm_Reward(K_index)
			value_index = K_index - 1
			self.r += v
			self.Q[value_index] = (self.Q[value_index] * self.count[value_index] + v) / (self.count[value_index] + 1)
			self.count[value_index] += 1
			self.r_array[i] = self.r / (i + 1.0)

def test():
	walk_count = 3000
	greedy0 = KRockerArm_Softmax(2, 0.001, walk_count)
	greedy0.EE()
	greedy1 = KRockerArm_Softmax(2, 0.01, walk_count)
	greedy1.EE()
	greedy2 = KRockerArm_Softmax(2, 0.1, walk_count)
	greedy2.EE()
	greedy3 = KRockerArm_Softmax(2, 1.0, walk_count)
	greedy3.EE()

	plt.plot(np.arange(walk_count), greedy0.r_array, label="0.001", color="blue")
	plt.plot(np.arange(walk_count), greedy1.r_array, label="0.01", color="green")
	plt.plot(np.arange(walk_count), greedy2.r_array, label="0.1", color="black")
	plt.plot(np.arange(walk_count), greedy3.r_array, label="1.0", color="red")
	# plt.plot(x, z, label="$cos(x)$")
	plt.xlabel("Time(s)")
	plt.ylabel("Average Reward")
	plt.title("K Rocker Arm with Softmax")

	plt.legend()     #显示图示
	plt.show()

if __name__ == '__main__':
	test()
