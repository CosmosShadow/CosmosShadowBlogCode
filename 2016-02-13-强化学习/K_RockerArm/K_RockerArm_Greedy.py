# coding: utf-8
from K_RockerArm_Reward import *
import random
import numpy as np
import matplotlib.pyplot as plt

class KRockerArm_Greedy(object):
	
	def __init__(self, K_count, greedy_parameter, walk_count):
		self.K_count = K_count
		self.greedy_parameter = greedy_parameter
		self.walk_count = walk_count
		self.r = 0.0
		self.r_array = np.zeros(walk_count)
		self.Q = np.zeros(K_count)
		self.count = np.zeros(K_count)

	def select_one_K(self):
		if random.random() < self.greedy_parameter:
			return random.randint(1, self.K_count)
		else:
			# 如果值一样，则随机选择
			max_value = np.max(self.Q)
			max_indexs = []
			for i in range(self.K_count):
				if max_value == self.Q[i]:
					max_indexs.append(i+1)
			return max_indexs[random.randint(0, len(max_indexs)-1)]

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

if __name__ == '__main__':
	walk_count = 3000
	greedy0 = KRockerArm_Greedy(2, 0.01, walk_count)
	greedy0.EE()
	greedy1 = KRockerArm_Greedy(2, 0.1, walk_count)
	greedy1.EE()
	greedy2 = KRockerArm_Greedy(2, 0.5, walk_count)
	greedy2.EE()
	greedy3 = KRockerArm_Greedy(2, 0.99, walk_count)
	greedy3.EE()

	plt.plot(np.arange(walk_count), greedy0.r_array, label="0.01", color="blue")
	plt.plot(np.arange(walk_count), greedy1.r_array, label="0.1", color="green")
	plt.plot(np.arange(walk_count), greedy2.r_array, label="0.5", color="black")
	plt.plot(np.arange(walk_count), greedy3.r_array, label="0.99", color="red")
	# plt.plot(x, z, label="$cos(x)$")
	plt.xlabel("Time(s)")
	plt.ylabel("Average Reward")
	plt.title("K Rocker Arm with Greedy")

	plt.legend()     #显示图示
	plt.show()

