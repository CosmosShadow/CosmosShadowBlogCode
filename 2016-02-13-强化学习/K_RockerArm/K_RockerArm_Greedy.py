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
		self.Q = np.zeros(self.K_count)
		self.count = np.zeros(K_count)

	def select_one_K(self):
		if random.random() < self.greedy_parameter:
			return random.randint(1, self.K_count)
		else:
			epsilon = 1e-9
			shift_Q = self.Q + np.random.uniform(-epsilon, epsilon, size=self.K_count)	#解决值一样，总先最前面的问题
			return np.argmax(shift_Q) + 1

	def update_Q(self, index, r):
		index -= 1
		self.Q[index] = (self.Q[index] * self.count[index] + r) / (self.count[index] + 1)
		self.count[index] += 1

	# EE: Exploration-Exploitation 探索利用
	def EE(self):
		for i in range(self.walk_count):
			K_index = self.select_one_K()
			r = K_RockerArm_Reward(K_index)
			self.update_Q(K_index, r)
			self.r += r
			self.r_array[i] = self.r / (i + 1.0)

if __name__ == '__main__':
	walk_count = 30000
	greedy_arr = [0.01, 0.1, 0.5, 0.99]
	colors_arr = ['blue', 'green', 'black', 'red']
	for i in xrange(len(greedy_arr)):
		greedy0 = KRockerArm_Greedy(2, greedy_arr[i], walk_count)
		greedy0.EE()
		plt.plot(np.arange(walk_count), greedy0.r_array, label=str(greedy_arr[i]), color=colors_arr[i])
	# plt.plot(x, z, label="$cos(x)$")
	plt.xlabel("Time(s)")
	plt.ylabel("Average Reward")
	plt.title("K Rocker Arm with Greedy")

	plt.legend()     #显示图示
	plt.show()

