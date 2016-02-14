# coding: utf-8
import random

def K_RockerArm_Reward(k):
	if k == 1 and random.random() < 0.4:
		return 1
	if k == 2 and random.random() < 0.2:
		return 1
	return 0

if __name__ == '__main__':
	for i in xrange(1,10):
		print K_RockerArm_Reward(1)
	print ''
	for i in xrange(1,10):
		print K_RockerArm_Reward(2)
	print ''
	for i in xrange(1,10):
		print K_RockerArm_Reward(3)