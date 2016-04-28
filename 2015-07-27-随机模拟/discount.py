#coding: utf-8

import random

def random_discount(cut_point, below_possibility):
	sample_below_possibility = cut_point / 1.0
	sample_above_possibility = (1 - cut_point) / 1.0
	above_possibility = 1 - below_possibility

	M = max(below_possibility / sample_below_possibility, above_possibility / sample_above_possibility)

	while True:
		discount = random.randint(1, 100)
		possibility = random.random()
		if discount < cut_point * 100:
			if possibility < below_possibility / (M * sample_below_possibility):
				return discount / 100.0
		if discount >= cut_point * 100:
			if possibility < above_possibility / (M * sample_above_possibility):
				return discount / 100.0

if __name__ == '__main__':
	cut_point = 0.2
	below_possibility = 0.8

	below_count = 0
	above_count = 0
	count = 100000
	for x in range(count):
		discount = random_discount(cut_point, below_possibility)
		if discount < 0.01 or discount > 1:
			print 'error: out of range'
		if discount < cut_point:
			below_count += 1
		else:
			above_count += 1

	print '< ', cut_point, ': ', below_count/float(count)
	print '> ', cut_point, ': ', above_count/float(count)
