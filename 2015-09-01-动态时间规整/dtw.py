#coding=utf-8
import numpy as np

P = np.array([1, 2, 2, 3, 3, 4, 4])
Q = np.array([1, 2, 3, 4, 5])
Pm, Qm = np.meshgrid(Q, P)
subValue = Pm - Qm
PQ = np.multiply(subValue, subValue)

print "PQ:"

print np.flipud(PQ)


DTW = PQ.copy()
for row in xrange(1, np.size(PQ, 0)):
	DTW[row][0] += DTW[row-1][0]
for column in xrange(1, np.size(PQ, 1)):
	DTW[0][column] += DTW[0][column-1]

for row in xrange(1, np.size(PQ, 0)):
	for column in xrange(1, np.size(PQ, 1)):
		DTW[row][column] += min(DTW[row-1][column], DTW[row][column-1], DTW[row-1][column-1])

print ""
print "DTW:"
print np.flipud(DTW)