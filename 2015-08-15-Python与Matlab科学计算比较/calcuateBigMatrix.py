#coding=utf-8
import numpy as np
import time


start = time.clock()

matrixRow = 1000
matrixColumn = 2000
a = np.arange(matrixRow*matrixColumn).reshape(matrixRow, matrixColumn)
b = np.arange(matrixRow*matrixColumn).reshape(matrixColumn, matrixRow)
c = np.dot(a, b)
print c.shape

end = time.clock()
print "spend time: %f s" % (end - start)