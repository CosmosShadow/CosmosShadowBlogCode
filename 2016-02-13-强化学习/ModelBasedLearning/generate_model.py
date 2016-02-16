# coding: utf-8
import random
import numpy as np
from config import *
from File.FilePath import *

print 'config, X count: %d, A count: %d' %(X_count, A_count)
P = np.random.rand(X_count, X_count, A_count)
P = P / np.expand_dims(np.sum(P, axis=2), axis=2)
print P
print ''

R = np.random.randint(5, size=(X_count, X_count, A_count)) + 1.0
R[0, 0, 0] += 5
R[1, 1, 1] += 10
R[2, 2, 2] += 15
print R
print ''

data_path = 'data'
removeAndCreateDir(data_path)
np.save(data_path + '/P', P)
np.save(data_path + '/R', R)