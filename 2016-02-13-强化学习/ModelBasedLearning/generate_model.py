# coding: utf-8
import random
import numpy as np
from config import *
from File.FilePath import *

print 'config, X count: %d, A count: %d' %(X_count, A_count)
P = np.random.rand(X_count, X_count, A_count)
P[0, 1, 0] += 0.5
P[1, 1, 0] += 1.7
P[2, 0, 2] += 0.8
P[0, 0, 3] += 0.2
P[2, 0, 2] += 0.5
P[4, 3, 1] += 1.2
P = P / np.expand_dims(np.sum(P, axis=2), axis=2)
# print P
# print ''

R = np.random.randint(50, size=(X_count, X_count, A_count)) + 1.0
# print R
# print ''

data_path = 'data'
removeAndCreateDir(data_path)
np.save(data_path + '/P', P)
np.save(data_path + '/R', R)