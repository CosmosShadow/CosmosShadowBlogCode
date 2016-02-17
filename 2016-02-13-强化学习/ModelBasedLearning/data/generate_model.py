# coding: utf-8
import random
import numpy as np
from config import *
from File.FilePath import *

print 'config, X count: %d, A count: %d' %(X_count, A_count)
P = np.random.rand(X_count, A_count, X_count)
P = P / np.expand_dims(np.sum(P, axis=2), axis=2)
# print P
# print ''

R = np.random.randint(5, size=(X_count, A_count, X_count)) + 1.0
# print R
# print ''

np.save('P', P)
np.save('R', R)