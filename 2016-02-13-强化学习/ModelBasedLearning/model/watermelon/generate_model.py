# coding: utf-8
import random
import numpy as np
from config import *
from File.FilePath import *

print 'config, X count: %d, A count: %d' %(X_count, A_count)
P = np.zeros((X_count, A_count, X_count))
R = np.zeros((X_count, A_count, X_count))

# X
# 0: 缺水
# 1: 健康
# 2: 溢水
# 3: 凋亡

# A
# 0: 浇水
# 1: 不浇水

# 缺水浇水
P[0, 0, 0] = 0.5
R[0, 0, 0] = -1
P[0, 0, 1] = 0.5
R[0, 0, 1] = 1
# 缺水不浇水
P[0, 1, 0] = 0.4
R[0, 1, 0] = -1
P[0, 1, 3] = 0.6
R[0, 1, 0] = -100
# 健康浇水
P[1, 0, 1] = 0.6
R[1, 0, 1] = 1
P[1, 0, 2] = 0.4
R[1, 0, 2] = 1
# 健康不浇水
P[1, 1, 0] = 0.6
R[1, 1, 0] = -1
P[1, 1, 1] = 0.4
R[1, 1, 1] = 1
# 溢水浇水
P[2, 0, 2] = 0.6
R[2, 0, 2] = -1
P[2, 0, 3] = 0.4
R[2, 0, 3] = -100
# 溢水不浇水
P[2, 1, 2] = 0.4
R[2, 1, 2] = -1
P[2, 1, 1] = 0.6
R[2, 1, 1] = 1
# 凋亡浇水
P[3, 0, 3] = 1.0
R[3, 0, 3] = -100
# 凋亡不浇水
P[3, 1, 3] = 1.0
R[3, 1, 3] = -100

np.save('P', P)
np.save('R', R)