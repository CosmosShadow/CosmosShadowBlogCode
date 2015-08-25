#coding=utf-8
# import numpy as np
# import matplotlib.pyplot as plt
import random
import math
import sys

piMap = [0.1, 0.3, 0.2, 0.4]
# piMap = [0.1, 0.9]
circleCount = 1000
# circleCount = 100000

# -------------计算转移概率与”接受-拒绝概率“-------------
value_count = len(piMap)
# 条件概率分布: 取平均分布
condition_possibility = 1.0 / value_count
# 接受-拒绝概率
accept_rejects = [[0]*value_count for i in range(value_count)]
translate_possibilitys = [[0]*value_count for i in range(value_count)]
for i in range(value_count):
	for j in range(value_count):
		alpha = piMap[j] / piMap[i]
		accept_rejects[i][j] = min(1, alpha)
		translate_possibilitys[i][j] = condition_possibility * accept_rejects[i][j]
for i in range(value_count):
	# 当前每一行的条件转移概率之和
	current_line_translate_possibility_sum = sum(translate_possibilitys[i])
	# 为平衡单个状态转移概率和为1，从新计算自转自的概率。
	translate_possibilitys[i][i] = 1 - (current_line_translate_possibility_sum - translate_possibilitys[i][i])
	# 自转自在条件概率下采样时的"接受-拒绝概率"
	accept_rejects[i][i] = translate_possibilitys[i][i] / condition_possibility

print "转移概率: "
print translate_possibilitys
print "接受-拒绝概率: "
print accept_rejects

# 保存计数
counts = [0 for i in range(len(piMap))]

# -------------生成序列------------
X = 0
for i in range(circleCount):
	# 按条件概率生成随机的下一个数Y
	Y = random.randint(0, len(piMap)-1)
	# 对应的“接受-拒绝概率"

	# 下面这三行可要可不要
	# 要的逻辑: 当下一个状态与当前状态相等时，无论"接受-拒绝概率"为多少，都会转向自身状态。
	# 此时可以不计算自转自的"接受-拒绝概率"，即也少了计算自转自的概率。
	# 不要的逻辑: 把自转自的变化也当作一般处理
	# 倾向于要下面三行，逻辑简单
	if X == Y:
		counts[X] += 1
		continue

	accept_reject = accept_rejects[X][Y]
	# uniform[0, 1]的随机数
	randomValue = random.random()
	# 当小于接收率时接收，大于时采用上一个状态为下一个状态
	if randomValue < accept_reject:
		counts[Y] += 1
		X = Y
	else:
		counts[X] += 1

# -------------计算生成序列的分布------------
percents = []
totalCount = sum(counts)
for i in range(len(counts)):
	percent = float(counts[i]) / totalCount
	percents.append(percent)

# 输出结果
print "循环次数: " + str(circleCount)
sys.stdout.write("原始分布: ")
print piMap
sys.stdout.write("生成分布: ")
print percents



