# coding: utf-8
import theano
import theano.tensor as T

# 定义两个double类型向量符号
# 'a', 'b'两个参数利于调试
a = T.dscalar('a')
b = T.dscalar('b')
# 相加
c = a + b
# build函数(c代码): [a, b]为inputs, c为outputs
f = theano.function([a,b], c)

# 运行函数
print f(1.5, 2.5)
assert 4.0 == f(1.5, 2.5)

# 这样也可以的: 只是没有function灵活，但测试时可以用，因为少了function申明
print c.eval({a : 16.3, b : 12.1})