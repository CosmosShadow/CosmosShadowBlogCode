# coding: utf-8
import theano
import theano.tensor as T

W = T.dmatrix('W')
V = T.dmatrix('V')
x = T.dvector('x')
y = T.dot(x, W)
JV = T.Rop(y, W, V)
f = theano.function([W, V, x], [y, JV])
print f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1])