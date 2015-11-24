# coding: utf-8

import theano
import theano.tensor as T
x = T.dvector('x')
y = x ** 2
a = T.dvector('x')
z = x * a

J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
f = theano.function([x], J, updates=updates)

print f([4, 4])

dJacobian = theano.gradient.jacobian(z, [x, a])

gradFunciton = theano.function([x, a], dJacobian)

print gradFunciton([1, 2], [3, 4])