# coding: utf-8

import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
f = theano.function([x], gy)

print f(4)
print pp(f.maker.fgraph.outputs[0])