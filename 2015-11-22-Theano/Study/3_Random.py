# coding: utf-8
from theano.tensor.shared_randomstreams import RandomStreams
import theano

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))		#均匀分布
rv_n = srng.normal((2,2))		#正态分布
f = theano.function([], rv_u)
g = theano.function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = theano.function([], rv_u + rv_u - 2 * rv_u)

print 'f'
print f()
print ''

print 'g'
print g()
print ''

print nearly_zeros()