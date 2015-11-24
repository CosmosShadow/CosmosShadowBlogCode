# coding: utf-8
import theano
import theano.tensor as T

state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])

# 运行结果可以看到，返回的state是未更新的state
print state.get_value()
print accumulator(1)
print state.get_value()

print accumulator(2)
print accumulator(3)

state.set_value(10)
print state.get_value()
