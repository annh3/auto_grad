import random
import numpy as np


def neuron(weights, inputs, relu=True):
	# Evaluate neuron with given weights on a given input
	v = sum(weights[i]*x for i,x in enumerate(inputs))
	return v.relu() if relu else v


class Net:
	# Depth 3 fully connected neural net
	def __init__(self, N=16):
		self.layer_1 = [[Value(),Value()] for i in range(N)]
		self.layer_2 = [[Value() for j in range (N)] for i in range(N)]
		self.output = [Value() for i in range(N)]
		self.parameters = [v for L in [self.layer_1,self.layer_2,[self.output]] for w in L for v in w] # should be flattened list

	def __call__(self,x):
		layer_1_vals = [neuron(w,x) for w in self.layer_1]
		layer_2_vals = [neuron(w,layer_1_vals) for w in self.layer_2]
		return neuron(self.output, layer_2_vals, relu=False)

	def zero_grad(self):
		for p in self.parameters:
			p.grad=0

"""
Instead of mse, use L(y,y') = max1 - y*y', 0 since the labels are
+-1, so we have a correct classification with the same sign

Instead of sgd, we do standard gradient descent, using all the data
points before taking ca step
"""
def train(X,Y,n=0.001,epochs=30):
	model = Net()
	for t in range(epochs):
		loss = sum([(1+ -y*model(x)).relu() for x,y in zip(X,Y)])/len(X)
		model.zero_grad()
		loss.backward()
		for p in model.parameters:
			p.data -= n*p.grad