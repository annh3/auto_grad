# scalar auto diff package
import random
import numpy as np

class Value:

    def __init__(self, data=None, _children=(), _op=''):
        if data is None:
          data = random.uniform(-2,2)
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad *(other * self.data**(other-1))
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def relu(self):
        out = Value(self.data, (self,), 'ReLU') if self.data >= 0 else Value(0, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # Other operations implemented in terms of prior ones 
    def __float__(self): return float(self.data)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        
        queue = []
        queue.append(self)
        while queue:
            cur = queue.pop(0)
            topo.append(cur)
            for child in cur._children:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in topo:
            v._backward()
