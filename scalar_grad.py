# scalar auto diff package

class Value:

	def __init__(self, data, _children=()):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, (self, other))

		def _backward():
			self.grad += out.grad
			other.grad += out.grad
		out.backward = _backward

		return out

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, (self, other))

		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out.backward = _backward

		return out


def backward(self, visited = None):
	if visited is None:
		visited = set([self])
		self.grad = 1
	self._backward()
	for child in self._children:
		if child not in visited:
			visited.add(child)
			child.backward(visited)

