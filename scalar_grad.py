# scalar auto diff package

class Value:

	def __init__(self, data, _children=(), _op=''):
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

	def __pow__(self, other):
		assert isinstance(other, (int, float))
		out = Value(self.data ** other, (self,), f'**{other}')

		def _backward():
			self.grad += out.grad *(other * self.data**(other-1))
		out._backward = _backward

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

def main():
	a = Value(5)
	def f(x): 
		return (x+2)**2 + x**3
	y = f(a)
	y.backward()
	print(y.data, a.grad)

if __name__ == "__main__":
	main()

