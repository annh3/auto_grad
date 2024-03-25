"""
This is an experimental file. With this simple example, We can see the
effect of learning rate on exploding loss (and gradients), simply 
adjust the learning rate from 1e-3 to 1e-1.
"""
from scalar_grad import Value
import random


class Linear:
    def __init__(self):
        self.a, self.b = Value(random.random()), Value(random.random())
    def __call__(self,x):
        return self.a*x+self.b
        
    def zero_grad(self):
        self.a.grad, self.b.grad = 0,0

def linear_regression(X,Y,n=0.0001,epochs=30):
    model = Linear()
    for t in range(epochs):
        for x,y in zip(X,Y):
            model.zero_grad()
            loss = (model(x) - y)**2
            loss.backward()
            print('loss: ', loss, 'epoch: ', t)
            print('model.a.grad: ', model.a.grad)
            print('model.b.grad: ', model.b.grad)
            model.a, model.b = (model.a - n*model.a.grad, model.b - n*model.b.grad)
    return model



def linear_regression_debug():
    X = list(range(100))
    X = X*3
    Y = [3*x + random.random() for x in X]
    print('X: ', X)
    print('Y: ', Y)
    model = linear_regression(X,Y)
    print('model.a', model.a.data)
    print('model.b', model.b.data)


def main():
    linear_regression_debug()
    

if __name__ == "__main__":
    main()

