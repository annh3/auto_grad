from scalar_grad import Value


def test():
    a = Value(5)
    def f(x): 
        return (x+2)**2 + x**3
    y = f(a)
    y.backward()
    assert y.data == 174
    assert a.grad == 89


def main():
    test()
    
if __name__ == "__main__":
    main()
