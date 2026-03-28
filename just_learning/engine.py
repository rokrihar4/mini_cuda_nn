class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data # Actual number
        self.grad = 0.0 # Derivative of final output with respect to this value
        self._prev = set(_children) # Parent nodesin the computation graph
        self._op = _op # Which operation this value
        self._backward = lambda: None # Function that knows how to pass gradient backward

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __mul__(self, other):
        out = Value(self.data * other.data, (self,other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __add__(self,other):
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad += 1.0 + out.grad
            other.grad += 1.0 + out.grad

        out._backward = _backward
        return out

x = Value(3.0)
y = Value(2.0)
z = x * y
print(z)

z.grad
z._backward
        


    