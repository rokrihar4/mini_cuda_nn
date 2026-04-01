# Building a autograd system from PyTorch
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
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = [] # topological graph
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __sub__(self, other):
        return self + (other * Value(-1.0))

    def __pow__(self, other):
        assert isinstance(other, (int,float)), "only supports int/float powers"

        out = Value(self.data ** other, (self,), f"**{other}" )

        def _backward():
            # cuz y = x^n
            # and it's derivative dy/dx = n * x^(n-1)
            self.grad += (other * (self.data ** (other-1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

x = Value(2.0)
y = Value(-3.0)

z = x * y + x**2
z.backward()

print("x.grad =", x.grad)
print("y.grad =", y.grad)

# x.grad is here 1 cuz
# z = x*y + x^2
# dz/dx = y + 2x = -3 + 4 = 1