import numpy as np
from nanotorch.tensor import Parameter

class Module:
    def parameters(self):
        self.params = []
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                self.params.append(v)
            elif isinstance(v, Module):
                self.params.extend(v.parameters())
        return self.params

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, dim_in, dim_out, bias=True, requires_grad=True, dtype=np.float32):
        self.weight = Parameter(np.random.randn(dim_in, dim_out) * np.sqrt(2.0 / dim_in), requires_grad=requires_grad, dtype=dtype)
        if bias:
            self.bias = Parameter(np.zeros((dim_out,)), requires_grad=requires_grad, dtype=dtype)
        else:
            self.bias = None
    
    def forward(self, x):
        if self.bias is not None:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight  

class ReLU(Module):
    def forward(self, x):
        return x.relu()


class BatchNorm2D(Module):
    def __init__(self, dim, eps=1e-5, requires_grad=True, dtype=np.float32):
        self.eps = eps
        self.gamma = Parameter(np.ones((dim,)), requires_grad=requires_grad, dtype=dtype)
        self.beta = Parameter(np.zeros((dim,)), requires_grad=requires_grad, dtype=dtype)
    
    def forward(self, x):
        mean = x.mean(dim=0, keepdims=True)
        var = x.var(dim=0, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.gamma + self.beta