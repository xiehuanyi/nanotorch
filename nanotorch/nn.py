import numpy as np
from nanotorch.tensor import Parameter, Tensor

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


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, requires_grad=True, dtype=np.float32):
        self.eps = eps
        self.momentum = momentum # 用于更新 running_stats
        self.training = True     # 标记当前是训练还是推理模式
        self.gamma = Parameter(np.ones((1, dim), dtype=dtype), requires_grad=requires_grad)
        self.beta = Parameter(np.zeros((1, dim), dtype=dtype), requires_grad=requires_grad)
        self.running_mean = np.zeros((1, dim), dtype=dtype)
        self.running_var = np.ones((1, dim), dtype=dtype)
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdims=True)
            var = x.var(dim=0, keepdims=True)
            x_hat = (x - mean) / (var + Tensor(self.eps, requires_grad=False)).sqrt()
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            r_mean = Tensor(self.running_mean, requires_grad=False)
            r_var = Tensor(self.running_var, requires_grad=False)            
            x_hat = (x - r_mean) / (r_var + Tensor(self.eps, requires_grad=False)).sqrt()

        return x_hat * self.gamma + self.beta