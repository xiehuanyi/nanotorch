import numpy as np

class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                g = param.grad
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g * g)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                

