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


