import numpy as np
from nanotorch.nn import Module
from nanotorch.tensor import Tensor
import nanotorch as nt

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x: (B, V), y: (B,)
        bs = x.shape[0]
        # numerical stability: subtract max
        x_max = x.data.max(axis=1, keepdims=True)
        x_shifted = x - Tensor(x.data.max(axis=1, keepdims=True), requires_grad=False) 
        exp_x = x_shifted.exp()
        sum_exp_x = exp_x.sum(dim=1, keepdims=True)
        log_sum_exp = sum_exp_x.log()
        log_probs = x_shifted - log_sum_exp
            
        loss = -log_probs[nt.arange(bs), y].mean()
        return loss


class MSELoss(Module):
    def forward(self, y_hat, y):
        loss = (y_hat - y) * (y_hat - y)
        return loss.mean()