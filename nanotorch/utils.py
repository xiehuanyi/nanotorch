import numpy as np
from nanotorch.tensor import Tensor

def arange(start, stop=None, step=1, dtype=np.int64, requires_grad=False):
    if stop is None:
        stop = start
        start = 0
    data = np.arange(start, stop, step, dtype=dtype)
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)