import numpy as np

class Tensor:
    __array_priority__ = 1000
    def __init__(self, data, dtype=np.float32, requires_grad=False, _children=()):
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.data = np.array(data, dtype=dtype)
        if requires_grad:
            self.grad = np.zeros_like(self.data, dtype=dtype)
        else:
            self.grad = None
        self._children = set(_children)
        self._back_fn = None
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data + other.data, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _back_fn():
            if self.requires_grad:
                grad = out.grad
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                grad = out.grad
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        out._back_fn = _back_fn        
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data * other.data, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _back_fn():
            if self.requires_grad:
                grad = other.data * out.grad
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        out._back_fn = _back_fn
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data / other.data, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _back_fn():
            if self.requires_grad:
                grad = out.grad / other.data
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                grad = -out.grad * self.data / (other.data * other.data)
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        out._back_fn = _back_fn
        return out

    def __neg__(self):
        out = Tensor(-self.data, dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += -out.grad
        out._back_fn = _back_fn
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data @ other.data, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                grad_a = self.data
                grad_b = out.grad
                if grad_a.ndim == 1 and grad_b.ndim == 1:
                    other.grad += np.outer(grad_a, grad_b)
                elif grad_a.ndim > 2:
                    # Reshape to 2D for batched matmul gradient: (B, ..., I) -> (B*..., I)
                    grad_a_2d = grad_a.reshape(-1, grad_a.shape[-1])
                    grad_b_2d = grad_b.reshape(-1, grad_b.shape[-1])
                    other.grad += grad_a_2d.T @ grad_b_2d
                else: 
                    other.grad += self.data.T @ out.grad
        out._back_fn = _back_fn
        return out   
        
    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1))
        out._back_fn = _back_fn
        return out
    
    def sum(self, dim=None, keepdims=False):
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                grad_output = out.grad
                if dim is not None and not keepdims:
                    grad_output = np.expand_dims(grad_output, axis=dim)
                self.grad += np.ones_like(self.data) * grad_output
        out._back_fn = _back_fn
        return out
    
    def mean(self, dim=None, keepdims=False):
        out = Tensor(self.data.mean(axis=dim, keepdims=keepdims), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                grad_output = out.grad
                if dim is not None and not keepdims:
                    grad_output = np.expand_dims(grad_output, axis=dim)
                self.grad += np.ones_like(self.data) * grad_output / (self.data.size / out.data.size)
        out._back_fn = _back_fn
        return out
    
    def relu(self):
        out = Tensor(np.maximum(self.data, 0), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        out._back_fn = _back_fn
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._back_fn = _back_fn
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._back_fn = _back_fn
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                self.grad += out.grad / (2 * out.data)
        out._back_fn = _back_fn
        return out

    def var(self, dim=None, keepdims=False):
        mu = self.mean(dim=dim, keepdims=True)
        diff = self - mu
        return (diff ** 2).mean(dim=dim, keepdims=keepdims)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=self.dtype)
        for node in reversed(topo):
            if node._back_fn:
                node._back_fn()

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        def unwrap(x):
            if isinstance(x, Tensor):
                return x.data
            if isinstance(x, tuple):
                return tuple(unwrap(i) for i in x)
            return x
        numpy_idx = unwrap(idx)
        out = Tensor(self.data[numpy_idx], dtype=self.dtype, requires_grad=self.requires_grad, _children=(self,))
        def _back_fn():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, numpy_idx, out.grad)
                self.grad += grad
        out._back_fn = _back_fn
        return out


        
class Parameter(Tensor):
    def __init__(self, data, dtype=np.float32, requires_grad=True):
        super().__init__(data, dtype=dtype, requires_grad=requires_grad)
    