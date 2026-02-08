import unittest
import numpy as np
import torch
import sys
import os

from nanotorch.tensor import Tensor

class TestTensorOps(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def _check_binary(self, op_name, data1, data2, requires_grad=True):
        t1 = Tensor(data1, requires_grad=requires_grad)
        t2 = Tensor(data2, requires_grad=requires_grad)
        
        pt1 = torch.tensor(data1, requires_grad=requires_grad, dtype=torch.float32)
        pt2 = torch.tensor(data2, requires_grad=requires_grad, dtype=torch.float32)
        
        # Operation
        if op_name == 'add':
            res = t1 + t2
            pres = pt1 + pt2
        elif op_name == 'sub':
            res = t1 - t2
            pres = pt1 - pt2
        elif op_name == 'mul':
            res = t1 * t2
            pres = pt1 * pt2
        elif op_name == 'div':
            res = t1 / t2
            pres = pt1 / pt2
        elif op_name == 'matmul':
            res = t1 @ t2
            pres = pt1 @ pt2
            
        # Forward check
        np.testing.assert_allclose(res.data, pres.detach().numpy(), atol=1e-5, rtol=1e-5, err_msg=f"Forward failed for {op_name}")
        
        # Backward check
        if requires_grad:
            res.backward()
            pres.retain_grad() # Just in case
            pres.backward(torch.ones_like(pres))
            
            np.testing.assert_allclose(t1.grad, pt1.grad.numpy(), atol=1e-5, rtol=1e-5, err_msg=f"Backward (grad1) failed for {op_name}")
            np.testing.assert_allclose(t2.grad, pt2.grad.numpy(), atol=1e-5, rtol=1e-5, err_msg=f"Backward (grad2) failed for {op_name}")

    def _check_unary(self, op_name, data, arg=None, requires_grad=True):
        t1 = Tensor(data, requires_grad=requires_grad)
        pt1 = torch.tensor(data, requires_grad=requires_grad, dtype=torch.float32)
        
        if op_name == 'neg':
            res = -t1
            pres = -pt1
        elif op_name == 'pow':
            res = t1 ** arg
            pres = pt1 ** arg
        elif op_name == 'relu':
            res = t1.relu()
            pres = torch.relu(pt1)
        elif op_name == 'log':
            res = t1.log()
            pres = torch.log(pt1)
        elif op_name == 'exp':
            res = t1.exp()
            pres = torch.exp(pt1)
        elif op_name == 'sqrt':
            res = t1.sqrt()
            pres = torch.sqrt(pt1)
        elif op_name == 'sum':
            res = t1.sum(dim=arg)
            pres = pt1.sum(dim=arg if arg is not None else tuple())
            if arg is None: # Torch sum() returns 0-d tensor, nanotorch matches implementation logic
                 pass
        elif op_name == 'mean':
            res = t1.mean(dim=arg)
            pres = pt1.mean(dim=arg if arg is not None else tuple())
        elif op_name == 'var':
            # PyTorch var uses Bessel's correction (unbiased=True by default), need to check what nanotorch assumes
            # nanotorch implementation: (diff ** 2).mean() -> This is biased variance (population variance, 1/N)
            # torch.var(unbiased=False) matches np.var()
            res = t1.var(dim=arg)
            if arg is None:
                pres = pt1.var(unbiased=False)
            else:
                pres = pt1.var(dim=arg, unbiased=False)
            
        # Forward check
        np.testing.assert_allclose(res.data, pres.detach().numpy(), atol=1e-5, rtol=1e-5, err_msg=f"Forward failed for {op_name}")
        
        # Backward check
        if requires_grad:
            res.backward()
            pres.backward(torch.ones_like(pres))
            np.testing.assert_allclose(t1.grad, pt1.grad.numpy(), atol=1e-5, rtol=1e-5, err_msg=f"Backward failed for {op_name}")

    def test_add(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        d2 = np.random.randn(10, 10).astype(np.float32)
        self._check_binary('add', d1, d2)
        
    def test_add_broadcast(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        d2 = np.random.randn(10).astype(np.float32) # Broadcast last dim
        self._check_binary('add', d1, d2)
        
        d3 = np.random.randn(10, 1).astype(np.float32)
        self._check_binary('add', d1, d3)
        
    def test_sub(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        d2 = np.random.randn(10, 10).astype(np.float32)
        self._check_binary('sub', d1, d2)

    def test_mul(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        d2 = np.random.randn(10, 10).astype(np.float32)
        self._check_binary('mul', d1, d2)
        
    def test_div(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        d2 = np.random.randn(10, 10).astype(np.float32) + 2.0 # avoid div by zero
        self._check_binary('div', d1, d2)

    def test_matmul(self):
        d1 = np.random.randn(10, 20).astype(np.float32)
        d2 = np.random.randn(20, 30).astype(np.float32)
        self._check_binary('matmul', d1, d2)
        
    def test_neg(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('neg', d1)

    def test_pow(self):
        d1 = np.abs(np.random.randn(10, 10).astype(np.float32)) + 0.1
        self._check_unary('pow', d1, arg=2)
        self._check_unary('pow', d1, arg=0.5)

    def test_relu(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('relu', d1)

    def test_log(self):
        d1 = np.abs(np.random.randn(10, 10).astype(np.float32)) + 0.1
        self._check_unary('log', d1)

    def test_exp(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('exp', d1)
        
    def test_sqrt(self):
        d1 = np.abs(np.random.randn(10, 10).astype(np.float32)) + 0.1
        self._check_unary('sqrt', d1)

    def test_sum(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('sum', d1) # All
        self._check_unary('sum', d1, arg=0) # Dim 0
        self._check_unary('sum', d1, arg=1) # Dim 1
        
    def test_mean(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('mean', d1) # All
        self._check_unary('mean', d1, arg=0) 
        self._check_unary('mean', d1, arg=1) 

    def test_var(self):
        d1 = np.random.randn(10, 10).astype(np.float32)
        self._check_unary('var', d1)
        self._check_unary('var', d1, arg=0)

    def test_getitem(self):
        d1 = np.random.randn(5, 5).astype(np.float32)
        t1 = Tensor(d1, requires_grad=True)
        pt1 = torch.tensor(d1, requires_grad=True, dtype=torch.float32)
        
        # Slice
        res = t1[1:3, 1:3]
        pres = pt1[1:3, 1:3]
        
        np.testing.assert_allclose(res.data, pres.detach().numpy(), atol=1e-5)
        
        res.backward()
        pres.backward(torch.ones_like(pres))
        
        np.testing.assert_allclose(t1.grad, pt1.grad.numpy(), atol=1e-5)

if __name__ == '__main__':
    unittest.main()
