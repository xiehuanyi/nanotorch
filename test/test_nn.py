import unittest
import numpy as np
import torch
import sys
import os

from nanotorch.tensor import Tensor
import nanotorch.nn as nn

class TestNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_linear(self):
        bs, in_feat, out_feat = 16, 20, 30
        
        # Initialize nanotorch model
        model = nn.Linear(in_feat, out_feat)
        
        # Initialize pytorch model
        pt_model = torch.nn.Linear(in_feat, out_feat)
        
        # Sync weights
        # nanotorch linear: weight shape (in, out), bias shape (out,)
        # pytorch linear: weight shape (out, in), bias shape (out,)
        
        with torch.no_grad():
            pt_model.weight.copy_(torch.tensor(model.weight.data.T))
            pt_model.bias.copy_(torch.tensor(model.bias.data))
            
        # Input
        input_data = np.random.randn(bs, in_feat).astype(np.float32)
        t_in = Tensor(input_data, requires_grad=True)
        pt_in = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)
        
        # Forward
        out = model(t_in)
        pt_out = pt_model(pt_in)
        
        np.testing.assert_allclose(out.data, pt_out.detach().numpy(), atol=1e-5, rtol=1e-5)
        
        # Backward
        out.backward()
        pt_out.backward(torch.ones_like(pt_out))
        
        # Check grads
        np.testing.assert_allclose(t_in.grad, pt_in.grad.numpy(), atol=1e-5, rtol=1e-5, err_msg="Input grad mismatch")
        np.testing.assert_allclose(model.weight.grad, pt_model.weight.grad.numpy().T, atol=1e-5, rtol=1e-5, err_msg="Weight grad mismatch")
        np.testing.assert_allclose(model.bias.grad, pt_model.bias.grad.numpy(), atol=1e-5, rtol=1e-5, err_msg="Bias grad mismatch")

    def test_relu(self):
        bs, feat = 16, 20
        model = nn.ReLU()
        
        input_data = np.random.randn(bs, feat).astype(np.float32)
        t_in = Tensor(input_data, requires_grad=True)
        pt_in = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)
        
        out = model(t_in)
        pt_out = torch.relu(pt_in)
        
        np.testing.assert_allclose(out.data, pt_out.detach().numpy(), atol=1e-5)
        
        out.backward()
        pt_out.backward(torch.ones_like(pt_out))
        
        np.testing.assert_allclose(t_in.grad, pt_in.grad.numpy(), atol=1e-5)

    def test_batchnorm2d(self):
        # Despite the name BatchNorm2D, the implementation performs normalization over dim 0 (batch dimension) only.
        # This matches BatchNorm1d behavior for 2D inputs (N, C).
        # We will test it as such.
        
        N, C = 32, 64
        model = nn.BatchNorm2D(C, eps=1e-5)
        # Force gamma/beta to something non-identity to test gradients better
        model.gamma.data = np.random.rand(C).astype(np.float32) + 0.5
        model.beta.data = np.random.randn(C).astype(np.float32)
        
        pt_model = torch.nn.BatchNorm1d(C, eps=1e-5, affine=True, track_running_stats=False)
        # Note: PyTorch BN uses unbiased=False for calculating variance used in normalization
        
        with torch.no_grad():
            pt_model.weight.copy_(torch.tensor(model.gamma.data))
            pt_model.bias.copy_(torch.tensor(model.beta.data))
            
        input_data = np.random.randn(N, C).astype(np.float32)
        
        t_in = Tensor(input_data, requires_grad=True)
        pt_in = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)
        
        out = model(t_in)
        pt_out = pt_model(pt_in)
        
        np.testing.assert_allclose(out.data, pt_out.detach().numpy(), atol=1e-4, rtol=1e-4) # Slightly looser tolerance for BN 
        
        out.backward()
        pt_out.backward(torch.ones_like(pt_out))
        
        np.testing.assert_allclose(t_in.grad, pt_in.grad.numpy(), atol=1e-4, rtol=1e-4, err_msg="Input grad mismatch")
        np.testing.assert_allclose(model.gamma.grad, pt_model.weight.grad.numpy(), atol=1e-4, rtol=1e-4, err_msg="Gamma grad mismatch")
        np.testing.assert_allclose(model.beta.grad, pt_model.bias.grad.numpy(), atol=1e-4, rtol=1e-4, err_msg="Beta grad mismatch")

if __name__ == '__main__':
    unittest.main()
