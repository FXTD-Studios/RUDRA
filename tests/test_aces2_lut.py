import pytest
import torch
pytest.importorskip('PyOpenColorIO')
from tools.audit_aces2_lut import sample_lut


def test_lut_axis_order_and_interpolation_gradients():
    axis = torch.linspace(0,1,4)
    b,g,r = torch.meshgrid(axis,axis,axis,indexing='ij')
    lut = torch.stack((2*r+g,3*g+b,4*b+r))[None]
    source = torch.tensor([[.1,.2,.3],[.95,.35,.75]],requires_grad=True)
    expected = torch.stack((2*source[:,0]+source[:,1],3*source[:,1]+source[:,2],
                            4*source[:,2]+source[:,0]),-1)
    actual = sample_lut(lut,source)
    torch.testing.assert_close(actual,expected)
    gradient, = torch.autograd.grad(actual.sum(),source)
    torch.testing.assert_close(gradient,torch.tensor([[3.,4.,5.],[3.,4.,5.]]))
