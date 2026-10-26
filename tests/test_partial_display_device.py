import pytest
import torch
from tools.train_partial_display import _Forward
from training.aces_reference import reference_view


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA comparison')
def test_display_forward_and_backward_match_cpu():
    reference_view().set_threads(2)
    x=torch.linspace(.02,2,3*9*13).reshape(1,3,9,13)
    values=[]
    for device in ('cpu','cuda'):
        a=x.detach().clone().to(device).requires_grad_();out=_Forward.apply(a)
        out.square().sum().backward();values.append((out.detach().cpu(),a.grad.cpu()))
    assert torch.equal(values[0][0],values[1][0])
    torch.testing.assert_close(values[0][1],values[1][1],rtol=1e-6,atol=1e-7)
