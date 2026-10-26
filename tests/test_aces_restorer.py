import torch
from training.aces_model import ACESModel,model_from_checkpoint
from training.aces_restorer import LocalCodeRestorer


def test_local_branch_zero_head_and_code_bound():
    branch=LocalCodeRestorer();source=torch.rand(1,3,80,80)
    assert torch.count_nonzero(branch(source))==0
    with torch.no_grad():branch.head.bias.fill_(100)
    restored=source+.49/255*torch.tanh(branch(source))
    assert (restored-source).abs().max()<=.49001/255


def test_v4_restoration_cannot_change_reconstruction_pixels():
    model=ACESModel(8,'pixel_rms','local_codes').eval()
    state={'model':model.state_dict(),'contract':model.contract()}
    restored=model_from_checkpoint(state).eval()
    source=torch.ones(1,3,80,80)
    source[:,:,:,:20]=0
    source[:,:,:,20:40]=.5
    with torch.no_grad():
        before=restored(source)
        restored.restorer.head.bias.fill_(1)
        after=restored(source)
    support=before.reconstruction_weight.expand_as(before.acescg)>0
    torch.testing.assert_close(before.acescg[support],after.acescg[support],rtol=0,atol=0)
    assert torch.any(before.acescg[~support]!=after.acescg[~support])
