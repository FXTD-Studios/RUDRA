import torch
from tools.audit_aces_gradient_alignment import geometry,summarize


def test_cosine_and_zero_norm_are_not_confused():
    a=torch.tensor([1.,0.]);assert geometry(a,-a)['cosine']==-1
    assert geometry(a,a)['cosine']==1
    assert geometry(a,torch.tensor([0.,1.]))['cosine']==0
    assert geometry(a,torch.zeros(2))['cosine'] is None


def test_conflict_does_not_imply_combined_step_harms_both():
    spatial=torch.tensor([2.,0.]);temporal=torch.tensor([-1.,1.]);total=spatial+temporal
    row={k:geometry(temporal,spatial) for k in ('temporal_photo','temporal_detail','temporal_spatial','photo_detail')}
    row.update(spatial_dot_total=float(spatial@total),temporal_dot_total=float(temporal@total))
    s=summarize([row]);assert s['temporal_spatial']['negative']==1
    assert s['combined_step_opposes_spatial']==0 and s['combined_step_opposes_temporal']==0
