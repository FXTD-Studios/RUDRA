import pytest
import torch
from training.aces_partial_pairs import validate_pair
from training.aces_active_temporal import active_temporal_loss


def test_pair_rejects_cross_scene_wrong_direction_and_gaps():
    a={'split':'train','scene_id':'s','group_id':'g','frame':0,'translation_xy':[0,0]}
    b={**a,'frame':1,'translation_xy':[.5,.25]}
    validate_pair(a,b)
    for change in ({'scene_id':'other'},{'group_id':'other'},{'frame':2},{'translation_xy':[-.5,.25]},{'split':'val'}):
        with pytest.raises(ValueError):validate_pair(a,{**b,**change})


def test_partial_active_union_includes_changed_neighbour_and_backpropagates():
    p=torch.zeros(2,3,12,12,requires_grad=True);target=torch.zeros_like(p)
    with torch.no_grad():p[1,:,6,6]=1
    high=torch.ones(2,1,12,12,dtype=torch.bool);partial=torch.zeros_like(high);partial[1,0,6,6]=True
    loss,count,_=active_temporal_loss(p,target,high,partial,border=2)
    assert count==4 and loss>0
    loss.backward();assert p.grad[1,:,6,6].abs().sum()>0
