import numpy as np
import torch
from training.aces_model import ACESModel,model_from_checkpoint
from tools.derive_aces_analytic_shadow import derive
from tools.compare_aces_analytic_shadow import outside_support_identical,motion_checks,support_of


def test_derive_keeps_weights_and_switches_policy():
    parent=ACESModel(8,'pixel_rms','local_codes','shared')
    state={'model':parent.state_dict(),'contract':parent.contract(),'pilot_sha256':'p','recipe':{'channels':8},'optimizer':{}}
    child=derive(state);model=model_from_checkpoint(child)
    assert model.shadow=='analytic' and 'optimizer' not in child and child['pilot_sha256']=='p'
    assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
    shared=ACESModel(8,'pixel_rms','local_codes','shared');shared.load_state_dict(state['model'])
    try:derive({'model':child['model'],'contract':child['contract']});raise AssertionError('analytic parent accepted')
    except ValueError:pass


def test_outside_support_check_detects_changes_only_outside():
    source=np.full((6,6,3),.5,np.float32);source[:2]=.002
    assert support_of(source)[:2].all() and not support_of(source)[2:].any()
    a=np.random.default_rng(0).random((6,6,3)).astype(np.float32);b=a.copy();b[:2]+=1
    assert outside_support_identical(source,a,b)==(True,12)
    b[4,4,0]+=1e-6
    assert outside_support_identical(source,a,b)[0] is False


def test_motion_checks():
    clip=lambda worst,neg:{'max_transition_cell_excess_ev':worst,'frames':[{'negative_luminance_pixels':neg,'shadow_lift_gt1code':0}]*96}
    assert all(motion_checks({'clips':[clip(.01,0)]*4}).values())
    c=motion_checks({'clips':[clip(.03,0)]+[clip(.01,1)]*3})
    assert not c['zero_negative_Y'] and not c['motion_worst_transition_002']
