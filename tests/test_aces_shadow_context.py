import torch
from training.aces_shadow import ShadowFields
from training.aces_model import ACESModel,model_from_checkpoint
from tools.train_aces_shadow_context import equal_domain_loss,initialize_shadow_pair
from tools.train_aces_shadow_context import valid_crop_context


def test_true_image_edges_are_kept_but_artificial_crop_edges_excluded():
    mask=valid_crop_context(0,0,300,300)
    assert mask[0,0,0]==1 and mask[0,151,151]==1 and mask[0,152,152]==0
    interior=valid_crop_context(50,50,300,300)
    assert interior[0,0,0]==0 and interior[0,40,40]==1
    assert valid_crop_context(0,0,192,192).min()==1


def test_valid_crop_context_matches_full_image_at_genuine_edges():
    torch.set_num_threads(2);torch.manual_seed(5)
    model=ShadowFields(True).eval()
    torch.nn.init.normal_(model.layers[-1].weight,std=.01)
    source=torch.rand(1,3,256,256)*.01;baseline=source*.1
    with torch.no_grad():
        full=model(source,baseline)
        for x,y in ((0,0),(64,64)):
            crop=model(source[:,:,y:y+192,x:x+192],baseline[:,:,y:y+192,x:x+192])
            valid=valid_crop_context(x,y,256,256).bool()[None].expand_as(crop)
            torch.testing.assert_close(crop[valid],full[:,:,y:y+192,x:x+192][valid],rtol=2e-4,atol=1e-6)


def test_domain_loss_does_not_allow_dense_synthetic_pixels_to_dominate():
    error=torch.ones(4,1,8,8);error[2:]=9
    weight=torch.ones_like(error);weight[:2]=0;weight[0,0,0,0]=1
    assert equal_domain_loss(error,weight)==5


def test_wide_branch_uses_signal_outside_old_radius_and_loads_contract():
    torch.set_num_threads(2);torch.manual_seed(2)
    local=ShadowFields();wide=ShadowFields(True);initialize_shadow_pair(local,wide)
    assert torch.equal(wide.layers[0].weight[:,:6],local.layers[0].weight)
    assert (wide.layers[0].weight[:,6:]==0).all()
    source=torch.zeros(1,3,96,96);changed=source.clone();changed[:,:,:,68:70]=.01
    with torch.no_grad():
        for model in (local,wide):
            for layer in model.layers:
                if isinstance(layer,torch.nn.Conv2d):layer.weight.fill_(.001);layer.bias.zero_()
        a=local(source,source);b=local(changed,changed)
        c=wide(source,source);d=wide(changed,changed)
    assert torch.equal(a[:,:,48,48],b[:,:,48,48])
    assert not torch.equal(c[:,:,48,48],d[:,:,48,48])
    model=ACESModel(4,'pixel_rms','local_codes','sqrt_luma_context')
    loaded=model_from_checkpoint({'model':model.state_dict(),'contract':model.contract()})
    assert loaded.contract()['shadow_context_radius']==40
