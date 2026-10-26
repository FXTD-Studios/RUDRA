import pytest
import torch
import numpy as np
from tools.train_aces_restorer import merge_training_ancestry,ordinary_code_mse,ordinary_gradient_mse
from tools.train_aces_restorer import training_domains,domain_crop
from tools.train_aces_restorer import quota_domains


def test_domain_quotas_reject_omissions_duplicates_and_wrong_batch_size():
    rows=[{'training_domain':d} for d in ('natural','old','balanced','balanced')]
    groups,counts=quota_domains(rows,['natural=2','old=1','balanced=1'])
    assert groups=={'natural':[0],'old':[1],'balanced':[2,3]}
    assert counts=={'natural':2,'old':1,'balanced':1}
    for invalid in (['natural=2','old=2'],['natural=1','old=1','balanced=1'],
                    ['natural=2','old=1','balanced=0'],['natural=1','natural=1','old=1','balanced=1']):
        with pytest.raises(ValueError):quota_domains(rows,invalid)


def test_domain_sampling_preserves_global_target_identity_and_equal_counts():
    rows=[{'training_domain':d} for d in ['natural','synthetic','synthetic','synthetic']]
    groups=training_domains(rows)
    data=[(np.full((8,8,3),i,dtype=np.float32),np.full((8,8,3),i+10,dtype=np.float32),[],[]) for i in range(4)]
    rng=np.random.default_rng(42);counts={'natural':0,'synthetic':0}
    for _ in range(20):
        for label,indices in groups.items():
            for _ in range(2):
                source,target,(index,x,y,mode)=domain_crop(data,indices,rng,4)
                assert index in indices and (source==index).all() and (target==index+10).all()
                counts[label]+=1
    assert counts=={'natural':40,'synthetic':40}
    with pytest.raises(ValueError,match='explicit'):training_domains([{}])
    with pytest.raises(ValueError,match='two'):training_domains(rows[:1])


def test_ancestry_deduplicates_same_content_but_rejects_changed_targets():
    row={'asset_id':'a','scene_id':'s','split':'train','sdr_sha256':'x','target_sha256':'y'}
    assert merge_training_ancestry([row],[row])==[row]
    with pytest.raises(ValueError,match='conflicting'):merge_training_ancestry([row],[{**row,'target_sha256':'z'}])
    with pytest.raises(ValueError,match='TRAIN'):merge_training_ancestry([],[{**row,'split':'val'}])


def test_mixed_loss_ignores_reconstruction_and_empty_support():
    source=torch.tensor([.5,1.]).reshape(1,1,1,2).expand(1,3,1,2)
    restored=source.clone().requires_grad_();target=source+.2/255
    loss,pixels=ordinary_code_mse(source,restored,target,0)
    assert pixels==1 and float(loss.detach())==pytest.approx(.04,abs=1e-5)
    loss.backward();assert (restored.grad[:,:,:,1]==0).all()
    loss,pixels=ordinary_code_mse(torch.ones_like(source),restored,target,0)
    assert pixels==0 and loss==0


def test_gradient_target_keeps_true_edges_and_excludes_reconstructed_endpoints():
    source=torch.full((1,3,2,3),.5)
    target=source.clone();target[:,:,:,-1]+=.4/255
    assert ordinary_gradient_mse(source,target,target,0)==0
    assert ordinary_gradient_mse(source,source,target,0)>0
    source[:,:,:,-1]=1
    assert ordinary_gradient_mse(source,torch.full_like(source,.5),target,0)==0
