import torch
from training.aces_model import ACESModel,model_from_checkpoint,shadow_support
from training.aces_shadow import luminance


def scene():
    torch.manual_seed(5)
    x=torch.rand(1,3,24,32)*.3+.35
    x[...,:8,:]=torch.rand(1,3,8,32)*.012      # crushed shadow band
    x[...,8:12,:]=.03                           # shadow ramp edge (partial weight)
    x[...,-4:,:]=.995                            # clipped highlight
    return x


def randomize(model):
    torch.manual_seed(11)
    with torch.no_grad():
        for p in model.parameters():p.add_(torch.randn_like(p)*.05)
    return model


def pair():
    shared=randomize(ACESModel(8,'pixel_rms','local_codes','shared')).eval()
    analytic=ACESModel(8,'pixel_rms','local_codes','analytic').eval()
    analytic.load_state_dict(shared.state_dict(),strict=True)
    return shared,analytic


def test_analytic_shadow_only_changes_shadow_support_and_is_exact_inside():
    shared,analytic=pair();x=scene()
    with torch.no_grad():a=shared(x);b=analytic(x)
    support=shadow_support(x,b.reconstruction_weight)
    assert support.any() and (~support).any()
    assert torch.equal(torch.where(support,0,a.acescg),torch.where(support,0,b.acescg))
    full=support&(b.reconstruction_weight>=1)
    assert full.any()
    assert torch.equal(torch.where(full,b.acescg,0),torch.where(full,b.baseline,0))


def test_analytic_shadow_never_negative_luminance_where_baseline_is_not():
    _,analytic=pair();x=scene()
    with torch.no_grad():b=analytic(x)
    support=shadow_support(x,b.reconstruction_weight)
    assert (luminance(b.baseline)[support]>=0).all()
    assert (luminance(b.acescg)[support]>=-1e-12).all()


def test_analytic_shadow_contract_round_trips_shared_weights():
    shared,analytic=pair()
    state={'model':shared.state_dict(),'contract':analytic.contract()}
    loaded=model_from_checkpoint(state)
    assert loaded.shadow=='analytic' and loaded.contract()['shadow_reconstructed'] is False


def test_analytic_shadow_display_stays_within_one_code_of_source():
    from training.aces_reference import reference_view
    _,analytic=pair();x=scene()
    with torch.no_grad():b=analytic(x)
    support=shadow_support(x,b.reconstruction_weight)[0,0].numpy()
    shown=reference_view().forward(b.acescg[0].permute(1,2,0).numpy())
    source=x[0].permute(1,2,0).numpy()
    assert (abs(shown-source).max(-1)[support]*255).max()<=1


def test_cached_inverse_matches_previous_composition_bit_exact(monkeypatch):
    import training.aces_model as module
    _,model=pair();x=scene();original=module.inverse_aces2_reference;calls=[]
    def counted(source):
        calls.append(1)
        return original(source)
    monkeypatch.setattr(module,'inverse_aces2_reference',counted)
    with torch.no_grad():
        result=model(x)
        assert len(calls)==2  # source and restored source; no redundant third inverse
        cached=result.acescg.clone()
        result.restored_acescg=None
        previous=module.analytic_shadow(x,result)
    assert torch.equal(cached,previous)
