import numpy as np
from tools.prepare_aces_shadow_data import shadow_surface


def test_original_shadow_surfaces_have_dark_values_colour_and_true_black():
    rng=np.random.default_rng(20261031)
    frames=[shadow_surface(rng,i,32)[0] for i in range(8)]
    assert all(np.isfinite(f).all() and f.min()>=0 and f.max()<.04 for f in frames)
    assert any(f.min()<1e-5 and f.max()>1e-3 for f in frames)
    assert np.array_equal(frames[0][...,0],frames[0][...,1])
    assert not np.array_equal(frames[1][...,0],frames[1][...,1])
    assert np.any(np.all(frames[7]==0,axis=-1))


def test_all_training_surfaces_include_deep_shadows():
    rng=np.random.default_rng(20261031)
    for i in range(96):
        frame,_=shadow_surface(rng,i)
        assert np.min(np.max(frame,axis=-1))<1.5e-5
