import numpy as np
import pytest
pytest.importorskip('PyOpenColorIO')
from pipeline.aces_targets import build_target
from rudra.aces_view import ACES2View


def test_ordinary_clean_reconstruction_preserves_observed_source_budget():
    source=np.full((24,24,3),.5,np.float32)
    clean=source+.02
    pair=build_target(source,clean,np.ones_like(source))
    assert not pair['weight'].any()
    restored=ACES2View().forward(pair['target_acescg'])
    assert np.max(np.abs(restored-source))*255 < 1


def test_reference_can_be_below_analytic_white_instead_of_forced_brighter():
    clean=np.repeat(np.linspace(.90,1,64,dtype=np.float32)[None,:,None],64,axis=0)
    clean=np.repeat(clean,3,axis=2)
    source=np.rint(clean*255)/255
    plate=np.ones_like(source)
    pair=build_target(source,clean,plate)
    assert pair['fits']['highlight']['gain'] is not None
    assert (pair['target_acescg'][:,-1] < pair['restored_acescg'][:,-1]).all()


def test_boundary_candidate_preserves_ordinary_values_and_changes_only_missing_support():
    clean=np.repeat(np.linspace(.90,1,64,dtype=np.float32)[None,:,None],64,axis=0)
    clean=np.repeat(clean,3,axis=2)
    source=np.rint(clean*255)/255
    old=build_target(source,clean,np.ones_like(source))
    new=build_target(source,clean,np.ones_like(source),boundary_method='poisson_asinh')
    ordinary=old['weight']==0
    np.testing.assert_array_equal(new['target_acescg'][ordinary],old['target_acescg'][ordinary])
    assert np.isfinite(new['target_acescg']).all()
    assert np.any(new['target_acescg'][~ordinary]!=old['target_acescg'][~ordinary])


def test_bounded_shadows_keep_fractional_detail_without_visible_channel_lift():
    clean=np.full((32,32,3),10/255,np.float32)
    source=clean.copy();source[8:24,8:24]=0
    plate=np.ones_like(source)*.01
    plate[8:24,8:24]*=np.linspace(.0001,.01,16,dtype=np.float32)[None,:,None]
    pair=build_target(source,clean,plate,bounded_shadows=True)
    codes=ACES2View().forward(pair['target_acescg'])
    assert ((codes-source)*255).max()<1
    dark=pair['target_acescg'][8:24,8:24]
    assert np.isfinite(dark).all() and dark.max()>0
    assert np.ptp(dark)>0


def test_direct_reference_preserves_shadow_ratios_with_verified_bound():
    source=np.full((32,32,3),10/255,np.float32)
    clean=source.copy();source[8:24,8:24]=0
    plate=np.full_like(source,.01)
    plate[8:24,8:24]*=np.linspace(.1,1,16,dtype=np.float32)[None,:,None]
    pair=build_target(source,clean,plate,region_shadows=True,direct_shadow_reference=True)
    output=pair['target_acescg'];ratios=output[8:24,8:24]/plate[8:24,8:24]
    np.testing.assert_allclose(ratios,ratios.flat[0],rtol=3e-7)
    assert ratios.min()>0
    assert ((ACES2View().forward(output)-source)*255).max()<1
    with pytest.raises(ValueError,match='requires region'):
        build_target(source,clean,plate,direct_shadow_reference=True)
