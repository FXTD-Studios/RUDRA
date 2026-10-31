"""Experimental AP1 targets with exact ACES2 view and source-preserving support.

Ordinary pixels use a bounded clean-code inverse. Missing-range targets use
scene-linear reference ratios fitted to observed boundary pixels. These are
virtual grading targets, not recovered physical exposure. Not the legacy corpus.
"""
import numpy as np
import torch
from rudra.aces_view import ACES2View
from training.aces_model import reconstruction_weight

LUMA_AP1 = np.array([.2722287168,.6740817658,.0536895174],np.float32)


def build_target(source, clean, plate_ap1, view=None, boundary_method='scalar',bounded_shadows=False,region_shadows=False,direct_shadow_reference=False,smooth_shadows=False):
    if smooth_shadows and (region_shadows or bounded_shadows or not direct_shadow_reference):
        raise ValueError('smooth shadows require direct reference and no other bound')
    if direct_shadow_reference and not (region_shadows or smooth_shadows):
        raise ValueError('direct shadow reference requires region brightness bound')
    if bounded_shadows and region_shadows:raise ValueError('choose one shadow bound')
    if boundary_method not in ('scalar','poisson_asinh'):
        raise ValueError('unknown boundary method')
    if source.shape!=clean.shape or source.shape!=plate_ap1.shape or source.shape[-1]!=3:
        raise ValueError('matching HWC RGB arrays required')
    if not all(np.isfinite(x).all() for x in (source,clean,plate_ap1)):
        raise ValueError('finite inputs required')
    view=view or ACES2View()
    corrected=np.clip(clean,source-.49/255,source+.49/255).clip(0,1).astype(np.float32)
    restored=view.inverse(corrected)
    clean_inverse=view.inverse(clean)
    source_tensor=torch.from_numpy(source.copy()).permute(2,0,1)[None]
    weight=reconstruction_weight(source_tensor)[0,0].numpy()
    top=source.max(-1);clean_top=clean.max(-1)
    clean_y=clean_inverse@LUMA_AP1;plate_y=plate_ap1@LUMA_AP1
    target=restored.copy();fits={}
    for region,boundary,missing in [
        ('highlight',(clean_top>=.90)&(clean_top<.98),top>.5),
        ('shadow',(clean_top>=5/255)&(clean_top<16/255),top<=.5)]:
        valid=boundary&(clean_y>1e-9)&(plate_y>1e-9)
        gain=None
        if valid.sum()>=200:
            gain=float(np.exp(np.median(np.log(clean_y[valid]/plate_y[valid]))))
            reference=plate_ap1*gain
            if region=='shadow' and direct_shadow_reference:
                # Diagnostic: isolate reference detail from quantized baseline
                # blend and Poisson correction. Same source-defined support;
                # final connected-region exposure bound is mandatory.
                support=(weight>0)&missing
                target[support]=reference[support]
                fits[region]={'fit_pixels':int(valid.sum()),'gain':gain}
                continue
            if region=='shadow':
                cap=float(view.inverse(np.full((1,3),5/255,np.float32))[0]@LUMA_AP1)
                reference=reference*np.minimum(1,cap/np.maximum(reference@LUMA_AP1,1e-12))[...,None]
            blend=(weight*missing)[...,None]
            if boundary_method=='poisson_asinh':
                from pipeline.aces_boundary import match_boundary
                reference=match_boundary(restored,reference,blend[...,0]>0)
            # Unlike the BT1886 target, don't force reference highlights above
            # the ACES inverse of white: that inverse already has a large value.
            target=target*(1-blend)+reference*blend
        fits[region]={'fit_pixels':int(valid.sum()),'gain':gain}
    if bounded_shadows:
        # A luminance cap alone cannot prevent coloured channels lifting.
        # Project viewed shadow RGB onto the same sub-code allowance as the
        # restoration branch; fractional dark values still carry grading detail.
        shadow=(weight>0)&(top<=.5)
        if shadow.any():
            codes=view.forward(target[shadow])
            codes=np.clip(codes,0,np.minimum(1,source[shadow]+.49/255))
            target[shadow]=view.inverse(codes.astype(np.float32))
    if region_shadows:
        from pipeline.aces_shadow_regions import bound_shadow_regions
        target,stats=bound_shadow_regions(target,source,(weight>0)&(top<=.5),view)
        fits['shadow_regions']=stats
    if smooth_shadows:
        from pipeline.aces_shadow_smooth import bound_smooth_shadows
        # Outside-mask values are restored baseline, but exposure anchors need
        # the same reference gain used inside each shadow region.
        gain=fits['shadow']['gain']
        if gain is not None:
            shadow=(weight>0)&(top<=.5)
            adjusted,stats=bound_smooth_shadows(plate_ap1*gain,restored,source,shadow,view)
            target[shadow]=adjusted[shadow]
            fits['shadow_smooth']=stats
    return {'target_acescg':target.astype(np.float32),'restored_acescg':restored,
            'corrected_source':corrected,'weight':weight,'fits':fits}
