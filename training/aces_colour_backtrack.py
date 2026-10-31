"""Experimental partial-channel highlight appearance constraint (NumPy/OCIO).

No radiance clipping. Discrete per-pixel backtracking may introduce temporal or
spatial discontinuities; this is a diagnostic, never a production guarantee.
"""
import numpy as np


def constrain_colour(source,prediction,baseline,weight,view,include_fully_clipped=False,repair_inverse_boundary=False,forbid_new_negative_y=False):
    source=np.asarray(source,dtype=np.float32)
    prediction=np.asarray(prediction,dtype=np.float32);baseline=np.asarray(baseline,dtype=np.float32)
    weight=np.asarray(weight,dtype=np.float32);result=prediction.copy()
    if result.shape!=source.shape or baseline.shape!=source.shape or weight.shape!=source.shape[:2]:
        raise ValueError('matching source, prediction, baseline and weight required')
    if not all(np.isfinite(x).all() for x in (source,prediction,baseline,weight)):raise ValueError('finite arrays required')
    support=(weight>0)&(source.max(-1)>.5)
    if not include_fully_clipped:support&=source.min(-1)<254.5/255
    alpha=np.ones(source.shape[:2],np.float32)
    active=support.copy()
    luma=np.array([.2722287168,.6740817658,.0536895174],np.float64)  # same as the audits' LUMA_AP1
    # Optional guard: a support pixel whose baseline luminance is >=0 must not
    # come out with negative luminance. Same backtracking towards the baseline.
    guard=forbid_new_negative_y
    baseline_y_ok=(baseline@luma>=0) if guard else None
    # Evaluate only active pixels. The declared OCIO transform is pointwise.
    for step in range(9):
        if not active.any():break
        indices=np.flatnonzero(active)
        codes=view.forward(result.reshape(-1,3)[indices])
        error=np.abs(codes-source.reshape(-1,3)[indices]).max(-1)*255
        ok=error<=1
        if guard:ok&=(result.reshape(-1,3)[indices]@luma>=0)|~baseline_y_ok.reshape(-1)[indices]
        active.reshape(-1)[indices[ok]]=False
        if not active.any():break
        alpha[active]=alpha[active]*.5 if step<8 else 0
        result[active]=baseline[active]+alpha[active,None]*(prediction[active]-baseline[active])
    error=np.abs(view.forward(result)-source).max(-1)*255
    repaired=0
    if repair_inverse_boundary:
        # For failing saturated-channel pixels only, try one16-bit code inside
        # the display cube. Accept only measured appearance and luminance fixes.
        failing=support&(error>1)&(source.max(-1)==1)
        if failing.any():
            indices=np.flatnonzero(failing);target=source.reshape(-1,3)[indices]
            candidate=view.inverse(np.minimum(target,1-1/65535))
            candidate_error=np.abs(view.forward(candidate)-target).max(-1)*255
            luma=np.array([.2722287168,.6740817658,.0536895174],np.float32)
            original_y=baseline.reshape(-1,3)[indices]@luma
            good=np.isfinite(candidate).all(-1)&(candidate_error<=1)&((original_y<0)|(candidate@luma>=0))
            result.reshape(-1,3)[indices[good]]=candidate[good];repaired=int(good.sum())
            error=np.abs(view.forward(result)-source).max(-1)*255
    luma=np.array([.2722287168,.6740817658,.0536895174],np.float64)
    new_negative=int((support&(result@luma<0)&(baseline@luma>=0)).sum())
    return result,{'include_fully_clipped':include_fully_clipped,'forbid_new_negative_y':bool(guard),'new_negative_luminance':new_negative,'support_pixels':int(support.sum()),'changed_pixels':int((support&(alpha<1)).sum()),
                   'baseline_fallback_pixels':int((support&(alpha==0)).sum()),
                   'remaining_display_errors':int((support&(error>1)).sum()),'inverse_boundary_repaired_pixels':repaired,
                   'outside_support_bit_exact':bool(np.array_equal(result[~support],prediction[~support]))}
