"""Research v8: independently trainable highlight context; v7 source/shadows frozen."""
import copy
import torch
from training.aces_model import ACESModel


class ACESContextModel(ACESModel):
    def __init__(self,channels=8):
        super().__init__(channels,'pixel_rms','local_codes','analytic')
        self.highlight_context=copy.deepcopy(self.predictor)
        for name,p in self.named_parameters():p.requires_grad_(name.startswith('highlight_context.'))

    @classmethod
    def from_parent(cls,parent):
        if parent.contract()['id']!='aces2_ap1_reconstruction_v7_analytic_shadow_experimental':raise ValueError('v7 parent required')
        result=cls(parent.predictor.stem.out_channels)
        state={**parent.state_dict(),**{'highlight_context.'+k:v for k,v in parent.predictor.state_dict().items()}}
        result.load_state_dict(state,strict=True)
        return result

    def contract(self):
        metadata=super().contract()
        metadata.update(id='aces2_ap1_reconstruction_v8_highlight_context_experimental',
            highlight_context='independent_pixel_rms',highlight_support='source reconstruction_weight>0 and max source>.5',
            frozen_parent_fields=True,temporal_inference=False,
            temporal_training='known subpixel planar-motion residual consistency; no recurrent inference')
        return metadata

    def predict_highlights(self,source,baseline,restored,weight):
        fields=self.highlight_context(source,baseline)[:,3:]
        return restored+weight*((restored.abs().amax(1,keepdim=True)+.001)*fields),fields

    def forward(self,source):
        with torch.no_grad():output=super().forward(source)
        high=(output.reconstruction_weight>0)&(source.amax(1,keepdim=True)>.5)
        predicted,fields=self.predict_highlights(source,output.baseline,output.restored_acescg,output.reconstruction_weight)
        output.acescg=torch.where(high,predicted,output.acescg)
        output.fields=torch.cat((output.fields[:,:3],torch.where(high,fields,output.fields[:,3:])),1)
        return output


def aligned_pair_residual_loss(prediction,target,high_mask,dx=.5,dy=.25,border=4,return_count=False):
    """Pair0/1 known translation: bilinear sample frame1 at (x+dx,y+dy).

    Compare prediction errors relative to each frame's own HDR target. Requiring
    all four bilinear neighbours valid avoids comparing known to clipped pixels.
    This is deterministic slicing, not a flow estimator or a motion inference step.
    """
    if prediction.shape[0]!=2 or not (0<=dx<=1 and 0<=dy<=1):raise ValueError('two frames and unit subpixel shift required')
    residual=torch.asinh(prediction)-torch.asinh(target)
    b=residual[1:2]
    aligned=(1-dy)*((1-dx)*b[:,:,:-1,:-1]+dx*b[:,:,:-1,1:])+dy*((1-dx)*b[:,:,1:,:-1]+dx*b[:,:,1:,1:])
    m=high_mask[1:2]
    valid=high_mask[:1,:,:-1,:-1]&m[:,:,:-1,:-1]&m[:,:,:-1,1:]&m[:,:,1:,:-1]&m[:,:,1:,1:]
    if border:
        valid=valid.clone();valid[:,:,:border]=False;valid[:,:,-border:]=False;valid[:,:,:,:border]=False;valid[:,:,:,-border:]=False
    error=(residual[:1,:,:-1,:-1]-aligned).square().mean(1,keepdim=True)
    count=valid.sum();loss=(error*valid).sum()/count.clamp_min(1)
    return (loss,count) if return_count else loss
