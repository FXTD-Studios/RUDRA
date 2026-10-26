"""Experimental isolated fully-clipped branch; no production default change."""
import copy
import torch
from training.aces_context_model import ACESContextModel


def fully_clipped_support(source,weight):
    return (weight>0)&(source.amin(1,keepdim=True)>=254.5/255)


class ACESFullContextModel(ACESContextModel):
    def __init__(self,channels=8):
        super().__init__(channels)
        self.full_context=copy.deepcopy(self.highlight_context)
        for name,p in self.named_parameters():p.requires_grad_(name.startswith('full_context.'))

    @classmethod
    def from_parent(cls,parent):
        if parent.contract()['id']!='aces2_ap1_reconstruction_v8_highlight_context_experimental':raise ValueError('v8 parent required')
        result=cls(parent.predictor.stem.out_channels)
        result.load_state_dict({**parent.state_dict(),**{'full_context.'+k:v for k,v in parent.highlight_context.state_dict().items()}},strict=True)
        return result

    def contract(self):
        result=super().contract()
        result.update(id='aces2_ap1_reconstruction_v9_full_context_experimental',
            full_context='independent_pixel_rms',full_support='reconstruction_weight>0 and min source>=254.5/255',
            frozen_parent_fields=True,full_transition='hard source support; boundary and temporal qualification required')
        return result

    def predict_highlights(self,source,baseline,restored,weight):
        with torch.no_grad():parent,fields=super().predict_highlights(source,baseline,restored,weight)
        full_fields=self.full_context(source,baseline)[:,3:]
        full=restored+weight*((restored.abs().amax(1,keepdim=True)+.001)*full_fields)
        support=fully_clipped_support(source,weight)
        return torch.where(support,full,parent),torch.where(support,full_fields,fields)
