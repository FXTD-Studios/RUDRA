"""Research-only frozen-parent residual with whole-image coarse conditioning."""
import torch
from torch import nn
from torch.nn import functional as F
from training.aces_full_context_model import ACESFullContextModel,fully_clipped_support


class ContextResidual(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.encoder=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.head=nn.Conv2d(16,3,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,conditioning):
        if self.mode=='coarse4_spatial':conditioning=F.adaptive_avg_pool2d(conditioning,(4,4))
        features=self.encoder(conditioning)
        if self.mode=='coarse64_global_mean':features=features+features.mean((2,3),keepdim=True)
        return self.head(features)


class ACESWideContextModel(ACESFullContextModel):
    def __init__(self,channels=8,mode='coarse64_global_mean'):
        if mode not in ('coarse64_global_mean','coarse4_spatial','local5'):raise ValueError('context mode')
        super().__init__(channels);self.context_mode=mode;self.wide_context=ContextResidual(mode)
        for name,p in self.named_parameters():p.requires_grad_(name.startswith('wide_context.'))

    @classmethod
    def from_parent(cls,parent,mode='coarse64_global_mean'):
        if parent.contract()['id']!='aces2_ap1_reconstruction_v9_full_context_experimental':raise ValueError('v9 parent required')
        model=cls(parent.predictor.stem.out_channels,mode)
        state=model.state_dict();state.update(parent.state_dict());model.load_state_dict(state,strict=True)
        return model

    def contract(self):
        result=super().contract();result.update(id='aces2_ap1_reconstruction_v10_wide_context_experimental',wide_context=self.context_mode,
            context_channels=16,context_initial='zero_output',context_parent_frozen=True,context_conditioning='full_image_before_crop',context_support='same_fully_clipped_support')
        return result

    def conditioning(self,source,baseline):
        inputs=torch.cat((source,torch.asinh(baseline)),1)
        if self.context_mode!='local5':inputs=F.adaptive_avg_pool2d(inputs,(64,64))
        return inputs

    def context_fields(self,conditioning,full_shape,crop=None):
        fields=self.wide_context(conditioning)
        if self.context_mode!='local5':fields=F.interpolate(fields,size=full_shape,mode='bilinear',align_corners=False)
        if crop is not None:
            y,x,h,w=crop;fields=fields[:,:,y:y+h,x:x+w]
        return fields

    @staticmethod
    def apply_context(source,parent_prediction,restored,weight,fields):
        support=fully_clipped_support(source,weight)
        delta=weight*(restored.abs().amax(1,keepdim=True)+.001)*fields
        return torch.where(support,parent_prediction+delta,parent_prediction)

    def predict_highlights(self,source,baseline,restored,weight):
        with torch.no_grad():parent,fields=super().predict_highlights(source,baseline,restored,weight)
        extra=self.context_fields(self.conditioning(source,baseline),source.shape[-2:])
        predicted=self.apply_context(source,parent,restored,weight,extra)
        return predicted,torch.where(fully_clipped_support(source,weight),fields+extra,fields)
