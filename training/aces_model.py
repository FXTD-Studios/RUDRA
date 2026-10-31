"""Experimental ACEScg model/composition; incompatible with legacy checkpoints.

Input is Rec709 Rec1886 display RGB codes, output is ACEScg linear. The exact
OCIO CPU decoder stays outside the tensor-only predictor. No physical source
exposure is inferred and no native v1 compatibility is claimed.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from rudra.sdr2hdr import ResidualBlock
from rudra.reconstruction_policy import reconstruction_bands
from training.aces_reference import inverse_aces2_reference, reference_view


CONTRACT = 'aces2_ap1_reconstruction_v2_experimental'


class PixelRMSNorm(nn.Module):
    """Channel-only normalization; unrelated image regions cannot alter it."""
    def __init__(self,channels,eps=1e-5):
        super().__init__();self.eps=eps
        self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))

    def forward(self,x):
        y=x*torch.rsqrt(x.square().mean(1,keepdim=True)+self.eps)
        return y*self.weight[None,:,None,None]+self.bias[None,:,None,None]


def replace_spatial_norm(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.GroupNorm):setattr(module,name,PixelRMSNorm(child.num_channels,child.eps))
        else:replace_spatial_norm(child)


def reconstruction_weight(source):
    hi, lo = reconstruction_bands('observed_clip_v1')
    top = source.amax(1,keepdim=True)
    def smooth(band):
        t = ((top-band[0])/(band[1]-band[0])).clamp(0,1)
        return t.square()*(3-2*t)
    evidence = torch.maximum(smooth(hi),1-smooth(lo))
    return ((evidence-.5)*2).clamp(0,1)


@dataclass
class ACESOutput:
    acescg: torch.Tensor
    baseline: torch.Tensor
    restored_source: torch.Tensor
    reconstruction_weight: torch.Tensor
    fields: torch.Tensor
    restored_acescg: torch.Tensor | None = None


def compose(source, fields, baseline=None):
    if source.ndim!=4 or source.shape[1]!=3 or fields.shape!=(source.shape[0],6,*source.shape[2:]):
        raise ValueError('expected matching BCHW source3 and fields6')
    source, fields = source.float(), fields.float()
    if not torch.isfinite(source).all() or not torch.isfinite(fields).all() or ((source<0)|(source>1)).any():
        raise ValueError('finite fields and source codes in[0,1] required')
    if baseline is None:
        baseline = inverse_aces2_reference(source)
    restored_source = (source + (.49/255)*torch.tanh(fields[:,:3])).clamp(0,1)
    restored = inverse_aces2_reference(restored_source)
    weight = reconstruction_weight(source)
    # Signed AP1 addition allows recovery from exact black and preserves any
    # signed reference values. .001 is a numerical AP1 residual scale floor,
    # not a luminance floor or inferred nit value. No radiance output clipping.
    # A multiplicative exponential residual cannot represent sufficiently
    # negative AP1 channels, even with a wider stop range. Share the RGB scale
    # so a near-zero channel can reconstruct colour from the same pixel's
    # signal. Linear signed fields have no asymmetric floor or stop ceiling.
    residual = (restored.abs().amax(1,keepdim=True)+.001)*fields[:,3:]
    result = restored + weight*residual
    return ACESOutput(result,baseline,restored_source,weight,fields,restored)


class ACESFields(nn.Module):
    """Six raw fields: three bounded source-code corrections, three AP1 residuals."""
    def __init__(self, channels=32):
        super().__init__()
        c=channels
        self.stem=nn.Conv2d(6,c,3,padding=1)
        self.enc1=nn.Sequential(ResidualBlock(c),ResidualBlock(c))
        self.down1=nn.Conv2d(c,2*c,3,stride=2,padding=1)
        self.enc2=nn.Sequential(ResidualBlock(2*c),ResidualBlock(2*c))
        self.down2=nn.Conv2d(2*c,4*c,3,stride=2,padding=1)
        self.mid=nn.Sequential(ResidualBlock(4*c),ResidualBlock(4*c))
        self.up2=nn.Conv2d(6*c,2*c,3,padding=1)
        self.dec2=nn.Sequential(ResidualBlock(2*c),ResidualBlock(2*c))
        self.up1=nn.Conv2d(3*c,c,3,padding=1)
        self.dec1=nn.Sequential(ResidualBlock(c),ResidualBlock(c))
        self.head=nn.Conv2d(c,6,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,source,baseline):
        x=torch.cat((source,torch.asinh(baseline)),dim=1)
        e1=self.enc1(self.stem(x));e2=self.enc2(self.down1(e1));m=self.mid(self.down2(e2))
        d2=self.dec2(self.up2(torch.cat((F.interpolate(m,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2),1)))
        d1=self.dec1(self.up1(torch.cat((F.interpolate(d2,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1),1)))
        return self.head(d1)


class ACESModel(nn.Module):
    def __init__(self,channels=32,normalization='group',restoration='shared',shadow='shared'):
        super().__init__()
        if normalization not in ('group','pixel_rms'):raise ValueError('unknown normalization')
        self.normalization=normalization
        if restoration not in ('shared','local_codes'):raise ValueError('unknown restoration')
        self.restoration=restoration
        if shadow not in ('shared','sqrt_luma','sqrt_luma_context','analytic'):raise ValueError('unknown shadow representation')
        self.shadow=shadow
        if shadow in ('sqrt_luma','sqrt_luma_context'):
            from training.aces_shadow import ShadowFields
            self.shadow_predictor=ShadowFields(wide=shadow=='sqrt_luma_context')
        self.predictor=ACESFields(channels)
        if normalization=='pixel_rms':replace_spatial_norm(self.predictor)
        if restoration=='local_codes':
            from training.aces_restorer import LocalCodeRestorer
            self.restorer=LocalCodeRestorer()

    def contract(self):
        metadata={'id':CONTRACT,**reference_view().identity,
                'input':'Rec709 Rec1886 display RGB codes', 'output':'ACEScg linear',
                'source_code_bound':.49/255,'reconstruction_policy':'observed_clip_v1',
                'residual':'signed_linear_shared_rgb_scale','residual_scale_floor':.001,
                'radiance_clamp':False,'scene_exposure_recovered':False,'release_authorized':False}
        if self.normalization!='group':
            metadata.update(id='aces2_ap1_reconstruction_v3_pixel_rms_experimental',normalization=self.normalization)
        if self.restoration=='local_codes':
            metadata.update(id='aces2_ap1_reconstruction_v4_local_codes_experimental',
                            normalization=self.normalization,restoration='local_codes',
                            restoration_support='source reconstruction_weight == 0',
                            restoration_context_kernels=[5,17,65])
        if self.shadow in ('sqrt_luma','sqrt_luma_context'):
            from training.aces_losses import shadow_reference_scale
            metadata.update(id='aces2_ap1_reconstruction_v5_shadow_sqrt_experimental',
                            shadow='sqrt_luma',shadow_scale=shadow_reference_scale(),
                            shadow_support='source reconstruction_weight > 0 and max source <= .5',
                            shadow_baseline='exact source inverse',shadow_residual='square-root luminance and signed R-Y/B-Y')
            if self.shadow=='sqrt_luma_context':
                metadata.update(id='aces2_ap1_reconstruction_v6_shadow_context_experimental',
                                shadow='sqrt_luma_context',shadow_context_kernels=[17,65],shadow_context_radius=40)
        if self.shadow=='analytic':
            metadata.update(id='aces2_ap1_reconstruction_v7_analytic_shadow_experimental',shadow='analytic',
                            shadow_support='source reconstruction_weight > 0 and max source <= .5',
                            shadow_output='w*exact source inverse + (1-w)*exact inverse of bounded restored source',
                            shadow_reconstructed=False,
                            shadow_reason='5-seed study 24 Oct 2026: no trained shadow branch reliably beat the analytic inverse')
        return metadata

    def forward(self,source):
        source=source.float()
        baseline=inverse_aces2_reference(source)
        fields=self.predictor(source,baseline)
        if self.restoration=='local_codes':
            ordinary=reconstruction_weight(source)==0
            fields=torch.cat((torch.where(ordinary,self.restorer(source),fields[:,:3]),fields[:,3:]),1)
        output=compose(source,fields,baseline)
        if self.shadow in ('sqrt_luma','sqrt_luma_context'):
            from training.aces_shadow import compose_shadow
            support=(output.reconstruction_weight>0)&(source.amax(1,keepdim=True)<=.5)
            reconstructed=compose_shadow(baseline,self.shadow_predictor(source,baseline),output.reconstruction_weight)
            output.acescg=torch.where(support,reconstructed,output.acescg)
        elif self.shadow=='analytic':
            output.acescg=analytic_shadow(source,output)
        return output


def shadow_support(source,weight):
    return (weight>0)&(source.amax(1,keepdim=True)<=.5)


def analytic_shadow(source,output):
    """No learned radiance in crushed shadows.

    Inside the source-defined shadow support the output is a convex blend of the
    exact inverse of the source (w=1, exact round trip, never lifted) and the exact
    inverse of the bounded restored source (w->0, continuous with ordinary pixels).
    At w==1 the result is bit-exact with the analytic baseline.
    """
    w=output.reconstruction_weight
    restored=output.restored_acescg
    if restored is None:restored=inverse_aces2_reference(output.restored_source)
    blend=torch.where(w>=1,output.baseline,w*output.baseline+(1-w)*restored)
    return torch.where(shadow_support(source,w),blend,output.acescg)


def model_from_checkpoint(state):
    if state['contract'].get('wide_context'):
        from training.aces_wide_context_model import ACESWideContextModel
        model=ACESWideContextModel(state['model']['predictor.stem.weight'].shape[0],state['contract']['wide_context'])
    elif state['contract'].get('full_context')=='independent_pixel_rms':
        from training.aces_full_context_model import ACESFullContextModel
        model=ACESFullContextModel(state['model']['predictor.stem.weight'].shape[0])
    elif state['contract'].get('highlight_context')=='independent_pixel_rms':
        from training.aces_context_model import ACESContextModel
        model=ACESContextModel(state['model']['predictor.stem.weight'].shape[0])
    else:
        model=ACESModel(state['model']['predictor.stem.weight'].shape[0],
                    state['contract'].get('normalization','group'),
                    state['contract'].get('restoration','shared'),
                    state['contract'].get('shadow','shared'))
    if model.contract()!=state['contract']:raise ValueError('checkpoint contract mismatch')
    model.load_state_dict(state['model'],strict=True)
    return model
