"""Experimental nonnegative-luminance, signed-chroma shadow reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F
from training.aces_losses import shadow_reference_scale

LUMA=(.2722287168,.6740817658,.0536895174)


def luminance(rgb):
    return (rgb*rgb.new_tensor(LUMA)[None,:,None,None]).sum(1,keepdim=True)


def compose_shadow(baseline,fields,weight):
    """Square-root luminance residual, with two independent signed chroma axes.

    For nonnegative baseline luminance, Y'=scale*(sqrt(Y/scale)+f0)^2.
    Algebraic delta form preserves the exact baseline at zero initialization.
    This is not RGB clipping, a positive luminance floor, or a radiance ceiling.
    Floating point cancellation still requires monitoring in actual inference.
    """
    scale=shadow_reference_scale();y=luminance(baseline)
    dy=scale*(2*torch.sqrt(y.clamp_min(0)/scale)*fields[:,:1]+fields[:,:1].square())
    dr=dy+scale*fields[:,1:2];db=dy+scale*fields[:,2:3]
    dg=(dy-LUMA[0]*dr-LUMA[2]*db)/LUMA[1]
    return baseline+weight*torch.cat((dr,dg,db),1)


class ShadowFields(nn.Module):
    def __init__(self,wide=False):
        super().__init__()
        self.wide=wide
        self.layers=nn.Sequential(nn.Conv2d(18 if wide else 6,16,3,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=2,dilation=2),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=4,dilation=4),nn.SiLU(),nn.Conv2d(16,3,3,padding=1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,source,baseline):
        features=torch.cat((source*255,torch.asinh(baseline/shadow_reference_scale())),1)
        if self.wide:
            context=[]
            for kernel in (17,65):
                radius=kernel//2
                pooled=F.avg_pool2d(F.pad(features,(radius,radius,0,0),mode='replicate'),(1,kernel),stride=1)
                pooled=F.avg_pool2d(F.pad(pooled,(0,0,radius,radius),mode='replicate'),(kernel,1),stride=1)
                context.append(pooled)
            features=torch.cat((features,*context),1)
        return self.layers(features)
