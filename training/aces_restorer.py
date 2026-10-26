"""Bounded source restoration from local differences measured in code units."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalCodeRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=2,dilation=2),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=4,dilation=4),nn.SiLU())
        self.head=nn.Conv2d(16,3,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,source):
        features=[source]
        for size in (5,17,65):
            radius=size//2
            mean=F.avg_pool2d(F.pad(source,(radius,radius,0,0),mode='replicate'),(1,size),stride=1)
            mean=F.avg_pool2d(F.pad(mean,(0,0,radius,radius),mode='replicate'),(size,1),stride=1)
            features.append(((mean-source)*255).clamp(-4,4))
        return self.head(self.body(torch.cat(features,1)))
