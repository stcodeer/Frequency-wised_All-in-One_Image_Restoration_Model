import torch
from torch import nn
from torch.nn import functional as F

import math


class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        
        self.param_free_norm = nn.LayerNorm(dim, eps=0.001, elementwise_affine=False)
        self.mlp_gamma = nn.Linear(inter_dim, dim)#.cuda()
        self.mlp_beta = nn.Linear(inter_dim, dim)#.cuda()

    def forward(self, x, inter):
        # x: B H*W dim
        # inter: B H*W inter_dim

        gamma = self.mlp_gamma(inter)
        beta = self.mlp_beta(inter)

        out = self.param_free_norm(x)
        out = out * (1.0 + gamma) + beta

        return out


if __name__ == "__main__":
    import os
    import sys
    p = os.path.join('.')
    sys.path.append(os.path.abspath(p))
    from option import options as opt
    import torch
    
    
    B = 2
    H = W = 8
    C = 16
    
    x = torch.zeros((B, H*W, C))
    inter = torch.zeros((B, H*W, C))
    
    model = SelfModulatedLayerNorm(C, C)
    
    normed = model(x, inter)