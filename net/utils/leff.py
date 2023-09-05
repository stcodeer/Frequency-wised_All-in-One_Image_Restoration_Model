import torch
import torch.nn as nn
import math
from einops import rearrange

from .deform_conv import DCN_layer

class FastLeFF(nn.Module):
    
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False, degradation_dim=-1, deform_conv=False):
        super().__init__()
        
        self.deform_conv = deform_conv
        
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        if deform_conv:
            self.linear_inter = nn.Sequential(nn.Linear(degradation_dim, hidden_dim),
                                act_layer())
            self.conv = nn.Sequential(DCN_layer(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                            act_layer())
        else:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                            act_layer())
            
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x, inter=None):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        if self.deform_conv:
            inter = self.linear_inter(inter)
            inter = rearrange(inter, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
            x = self.conv[0](x, inter)
            x = self.conv[1](x)
        else:
            x = self.conv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x