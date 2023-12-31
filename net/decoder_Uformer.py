import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import einsum

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat

import math

from .utils.leff import LeFF, FastLeFF
from .utils.self_modulated_layernorm import SelfModulatedLayerNorm
from .utils.frequency_decompose import FrequencyDecompose

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, dimkv=None, kv_source=None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        if kv_source == 'attention_kv':
            self.to_k = nn.Linear(dimkv, inner_dim, bias = bias)
            self.to_v = nn.Linear(dimkv, inner_dim, bias = bias)
        elif kv_source == 'attention_residual':
            self.to_kv = nn.Linear(dimkv, inner_dim * 2, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim
        self.dimkv = dimkv
        self.kv_source = kv_source

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        
        if self.kv_source == 'attention_kv':
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)[0]
            
            embed_dim = attn_kv[0].shape[3]
            k = rearrange(attn_kv[0], 'b heads win_size embed_dim -> b win_size (heads embed_dim)')
            k = self.to_k(k)
            k = rearrange(k, 'b win_size (heads embed_dim) -> b heads win_size embed_dim', embed_dim=embed_dim)
            
            v = rearrange(attn_kv[1], 'b heads win_size embed_dim -> b win_size (heads embed_dim)')
            v = self.to_v(v)
            v = rearrange(v, 'b win_size (heads embed_dim) -> b heads win_size embed_dim', embed_dim=embed_dim)
            
            return q, k, v
            
            
        if self.kv_source == 'attention_residual':
            pass
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v


class WindowAttention(nn.Module):
    def __init__(self, input_resolution, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 frequency_decompose_type=None,
                 all_degradation_embedding_method=[],
                 degradation_dim=-1,
                 degradation_embedding_method=[],
                 debug_mode=False):
        
        import warnings
        warnings.filterwarnings('ignore')

        super().__init__()
        self.input_resolution = input_resolution
        self.num_win = input_resolution[0] // win_size[0] * input_resolution[1] // win_size[1]
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.frequency_decompose_type = frequency_decompose_type
        
        self.all_degradation_embedding_method = all_degradation_embedding_method
        
        self.debug_mode = debug_mode
        
        if not frequency_decompose_type == None:
            assert False
            if frequency_decompose_type.split('_')[-1] == 'bands':
                self.num_bands = int(frequency_decompose_type.split('_')[0])
                self.decompose = FrequencyDecompose('frequency_decompose', 1./self.num_bands, win_size[0]*win_size[1], win_size[0]*win_size[1])
            
            elif frequency_decompose_type == 'DC':
                self.num_bands = 2
                self.decompose = FrequencyDecompose('frequency_decompose_dc', 1./self.num_bands, win_size[0]*win_size[1], win_size[0]*win_size[1])
            
            # self.lamb = nn.Parameter(torch.zeros(self.num_bands, self.num_win, num_heads))
            self.lamb = nn.Parameter(torch.zeros(self.num_bands - 1, 1, num_heads))
        elif len(all_degradation_embedding_method) > 0:
            for type in all_degradation_embedding_method:
                if type.split('_')[-1] == 'bands':
                    self.num_bands = int(type.split('_')[-2])
                    self.decompose = FrequencyDecompose('frequency_decompose_1', 1./(self.num_bands-1), win_size[0]*win_size[1], win_size[0]*win_size[1])
                
                elif type.split('_')[-1] == 'DC':
                    self.num_bands = 2
                    self.decompose = FrequencyDecompose('frequency_decompose_dc', 1./self.num_bands, win_size[0]*win_size[1], win_size[0]*win_size[1])
                    
            encoder_embed_dim = 28
            
            self.mlp_head = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(encoder_embed_dim * 16),
                nn.Linear(encoder_embed_dim * 16, num_heads))
                if i > 0 else None
                for i in range(self.num_bands)])
            
            self.avg = nn.ModuleList([nn.AdaptiveAvgPool1d(1)
                        if i > 0 else None
                        for i in range(self.num_bands)])
            
            self.mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(num_heads, num_heads),
                nn.LeakyReLU(0.1, True),
                nn.Linear(num_heads, num_heads),)
                if i > 0 else None
                for i in range(self.num_bands)])
                

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='conv':
            assert False
            # self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            if 'attention_kv' in degradation_embedding_method:
                self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias, dimkv=degradation_dim, kv_source='attention_kv')
            elif 'attention_residual' in degradation_embedding_method:
                self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias, dimkv=degradation_dim, kv_source='attention_residual')
            else:
                self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
                
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, all_inter=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).long()].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        # frequency decompose here
        if not self.frequency_decompose_type == None:
            assert False
            attn_bands = self.decompose(attn) # [num_bands, B, num_heads, N, N]
            # B, num_winH, num_winW = B.view(-1, H//win_size, W//win_size)
            
            # assert torch.all(torch.abs(attn - torch.sum(attn_bands, dim=0)) < 1e-5), 'oops'
            # attn_bands = attn_bands.view(attn_bands.shape[0], 10, -1, attn_bands.shape[2], attn_bands.shape[3], attn_bands.shape[4]) * self.lamb[:, None, :, :, None, None]
            # attn_bands = attn_bands.view(attn_bands.shape[0], -1, attn_bands.shape[3], attn_bands.shape[4], attn_bands.shape[5])
            # attn_bands = attn_bands * self.lamb[:, :, :, None, None]
            attn_bands = attn_bands[1:] * self.lamb[:, :, :, None, None]
            
            attn = attn + torch.sum(attn_bands, dim=0)
        elif len(self.all_degradation_embedding_method) > 0:
            attn_bands = self.decompose(attn) # [num_bands, B*num_win, num_heads, N, N]
            attn_bands = torch.unbind(attn_bands, 0)
            
            for i in range(1, self.num_bands):
                embed_lamb = self.mlp_head[i](all_inter[i])
                embed_lamb = embed_lamb.permute(0, 2, 1)
                embed_lamb = self.avg[i](embed_lamb)
                embed_lamb = embed_lamb.permute(0, 2, 1)
                embed_lamb = self.mlp[i](embed_lamb)
                attn_band = attn_bands[i].view(-1, self.num_win, attn_bands[i].shape[1], attn_bands[i].shape[2], attn_bands[i].shape[3])
                attn_band = attn_band * embed_lamb[:, :, :, None, None]
                attn_band = attn_band.view(-1, attn_band.shape[2], attn_band.shape[3], attn_band.shape[4])
                attn = attn + attn_band
            
            
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.debug_mode and len(self.all_degradation_embedding_method) > 0:
            return x, embed_lamb
        else:
            return x, []

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                 modulator=False,cross_modulator=False,
                 frequency_decompose_type=None,
                 all_degradation_embedding_method=[],
                 degradation_dim=-1,
                 degradation_embedding_method=[],
                 debug_mode=False):
        super().__init__()
        
        self.debug_mode = debug_mode
        
        if debug_mode:
            self.visual_decompose = FrequencyDecompose('frequency_decompose', 1, input_resolution[0], input_resolution[1], inverse='visual')
        
        self.degradation_dim = degradation_dim
        self.degradation_embedding_method = degradation_embedding_method
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size*win_size, dim) # modulator
        else:
            self.modulator = None
            
        if 'modulator' in degradation_embedding_method:
            kernel_size = 1
            self.degradation_modulator = Downsample(degradation_dim, dim, kernel_size=kernel_size, stride=(input_resolution[0]//win_size, input_resolution[1]//win_size), padding=(kernel_size-1)//2)
            self.degradation_modulator_embed = nn.Linear(2 * dim, dim)#.cuda()
            self.norm_degradation_modulator = nn.Sequential(
                norm_layer(dim),
                nn.LeakyReLU(0.1, True),
            )
        else:
            self.degradation_modulator = None

        # if cross_modulator:
        #     self.cross_modulator = nn.Embedding(win_size*win_size, dim) # cross_modulator
        #     self.cross_attn = Attention(dim,num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        #             token_projection=token_projection,)
        #     self.norm_cross = norm_layer(dim)
        # else:
        #     self.cross_modulator = None

        if 'self_modulator' in degradation_embedding_method:
            self.norm1 = SelfModulatedLayerNorm(dim, degradation_dim)
            self.norm1_norm_degradation = nn.Sequential(
                norm_layer(degradation_dim),
                nn.LeakyReLU(0.1, True),
            )
        else:
            self.norm1 = norm_layer(dim)
            
        if 'attention_residual' in self.degradation_embedding_method:
            self.norm_degradation_attention = nn.Sequential(
                norm_layer(degradation_dim),
                nn.LeakyReLU(0.1, True),
            )
            
        self.attn = WindowAttention(input_resolution=input_resolution,
            dim=dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,
            frequency_decompose_type=frequency_decompose_type,
            all_degradation_embedding_method=all_degradation_embedding_method,
            degradation_dim=degradation_dim,
            degradation_embedding_method=degradation_embedding_method,
            debug_mode=debug_mode)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if 'self_modulator' in degradation_embedding_method:
            self.norm2 = SelfModulatedLayerNorm(dim, degradation_dim)
            self.norm2_norm_degradation = nn.Sequential(
                norm_layer(degradation_dim),
                nn.LeakyReLU(0.1, True),
            )
        else:
            self.norm2 = norm_layer(dim)
            
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) 
        elif token_mlp=='leff':
            if 'deform_conv' in degradation_embedding_method:
                self.mlp =  LeFF(dim, dim, act_layer=act_layer, drop=drop, degradation_dim=degradation_dim, deform_conv=True)
            else:
                self.mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        
        elif token_mlp=='fastleff':
            self.mlp =  FastLeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)    
        else:
            raise Exception("FFN error!") 


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, inter=None, inter_kv=None, all_inter=None, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask


        # if self.cross_modulator is not None:
        #     shortcut = x
        #     x_cross = self.norm_cross(x)
        #     x_cross = self.cross_attn(x, self.cross_modulator.weight)
        #     x = shortcut + x_cross
    
        shortcut = x
        
        if 'self_modulator' in self.degradation_embedding_method:
            x = self.norm1(x, self.norm1_norm_degradation(inter))
        else:
            x = self.norm1(x)

        if self.debug_mode:
            visual_freq_before = x.view(B, H, W, C).permute(0, 3, 1, 2)
            visual_freq_before = self.visual_decompose(visual_freq_before)
            visual_freq_before = visual_freq_before.squeeze(0)
            visual_freq_before = torch.mean(visual_freq_before, 0)
            visual_freq_before = torch.mean(visual_freq_before, 0)

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows,self.modulator.weight)
        else:
            wmsa_in = x_windows

        if 'modulator' in self.degradation_embedding_method:
            win_num = self.input_resolution[0] // self.win_size * self.input_resolution[1] // self.win_size
            
            modulator = self.degradation_modulator(inter)
            modulator = self.norm_degradation_modulator(modulator)
            modulator = torch.unsqueeze(modulator, dim=1)
            modulator = modulator.repeat(1, win_num, 1, 1)
            
            wmsa_in = wmsa_in.view(B, win_num, self.win_size*self.win_size, C)
            
            wmsa_in = torch.cat([modulator, wmsa_in], -1)
            wmsa_in = self.degradation_modulator_embed(wmsa_in)
            
            wmsa_in = wmsa_in.view(-1, self.win_size*self.win_size, C)
                
        # W-MSA/SW-MSA
        if 'attention_residual' in self.degradation_embedding_method:
            attn_inter = self.norm_degradation_attention(inter)
            attn_inter = attn_inter.view(B, H, W, self.degradation_dim)
            attn_inter_windows = window_partition(attn_inter, self.win_size)
            attn_inter_windows = attn_inter_windows.view(-1, self.win_size * self.win_size, self.degradation_dim)
            attn_windows, embed_lamb = self.attn(wmsa_in, attn_kv=attn_inter_windows, all_inter=all_inter, mask=attn_mask)
        elif 'attention_kv' in self.degradation_embedding_method:
            attn_windows, embed_lamb = self.attn(wmsa_in, attn_kv=inter_kv, all_inter=all_inter, mask=attn_mask)
        else:
            attn_windows, embed_lamb = self.attn(wmsa_in, all_inter=all_inter, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        if self.debug_mode:
            visual_freq_after = x.view(B, H, W, C).permute(0, 3, 1, 2)
            visual_freq_after = self.visual_decompose(visual_freq_after)
            visual_freq_after = visual_freq_after.squeeze(0)
            visual_freq_after = torch.mean(visual_freq_after, 0)
            visual_freq_after = torch.mean(visual_freq_after, 0)

        # FFN
        x = shortcut + self.drop_path(x)
        
        if 'self_modulator' in self.degradation_embedding_method:
            norm = self.norm2(x, self.norm2_norm_degradation(inter))
        else:
            norm = self.norm2(x)
        
        if 'deform_conv' in self.degradation_embedding_method:
            y = self.mlp(norm, inter)
        else:
            y = self.mlp(norm)
        
        x = x + self.drop_path(y)
        del attn_mask
        if self.debug_mode:
            return x, [visual_freq_before, visual_freq_after, embed_lamb]
        else:
            return x


#########################################
########### Basic layer of Uformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 modulator=False,cross_modulator=False,
                 frequency_decompose_type=None,
                 all_degradation_embedding_method=[],
                 degradation_dim=-1,
                 degradation_embedding_method=[],
                 debug_mode=False):

        super().__init__()
        
        self.debug_mode = debug_mode
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator,
                                    frequency_decompose_type=frequency_decompose_type,
                                    all_degradation_embedding_method=all_degradation_embedding_method,
                                    degradation_dim=degradation_dim,
                                    degradation_embedding_method=degradation_embedding_method,
                                    debug_mode=debug_mode)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator,
                                    frequency_decompose_type=frequency_decompose_type,
                                    all_degradation_embedding_method=all_degradation_embedding_method,
                                    degradation_dim=degradation_dim,
                                    degradation_embedding_method=degradation_embedding_method,
                                    debug_mode=debug_mode)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, inter=None, inter_kv=None, all_inter=None, mask=None):
        visual_freqs = []
        for blk in self.blocks:
            if self.use_checkpoint:
                assert False
                # x = checkpoint.checkpoint(blk, x)
            else:
                if self.debug_mode:
                    x, visual_freq = blk(x, inter, inter_kv, all_inter, mask)
                    visual_freqs.append(visual_freq)
                else:
                    x = blk(x, inter, inter_kv, all_inter, mask)
        return x, visual_freqs


class UformerDecoder(nn.Module):
    def __init__(self, opt, img_size=128, in_chans=3, out_chans=3,
                 depths=[2, 2, 8, 8, 2, 8, 8, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True,
                 **kwargs):
        super().__init__()
        
        self.opt = opt
        self.debug_mode = opt.debug_mode
        embed_dim = opt.embed_dim
        
        modulator = opt.learnable_modulator
        cross_modulator = False
        
        if opt.frequency_decompose_type == 'none':
            self.frequency_decompose_type = None
        else:
            self.frequency_decompose_type = opt.frequency_decompose_type
        
        all_degradation_embedding_method = []
        for type in opt.degradation_embedding_method:
            if 'all' in type:
                all_degradation_embedding_method.append(type)
        
        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.in_chans = in_chans

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers
        
        # Degradation Representation Embedding
        if 'residual' in opt.degradation_embedding_method:
            self.degradation_embed = [nn.Linear((2 ** (i + 1)) * embed_dim, (2 ** i) * embed_dim).cuda()
                              for i in range(self.num_enc_layers + 1)]

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=out_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.bottleneck_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            debug_mode=self.debug_mode)
        
        self.bottleneck_1 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            # modulator=modulator,cross_modulator=cross_modulator,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            degradation_dim=embed_dim*16,
                            degradation_embedding_method=opt.degradation_embedding_method,
                            debug_mode=self.debug_mode)

        # Decoder
        self.upsample_3 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            degradation_dim=embed_dim*8,
                            degradation_embedding_method=opt.degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.upsample_2 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            degradation_dim=embed_dim*4,
                            degradation_embedding_method=opt.degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.upsample_1 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            degradation_dim=embed_dim*2,
                            degradation_embedding_method=opt.degradation_embedding_method,
                            debug_mode=self.debug_mode)
        self.upsample_0 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            frequency_decompose_type=self.frequency_decompose_type,
                            all_degradation_embedding_method=all_degradation_embedding_method,
                            degradation_dim=embed_dim,
                            degradation_embedding_method=opt.degradation_embedding_method,
                            debug_mode=self.debug_mode)
        
        self.decoderlayer = [self.decoderlayer_0, self.decoderlayer_1, self.decoderlayer_2, self.decoderlayer_3]
        self.upsample = [self.upsample_0, self.upsample_1, self.upsample_2, self.upsample_3]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine == True:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, inter, mask=None):
        '''
        :param x: feature map: (B, C, H, W)
        :param inter: degradation representation: [(B, H*W, embed_dim), (B, H/2*W/2, embed_dim*2), (B, H/4*W/4, embed_dim*4), (B, H/8*W/8, embed_dim*8), (B, H/16*W/16, embed_dim*16)]
        '''
        assert mask == None
        # tmp
        inter = [None, None, None, None, inter, [None, None, None, None, None]]
        
        # Input Projection
        y = self.input_proj(x) # B H*W embed_dim
        y = self.pos_drop(y)
        
        visual_freqs = []
        
        # Encoder
        conv0, visual_freq_0 = self.encoderlayer_0(y, all_inter=inter[4], mask=mask)
        pool0 = self.dowsample_0(conv0) # B H/2*W/2 embed_dim*2
        conv1, visual_freq_1 = self.encoderlayer_1(pool0, all_inter=inter[4], mask=mask)
        pool1 = self.dowsample_1(conv1) # B H/4*W/4 embed_dim*4
        conv2, visual_freq_2 = self.encoderlayer_2(pool1, all_inter=inter[4], mask=mask)
        pool2 = self.dowsample_2(conv2) # B H/8*W/8 embed_dim*8
        conv3, visual_freq_3 = self.encoderlayer_3(pool2, all_inter=inter[4], mask=mask)
        pool3 = self.dowsample_3(conv3) # B H/16*W/16 embed_dim*16
        
        conv = [conv0, conv1, conv2, conv3]
        
        # Bottleneck
        conv4, visual_freq_4 = self.bottleneck_0(pool3, all_inter=inter[4], mask=mask)
        
        if 'residual' in self.opt.degradation_embedding_method:
            conv4 = self.degradation_embed[4](torch.cat([inter[4], conv4], -1))
        
        fea = conv4
        fea, visual_freq_5 = self.bottleneck_1(fea, inter[4], inter[5][4], all_inter=inter[4], mask=mask)
        
        visual_freqs.extend([visual_freq_0, visual_freq_1, visual_freq_2, visual_freq_3, visual_freq_4, visual_freq_5])

        #Decoder
        for i in reversed(range(self.num_dec_layers)):
            fea = self.upsample[i](fea)
            
            if 'residual' in self.opt.degradation_embedding_method:
                conv[i] = self.degradation_embed[i](torch.cat([inter[i], conv[i]], -1))
                
            fea = torch.cat([fea, conv[i]], -1)
            fea, visual_freq = self.decoderlayer[i](fea, inter[i], inter[5][i], all_inter=inter[4], mask=mask)
            visual_freqs.append(visual_freq)

        # Output Projection
        y = self.output_proj(fea)
        if self.debug_mode:
            return (x + y if self.in_chans ==3 else y), visual_freqs
        else:
            return x + y if self.in_chans ==3 else y


if __name__ == "__main__":
    import os
    import sys
    p = os.path.join('.')
    sys.path.append(os.path.abspath(p))
    from option import options as opt
    
    opt.degradation_embedding_method = ['all_DC']
    opt.frequency_decompose_type = 'none'
    
    model_restoration = UformerDecoder(opt)
    # print(model_restoration)
    
    print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))
    
    B = 4
    C = 3
    H = W = 128
    encoder_embed_dim = 28
    x = torch.zeros((B, C, H, W))
    inter = [torch.zeros((B, H*W, encoder_embed_dim)), torch.zeros((B, H//2*W//2, encoder_embed_dim*2)), torch.zeros((B, H//4*W//4, encoder_embed_dim*4)), torch.zeros((B, H//8*W//8, encoder_embed_dim*8)), torch.zeros((B, H//16*W//16, encoder_embed_dim*16))]
    
    win_size = 8
    inter_kv = []
    for i in range(5):
        tmp = torch.zeros(B * H//win_size//(2 ** i) * W//win_size//(2 ** i), 2 ** i, win_size ** 2, encoder_embed_dim)
        inter_kv.append([tmp, tmp])
    inter.append(inter_kv)
    
    restored = model_restoration(x, inter)
    
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("number of GFLOPs: %.2f G"%(model_restoration.flops() / 1e9))