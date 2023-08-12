# based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from net.frequency_decompose import FrequencyDecompose

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,
                 decompose_type = 'none',
                 wised_batch = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        
        self.num_bands = None
        
        if not decompose_type == 'none':
            if decompose_type.split('_')[-1] == 'bands':
                self.num_bands = int(decompose_type.split('_')[0])
                self.decompose = FrequencyDecompose('frequency_decompose', 1./self.num_bands, dim_head, dim_head)
            
            elif decompose_type == 'DC':
                self.num_bands = 2
                self.decompose = FrequencyDecompose('frequency_decompose_dc', 1./self.num_bands, dim_head, dim_head)
                
            if wised_batch == None:
                self.lamb = nn.Parameter(torch.zeros(self.num_bands, 1, heads))
            else:
                self.lamb = nn.Parameter(torch.zeros(self.num_bands, wised_batch, heads))
        
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        
        # frequency decompose here
        if not self.num_bands == None:
            attn_bands = self.decompose(attn) # [num_bands, B, num_heads, N, N]
            
            # assert torch.all(torch.abs(attn - torch.sum(attn_bands, dim=0)) < 1e-5), 'oops'
            
            attn_bands = attn_bands * self.lamb[:, :, :, None, None]
            
            attn = attn + torch.sum(attn_bands, dim=0)
        
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,
                 decompose_type = 'none',
                 wised_batch = None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,
                                       decompose_type = decompose_type,
                                       wised_batch = wised_batch)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTEncoder(nn.Module):
    def __init__(self,
                 opt,
                 image_size = 128,
                 patch_size = 16, 
                 depth = 12, # [12, 24, 32]
                 heads = 12, # [12, 16, 16]
                 mlp_dim = 3072, # [3072, 4096, 5120]
                 channels = 3,
                 dropout = 0.1, 
                 emb_dropout = 0.1):
        super().__init__()
        
        out_channels = opt.out_channels
        
        dim = out_channels * patch_size * patch_size
        
        self.opt = opt
        
        self.depth = depth
        
        dim_head = dim // heads
        
        self.image_height, self.image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert self.image_height % patch_height == 0 and self.image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (self.image_height // patch_height) * (self.image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                       decompose_type = opt.frequency_decompose_type,
                                       wised_batch = opt.batch_size if opt.batch_wise_decompose else None)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // out_channels * opt.encoder_dim)
        )
        
        self.norm = nn.Sequential(
            nn.BatchNorm2d(opt.encoder_dim),
            nn.LeakyReLU(0.1, True),
        )
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(opt.encoder_dim, opt.encoder_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(opt.encoder_dim, opt.encoder_dim),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = self.mlp_head(x)
        
        inter = x.reshape(-1, self.opt.encoder_dim, self.image_height, self.image_width)
        
        # inter = self.norm(inter)
        
        fea = self.avg(inter).squeeze(-1).squeeze(-1)
        
        out = self.mlp(fea)

        return fea, out, inter