import torch
from torch import nn
import math

class FrequencyDecompose(nn.Module):
    def __init__(self, type, size, h, w, inverse=True):
        super().__init__()
        self.type = type
        self.size = size
        self.h = h
        self.w = w
        self.inverse = inverse
        
        assert size > 0 and size < 1, 'invalid frequency band width()'%(size)
        
        if self.type == 'frequency_decompose':
            self.Y = torch.arange(h).unsqueeze(1).cuda()
            self.X = torch.arange(w).unsqueeze(0).cuda()
            
            self.num_bands = math.floor(1. / self.size + 0.1)
            
            center = torch.tensor([int(w / 2), int(h / 2)]).cuda()
            
            self.dist_from_center = torch.sqrt((self.X - center[0]) ** 2 + (self.Y - center[1]) ** 2) # [N, N]
            
            self.max_radius = torch.sqrt(center[0] ** 2 + center[1] ** 2) # center[0]
        
    def frequency_decompose(self, x):
        # B, nH, N, N
        B, C = x.shape[0], x.shape[1]

        fre_x = torch.fft.fftshift(torch.fft.fft2(x)) # [B, C, N, N]
        
        last_mask = torch.zeros((self.h, self.w), dtype=bool).cuda()
        
        decomposed_x = []
        
        for sz in torch.linspace(self.size, 1, self.num_bands):
            radius = self.max_radius * sz
            
            if sz == 1.0:
                mask = self.dist_from_center <= radius
            else:
                mask = self.dist_from_center < radius
                
            mask_now = mask ^ last_mask # [N, N]
            
            last_mask = mask
            
            mask_now = mask_now.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
            
            decomposed_fre_x = mask_now.repeat(B, C, 1, 1) * fre_x # [B, C, N, N]
            
            decomposed_fre_x = torch.fft.ifftshift(decomposed_fre_x)
            
            if self.inverse:
                decomposed_fre_x = torch.fft.ifft2(decomposed_fre_x).real
            else:
                decomposed_fre_x = torch.abs(decomposed_fre_x)
            
            decomposed_x.append(decomposed_fre_x.unsqueeze(0))
        
        return torch.concat(decomposed_x, dim=0)

    def frequency_decompose_dc(self, x):
        # B, nH, N, N
        N = x.shape[2]
        x_d = torch.mean(torch.mean(x, -1, keepdim=True), -2, keepdim=True)
        x_d = x_d.repeat(1, 1, N, N)
        x_h = x - x_d
        
        x_d, x_h = x_d.unsqueeze(0), x_h.unsqueeze(0)
        
        return torch.concat([x_d, x_h], dim=0)
    
    def forward(self, x):
        if self.type == 'frequency_decompose':
            return self.frequency_decompose(x)
        else:
            return self.frequency_decompose_dc(x)