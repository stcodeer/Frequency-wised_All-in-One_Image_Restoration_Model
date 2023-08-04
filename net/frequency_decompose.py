import torch

def frequency_decompose(self, x, size):
    # B, nH, N, N
    B, C, N = x.shape[0], x.shape[1], x.shape[2]
    
    assert size > 0 and size < 1, 'invalid frequency band width()'%(size)

    fre_x = torch.fft.fftshift(torch.fft.fft2(x)) # [B, C, N, N]

    h, w = N, N

    center = (int(w / 2), int(h / 2))
    
    Y = torch.arange(h).unsqueeze(1)
    X = torch.arange(w).unsqueeze(0)

    dist_from_center = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) # [N, N]
    
    num_bands = torch.floor(1. / size + 0.1)
    
    decomposed_x = torch.zeros((num_bands, B, C, h, w))
    
    max_radius = torch.sqrt(center[0] ** 2 + center[1] ** 2) # center[0]
    
    last_mask = torch.zeros((h, w), dtype=bool)
    
    for idx, sz in enumerate(torch.linspace(size, 1, num_bands)):
        radius = max_radius * sz
        
        if sz == 1.0:
            mask = dist_from_center <= radius
        else:
            mask = dist_from_center < radius
            
        mask = mask ^ last_mask # [N, N]
        
        last_mask = mask
        
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
        
        decomposed_fre_x = mask.repeat(B, C, 1, 1) * fre_x # [B, C, N, N]
        
        decomposed_x[idx] = torch.fft.ifft2(torch.fft.ifftshift(decomposed_fre_x))
    
    return decomposed_x

def frequency_decompose_dc(self, x, size):
    # B, nH, N, N
    N = x.shape[2]
    x_d = torch.mean(torch.mean(x, -1, keepdim=True), -2, keepdim=True)
    x_d = x_d.repeat(1, 1, N, N)
    x_h = x - x_d
    
    x_d, x_h = x_d.unsqueeze(0), x_h.unsqueeze(0)
    
    return torch.concat([x_d, x_h], dim=0)