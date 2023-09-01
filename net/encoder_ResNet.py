from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResNetEncoder(nn.Module):
    def __init__(self, opt):
        super(ResNetEncoder, self).__init__()
        
        self.dim = opt.encoder_dim

        self.E_pre = ResBlock(in_feat=3, out_feat=self.dim//4, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=self.dim//4, out_feat=self.dim//2, stride=2),
            ResBlock(in_feat=self.dim//2, out_feat=self.dim, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


if __name__ == "__main__":
    import os
    import sys
    
    p = os.path.join('.')
    sys.path.append(os.path.abspath(p))
    from option import options as opt
    import torch
    
    model_restoration = ResNetEncoder(opt)
    # print(model_restoration)
    
    print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))
    
    x = torch.zeros((4, 3, 128, 128))
    fea, out, inter = model_restoration(x)
    
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("number of GFLOPs: %.2f G"%(model_restoration.flops() / 1e9))
