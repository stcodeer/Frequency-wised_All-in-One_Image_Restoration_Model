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
