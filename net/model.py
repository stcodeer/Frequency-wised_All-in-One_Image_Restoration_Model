from torch import nn

from net.DGRN import DGRN as Decoder
from net.encoder_ResNet import ResNetEncoder
from net.encoder_ViT import ViTEncoder
from net.moco import MoCo


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        encoder = globals()[opt.encoder_type + 'Encoder']
        
        if opt.encoder_type == 'ResNet':
            dim = 256
        elif opt.encoder_type == 'ViT':
            dim = 768
        
        # Encoder
        self.E = MoCo(opt=opt, base_encoder=encoder, dim=dim, K=opt.batch_size * dim)

    def forward(self, x_query, x_key):
        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels, inter = self.E(x_query, x_key)

            return fea, logits, labels, inter
        else:
            # degradation-aware represenetion learning
            fea, inter = self.E(x_query, x_query)
            return fea, inter


class AirNet(nn.Module):
    def __init__(self, opt):
        super(AirNet, self).__init__()

        # Restorer
        self.R = Decoder(opt)

        # Encoder
        self.E = Encoder(opt)

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)

            return restored
