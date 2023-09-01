from torch import nn

from .decoder_DGRN import DGRN as ResNetDecoder
from .decoder_Uformer import UformerDecoder

from .encoder_ResNet import ResNetEncoder
from .encoder_ViT import ViTEncoder
from .encoder_Uformer import UformerEncoder

from .utils.moco import MoCo


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        
        decoder = globals()[opt.decoder_type + 'Decoder']
        
        self.R = decoder(opt)

    def forward(self, x_query, inter):
        restored = self.R(x_query, inter)

        return restored


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        encoder = globals()[opt.encoder_type + 'Encoder']
        
        # Encoder
        # self.E = MoCo(opt=opt, base_encoder=encoder, dim=opt.encoder_dim, K=opt.batch_size * opt.encoder_dim)
        self.E = MoCo(opt=opt, base_encoder=encoder, dim=opt.encoder_dim, K=opt.batch_size * 3)

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
