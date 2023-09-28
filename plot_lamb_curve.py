from utils.visualization_utils import plot_curve
import os
import re
import numpy as np

from torchvision import datasets, transforms
import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader
from utils.dataset_utils import TestDataset, checkout
from utils.visualization_utils import plot_scatter
from option import options as opt

from net.model import AirNet

if __name__ == '__main__':
    opt.output_path = 'output/h__cuda_0_frequency_decompose_type_DC_degradation_embedding_method_None_embed_dim_14_de_type_denoising_50_test_de_type_denoising_bsd68_50_epochs_encoder_0_epochs_300/'
    opt.ckpt_path = opt.output_path + 'ckpt/'
    opt.frequency_decompose_type = 'DC'
    opt.degradation_embedding_method = ['None']
    opt.batch_size = 1
    opt.encoder_dim = 256
    opt.epochs = 300
    opt.embed_dim = 14
    torch.cuda.set_device(0)
    
    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))

    band_0 = []
    band_1 = []
    
    for param, weight in net.R.R.named_parameters():
        if 'lamb' in param and ('bottleneck_1' in param):
            print(param)
            # band_0.append(np.mean(weight[0, 0, :].tolist()))
            # band_1.append(np.mean(weight[0, 0, :].tolist()))
            # band_0.extend(weight[0, 0, :].tolist())
            band_1.extend(weight[0, 0, :].tolist())
            
    # print(
    #         '(%s) LFC average weight: %0.4f HFC average weight: %0.4f\n' % (
    #             opt.de_type, np.mean(band_0), np.mean(band_1),
    #         ), '\r', end='')
    print(np.mean(band_1))        
    
    # lamb_k = np.mean(lamb_k, axis=3)
    # lamb_k = np.mean(lamb_k, axis=0)
    # lamb_k = np.transpose(lamb_k, (1, 0)) # [wised_batch, num_bands]
        
    # plot_curve(lamb_k, 
    #         ylim=(-1, 1),
    #         labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
    #         xlabel='Frequency Bands', 
    #         ylabel='Lamb', 
    #         save_path=os.path.join(opt.output_path, 'lamb_k_curve.png'))     

