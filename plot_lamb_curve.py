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
    opt.output_path = 'output/2layer_lowhigh_patchwise_num_frequency_bands_l1_1_frequency_decompose_type_5_bands_degradation_embedding_method_None_de_type_denoising_75_test_de_type_denoising_bsd68_75_epochs_encoder_0_epochs_300/'
    opt.ckpt_path = opt.output_path + 'ckpt/'
    opt.frequency_decompose_type = '5_bands'
    opt.degradation_embedding_method = ['None']
    opt.batch_size = 1
    opt.encoder_dim = 256
    opt.epochs = 300
    opt.embed_dim = 56
    torch.cuda.set_device(0)
    
    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))

    num_bands = 5

    bands = []
    for i in range(num_bands):
        bands.append([])
    
    for param, weight in net.R.R.named_parameters():
        if 'lamb' in param:
            print(param)
            for i in range(num_bands):
                bands[i].append(np.mean(weight[i, :, :].tolist()))
                # bands[i].extend(weight[i, :, :].tolist())
        
    for i in range(num_bands):
        print('%.2f'%(np.mean(bands[i])*100))
    
    # lamb_k = np.mean(lamb_k, axis=3)
    # lamb_k = np.mean(lamb_k, axis=0)
    # lamb_k = np.transpose(lamb_k, (1, 0)) # [wised_batch, num_bands]
        
    # plot_curve(lamb_k, 
    #         ylim=(-1, 1),
    #         labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
    #         xlabel='Frequency Bands', 
    #         ylabel='Lamb', 
    #         save_path=os.path.join(opt.output_path, 'lamb_k_curve.png'))     

