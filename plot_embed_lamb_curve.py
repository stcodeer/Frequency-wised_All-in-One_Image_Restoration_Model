import subprocess
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestDataset, checkout
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.visualization_utils import plot_image_grid

from net.model import AirNet

from option import options as opt

from utils.visualization_utils import get_frequency_distribution, plot_scatter, rgb2gray

def plot_by_task(net, task, epochs):
    print('starting testing %s...'%(task))
    
    testset = TestDataset(opt, task)
    testloader = DataLoader(testset, batch_size=10, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    
    with torch.no_grad():
        for ([img_name], degrad_patch, clean_patch) in tqdm(testloader):
            
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            fea, inter = net.E(degrad_patch, degrad_patch)
            
            restored, visual_freqs = net.R(x_query=degrad_patch, inter=inter)
            
            return visual_freqs

opt.cuda = 0
opt.output_path = 'output/h__cuda_0_num_frequency_bands_l1_1_degradation_embedding_method_all_DC_de_type_denoising_75_test_de_type_denoising_bsd68_75_epochs_encoder_50_epochs_350/'
opt.ckpt_path = opt.output_path + 'ckpt/'
opt.de_type = ['denoising_75']
opt.test_de_type = ['denoising_bsd68_75']
opt.degradation_embedding_method = ['all_DC']
opt.batch_size = 1
opt.encoder_dim = 256
opt.embed_dim = 56
opt.epochs = 350

opt.debug_mode = True

torch.cuda.set_device(opt.cuda)
np.random.seed(0)
torch.manual_seed(0)

# Make network
net = AirNet(opt).cuda()
net.eval()
net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))

num_bands = 1

bands = []
for i in range(num_bands):
    bands.append([])

for task in opt.test_de_type:
    visual_freqs = plot_by_task(net, task=task, epochs=opt.epochs)
    
    print(len(visual_freqs))
    
    for i, v1 in enumerate(visual_freqs):
        for j, v in enumerate(v1):
            print(i, j, v[2].shape)
            for t in range(num_bands):
                bands[t].append(np.mean(v[2].tolist()))
                # bands[t].extend(np.array(v[2].cpu()).flatten().tolist())
        
    for i in range(num_bands):
        print('%.2f'%(np.mean(bands[i])*100))