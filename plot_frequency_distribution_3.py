import subprocess
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestDataset, checkout
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model import AirNet

from option import options as opt

from utils.visualization_utils import get_frequency_distribution, plot_scatter, rgb2gray

def plot_by_task(net, task, epochs):
    print('starting testing %s...'%(task))
    
    testset = TestDataset(opt, task)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    low = []
    high = []
    iterator = 0
    
    with torch.no_grad():
        for ([img_name], degrad_patch, clean_patch) in tqdm(testloader):
            iterator = iterator + 1
            if iterator == 41:
                break
            
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            fea, inter = net.E(x_query=degrad_patch, x_key=degrad_patch)
            inter = inter[0].cpu().numpy()
            inter = np.mean(inter, 0)
            inter = get_frequency_distribution(inter, norm=False)
                
            low.append(inter[0])
            high.append(np.sum(inter[1:]))
            
    return low, high

opt.cuda = 2
opt.output_path = 'output/encoder_type_ViT_encoder_dim_64_batch_wise_decompose_True_frequency_decompose_type_5_bands_crop_test_imgs_size_128_de_type_denoising_15_denoising_25_denoising_50_deraining_test_de_type_denoising_bsd68_15_denoising_bsd68_25_denoising_bsd68_50_deraining/'
opt.ckpt_path = opt.output_path + 'ckpt/'
opt.de_type = ['denoising_15', 'denoising_25', 'denoising_50', 'deraining']
opt.test_de_type = ['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining']
opt.batch_size = len(opt.de_type)
opt.encoder_type = 'ViT'
opt.encoder_dim = 64
opt.batch_wise_decompose = True
opt.frequency_decompose_type = '5_bands'
opt.crop_test_imgs_size = 128

torch.cuda.set_device(opt.cuda)
np.random.seed(0)
torch.manual_seed(0)

# Make network
net = AirNet(opt).cuda()
net.eval()
net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))

x = []
y = []

for task in opt.test_de_type:
    low, high = plot_by_task(net, task=task, epochs=opt.epochs)
    x.append(low)
    y.append(high)
    
plot_scatter(x, y,
        labels=opt.de_type,
        xlabel='LC Gray Value', 
        ylabel='HC Gray Value', 
        # xlim=(25000, 75000),
        # ylim=(400000, 450000),
        xlim=(np.min(x), np.max(x)),
        ylim=(np.min(y), np.max(y)),
        save_path=opt.output_path+'frequency_distribution_latent_scatter_center0_1.png')   
