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
            
            restored, visual_freqs = net.R(x_query=degrad_patch, inter=[0, 0, 0, 0, 0, [0, 0, 0, 0, 0]])
            
            return visual_freqs

opt.cuda = 0
opt.output_path = 'output/h__cuda_0_degradation_embedding_method_None_de_type_denoising_15_test_de_type_denoising_bsd68_15_epochs_encoder_0_epochs_300/'
opt.ckpt_path = opt.output_path + 'ckpt/'
opt.de_type = ['denoising_15']
opt.test_de_type = ['denoising_bsd68_15']
# opt.frequency_decompose_type = 'DC'
opt.degradation_embedding_method = ['None']
opt.batch_size = 1
opt.encoder_dim = 256
opt.embed_dim = 56
opt.epochs = 300

opt.debug_mode = True

torch.cuda.set_device(opt.cuda)
np.random.seed(0)
torch.manual_seed(0)

# Make network
net = AirNet(opt).cuda()
net.eval()
net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))

x = []
y = []

checkout('output_img')
checkout('output_img/msa_denoising_15/')

for task in opt.test_de_type:
    visual_freqs = plot_by_task(net, task=task, epochs=opt.epochs)
    
    print(len(visual_freqs))
    
    for i, v1 in enumerate(visual_freqs):
        for j, v in enumerate(v1):
            print(i, j, v[0].shape, v[1].shape)
            before_msa = v[0].cpu().numpy()
            after_msa = v[1].cpu().numpy()
            before_msa = np.expand_dims(before_msa, 0)
            after_msa = np.expand_dims(after_msa, 0)
            plot_image_grid([np.log10(before_msa), np.log10(after_msa)], padding=0, factor=0, title='layer_%d block_%d'%(i, j), save_path='output_img/msa_denoising_15/layer_%d_block_%d'%(i, j))