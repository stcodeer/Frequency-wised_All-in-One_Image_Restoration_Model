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
opt.output_path = 'output/degradation_embedding_method_None_de_type_denoising_75_test_de_type_denoising_bsd68_75_epochs_encoder_0_epochs_300/'
opt.ckpt_path = opt.output_path + 'ckpt/'
opt.de_type = ['denoising_75']
opt.test_de_type = ['denoising_bsd68_75']
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

# checkout('output_img')
# checkout('output_img/msa_denoising_75/')

log_file = open('after_MSA_75.log', 'w')

for task in opt.test_de_type:
    visual_freqs = plot_by_task(net, task=task, epochs=opt.epochs)
    
    print(len(visual_freqs))
    
    for i, v1 in enumerate(visual_freqs):
        for j, v in enumerate(v1):
            print(i, j, v[0].shape, v[1].shape)
            v[0], v[1] = v[1], v[0]
            # before_msa = v[0].cpu().numpy()
            # after_msa = v[1].cpu().numpy()
            # before_msa = np.expand_dims(before_msa, 0)
            # after_msa = np.expand_dims(after_msa, 0)
            # plot_image_grid([np.log10(before_msa), np.log10(after_msa)], padding=0, factor=0, title='layer_%d block_%d'%(i, j), save_path='output_img/msa_denoising_75/layer_%d_block_%d'%(i, j))
            
            w = v[0].shape[0]
            h = v[0].shape[1]
            Y = torch.arange(h).unsqueeze(1).cuda()
            X = torch.arange(w).unsqueeze(0).cuda()
            
            center = torch.tensor([int(w / 2), int(h / 2)]).cuda()
            
            dist_from_center = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) # [N, N]
            
            max_radius = torch.sqrt(center[0] ** 2 + center[1] ** 2) # center[0]
            
            last_mask = torch.zeros((h, w), dtype=bool).cuda()
        
            decomposed_x = []
            
            for sz in torch.linspace(0.2, 1, 5):
                radius = max_radius * sz
                
                if sz == 1.0:
                    mask = dist_from_center <= radius
                else:
                    mask = dist_from_center < radius
                    
                mask_now = mask ^ last_mask # [N, N]
                
                last_mask = mask
                
                decomposed_x.append(torch.sum(mask_now * v[0]))
                
            print('sum ', torch.sum(v[0]), torch.sum(torch.tensor(decomposed_x)))
            log_file.write('%d %d %.6f %.6f %.6f %.6f %.6f\n'%(i, j, decomposed_x[0], decomposed_x[1], decomposed_x[2], decomposed_x[3], decomposed_x[4]))
            