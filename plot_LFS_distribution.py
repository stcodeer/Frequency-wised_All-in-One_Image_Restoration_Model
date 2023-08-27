import subprocess
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, checkout
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.lfs_visualization_utils import Pruning

from net.model import AirNet
from option import options as opt

from utils.visualization_utils import get_frequency_distribution, plot_scatter, rgb2gray

opt.cuda = 2
opt.output_path = 'output/encoder_type_ViT_encoder_dim_64_batch_wise_decompose_True_frequency_decompose_type_5_bands_crop_test_imgs_size_128_4tasks/'
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

lfs = Pruning(model=net, type='low')
lfs.get_mask_and_newcfg(dataset=TrainDataset(opt), device=torch.device(opt.cuda), batch_size=opt.batch_size)

# x = []
# y = []

# plot_scatter(x, y,
#         labels=opt.de_type,
#         xlabel='LC Gray Value', 
#         ylabel='HC Gray Value', 
#         # xlim=(25000, 75000),
#         # ylim=(400000, 450000),
#         xlim=(np.min(x), np.max(x)),
#         ylim=(np.min(y), np.max(y)),
#         save_path=opt.output_path+'frequency_distribution_latent_channel_center0_1.png')