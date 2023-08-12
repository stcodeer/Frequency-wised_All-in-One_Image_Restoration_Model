from tqdm import tqdm

from torch.utils.data import DataLoader

import numpy as np

from utils.dataset_utils import TrainDataset, checkout
from utils.visualization_utils import plot_image_grid, get_frequency_distribution, plot_curve, rgb2gray

from option import options as opt

if __name__ == '__main__':
    opt.de_type = ['denoising_15', 'denoising_25', 'denoising_50', 'deraining', 'dehazing', 'deblurring']
    opt.batch_size = 6
    
    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=0)
    print('loading %s data pairs in total.'%(str(trainset.len())))

    iterator = 0
    
    checkout('output_img')
    
    for ([clean_name, de_id], degrad_patch, degrad_patch_2, clean_patch, clean_patch_2) in tqdm(trainloader):
        iterator = iterator + 1
        if iterator == 5:
            break
        
        degrad_patch = degrad_patch.numpy()
        clean_patch = clean_patch.numpy()
        
        plot_image_grid(degrad_patch, dpi=1000, save_path='output_img/degrad_patch_%s.png'%str(iterator))
        plot_image_grid(clean_patch, dpi=1000, save_path='output_img/clean_patch_%s.png'%str(iterator))
        
        for idx in range(len(opt.de_type)):
            degrad = rgb2gray(degrad_patch[idx])
            degrad = get_frequency_distribution(degrad)
            
            clean = rgb2gray(clean_patch[idx])
            clean = get_frequency_distribution(clean)
        
            plot_curve((degrad, clean),
                    labels=('degradation', 'clean'), 
                    xlabel='Frequency Bands', 
                    ylabel='Gray Value', 
                    ylim=(0, 1),
                    save_path='output_img/frequency_distribution_center0_ratio_patch%s_degrad%s.png'%(iterator, idx))   