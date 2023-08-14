from tqdm import tqdm

from torch.utils.data import DataLoader

import numpy as np

from utils.dataset_utils import TrainDataset, checkout
from utils.visualization_utils import get_frequency_distribution, plot_scatter, rgb2gray

from option import options as opt

if __name__ == '__main__':
    opt.de_type = ['denoising_15', 'denoising_25', 'denoising_50', 'deraining', 'dehazing', 'deblurring']
    opt.batch_size = 6
    
    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=0)
    print('loading %s data pairs in total.'%(str(trainset.len())))

    checkout('output_img')
    
    x = []
    y = []
    for idx in range(len(opt.de_type)):
        x.append([])
        y.append([])
    
    iterator = 0
    
    for ([clean_name, de_id], degrad_patch, degrad_patch_2, clean_patch, clean_patch_2) in tqdm(trainloader):
        iterator = iterator + 1
        if iterator == 41:
            break
        
        degrad_patch = degrad_patch.numpy()
        clean_patch = clean_patch.numpy()
        
        for idx in range(len(opt.de_type)):
            degrad = rgb2gray(degrad_patch[idx])
            degrad = get_frequency_distribution(degrad)
            
            clean = rgb2gray(clean_patch[idx])
            clean = get_frequency_distribution(clean)
            
            x[idx].append(clean[0] / degrad[0])
            y[idx].append(np.sum(clean[1:]) / np.sum(degrad[1:]))
    
    plot_scatter(x, y,
            labels=opt.de_type,
            xlabel='LFC_clean/LFC_degrad', 
            ylabel='HFC_clean/HFC_degrad', 
            xlim=(0, 2),
            ylim=(0, 4),
            save_path='output_img/frequency_distribution_scatter_center0.png')   