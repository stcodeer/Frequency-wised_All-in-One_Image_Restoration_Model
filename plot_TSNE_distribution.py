from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os

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

def plot_by_task(net, task, epochs, samples):
    print('starting testing %s...'%(task))
    
    testset = TestDataset(opt, task)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    data = []
    
    iterator = 0
    
    with torch.no_grad():
        for ([img_name], degrad_patch, clean_patch) in tqdm(testloader):
            iterator = iterator + 1
            if iterator == samples + 1:
                break
            
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            embedding, _ = net.E(x_query=degrad_patch, x_key=degrad_patch)
            embedding = embedding[0].cpu().numpy()
            data.append(np.squeeze(embedding))
    
    data = np.array(data)
    print(data.shape)
    return data

if __name__ == '__main__':
    opt.cuda = 1
    opt.output_path = 'output/degradation_embedding_method_self_modulator_de_type_4tasks/'
    opt.ckpt_path = opt.output_path + 'ckpt/'
    opt.de_type = ['denoising_15', 'denoising_25', 'denoising_50', 'deraining']
    opt.test_de_type = ['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining']
    opt.batch_size = len(opt.de_type)
    opt.crop_test_imgs_size = 128
    opt.encoder_type = 'Uformer'
    opt.decoder_type = 'Uformer'
    opt.encoder_dim = 256
    opt.degradation_embedding_method = ['self_modulator']
    
    torch.cuda.set_device(opt.cuda)
    
    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))
    
    num_samples = 400
    embed_dim = 100

    data = []
    target = []

    for i, task in enumerate(opt.test_de_type):
        data_by_task = plot_by_task(net, task=task, epochs=opt.epochs, samples=num_samples)
        data.extend(data_by_task)
        target.extend([i] * len(data_by_task))
    

    checkout('output_img')
    

    x_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    
    x = []
    y = []
    for i in range(opt.batch_size):
        x.append([])
        y.append([])
    for i in range(len(target)):
        x[target[i]].append(x_tsne[i][0])
        y[target[i]].append(x_tsne[i][1])
        
    plot_scatter(x, y, labels=opt.de_type, title='T-SNE', set_lim=False, save_path='output_img/TSNE_%s.png'%opt.encoder_type)