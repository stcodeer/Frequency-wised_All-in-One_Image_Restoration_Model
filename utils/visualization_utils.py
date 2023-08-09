import torch
import torch.nn as nn
import torchvision

import numpy as np
from PIL import Image
import PIL
import numpy as np
import imageio
import re
import os

import matplotlib.pyplot as plt


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos', plot=False, title='default', save_path='default'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
        
    if not title == 'default':
        plt.title(title)
        
    if not save_path == 'default': 
        plt.savefig(save_path)
        
    if plot:
        print("plotting...")
        plt.show()
        
    plt.close()
    
    return grid


def plot_loss_curve(path, num_epochs=None, ylim=[4, 0.05], save_path=None):
    file = open(os.path.join(path, 'train.log'), 'r')
    Lines = file.readlines()
    file.close()
    
    contrast_loss = []
    l1_loss = []
    
    first_epochs = -1
    
    for idx, line in enumerate(Lines):
        strings = re.split('[:\s]', line.strip())
        if len(strings) < 9:
            l1_loss.append(0.)
            contrast_loss.append(float(strings[6]))
            
        else:
            if first_epochs == -1:
                first_epochs = idx
            l1_loss.append(float(strings[6]))
            contrast_loss.append(float(strings[8]))
            
    if num_epochs == None:
        num_epochs = len(contrast_loss)
    
    c = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
        
    fig, ax1 = plt.subplots(figsize=(20, 6))
    
    ax1.set_xlim(0, num_epochs)
    ax1.set_xlabel('Epochs')
    
    ax1.set_ylim(0, ylim[0])
    ax1.set_ylabel('Contrast Loss')
    ax1.plot(range(0, num_epochs), contrast_loss[0: num_epochs], color=c[0], label='Contrast Loss', linewidth=4)
    
    ax2 = ax1.twinx()
    
    ax2.set_ylim(0, ylim[1])
    ax2.set_ylabel('L1 Loss')
    ax2.plot(range(first_epochs, num_epochs), l1_loss[first_epochs: num_epochs], color=c[1], label='L1 Loss', linewidth=4)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        
    plt.grid()
    if save_path == None:
        save_path = os.path.join(path, 'loss_curve.png')
    plt.savefig(save_path)
    # plt.show()
    plt.close()