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


def plot_image_grid(images_np, nrow=8, factor=1, dpi=100, interpolation='lanczos', plot=False, title='default', save_path='default'):
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
    
    plt.figure(figsize=(len(images_np) + factor, 2 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
        
    if not title == 'default':
        plt.title(title)
        
    if not save_path == 'default': 
        plt.savefig(save_path, dpi=dpi)
        
    if plot:
        print("plotting...")
        plt.show()
        
    plt.close()
    
    return grid


def plot_loss_curve(path, num_epochs=None, ylim=((0, 4), (0, 0.05)), save_path=None):
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
    
    ax1.set_ylim(ylim[0][0], ylim[0][1])
    ax1.set_ylabel('Contrast Loss')
    ax1.plot(range(0, num_epochs), contrast_loss[0: num_epochs], color=c[0], label='Contrast Loss', linewidth=4)
    
    ax2 = ax1.twinx()
    
    ax2.set_ylim(ylim[1][0], ylim[1][1])
    ax2.set_ylabel('L1 Loss')
    ax2.plot(range(first_epochs, num_epochs), l1_loss[first_epochs: num_epochs], color=c[1], label='L1 Loss', linewidth=4)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        
    plt.grid()
    if save_path == None:
        save_path = os.path.join(path, 'loss_curve.png')
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    
    
def plot_curve(f, x_range=None, labels=None, xlabel=None, ylabel=None, ylim=(0, 40), figsize=(7, 6), scale='linear', save_path=None):
    c = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
        
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.yscale(scale)
    
    if x_range == None:
        x_range = (0, len(f[0]))
    ax.set_xlim(0, x_range[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    if not xlabel == None:
        ax.set_xlabel(xlabel)

    if not ylabel == None:
        ax.set_ylabel(ylabel)
    
    for idx in range(len(f)):
        if not labels == None:
            ax.plot(range(*x_range), f[idx], color=c[idx], label=labels[idx], linewidth=4)
        else:
            ax.plot(range(*x_range), f[idx], color=c[idx], linewidth=4)
    
    plt.legend(loc="lower right")
        
    plt.grid()
    if not save_path == None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def rgb2gray(rgb):
    if rgb.shape[2] == 1:
        return rgb[:,:,0]

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return np.clip(gray, 0, 255)


def get_frequency_distribution(img, size=0.2, norm=True):

    ftimage = np.fft.fft2(img)
    ftimage = abs(np.fft.fftshift(ftimage))

    h, w = ftimage.shape

    center = (int(w / 2), int(h / 2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    
    diag = center[0] # np.sqrt(center[0] ** 2 + center[1] ** 2)

    tot_value = [0.] * int(1 / size)
    
    for idx, sz in enumerate(np.linspace(size, 1, int(1 / size))):
        for y in range(h):
            for x in range(w):
                if sz == 1 and diag * (sz - size) <= dist_from_center[y][x] <= diag * sz:
                    tot_value[idx] = tot_value[idx] + ftimage[y][x]
                elif sz < 1 and diag * (sz - size) <= dist_from_center[y][x] < diag * sz:
                    tot_value[idx] = tot_value[idx] + ftimage[y][x]
    
    if norm: 
        tot_value = tot_value / np.sum(tot_value)
                    
    return tot_value
    
    
def plot_scatter(x, y, labels=None, xlabel=None, ylabel=None, title=None, set_lim=True, xlim=(0, 40), ylim=(0, 40), figsize=(7, 7), save_path=None):
    c = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
        
    fig, ax = plt.subplots(figsize=figsize)
    
    if set_lim:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
    
    if not xlabel == None:
        ax.set_xlabel(xlabel)

    if not ylabel == None:
        ax.set_ylabel(ylabel)
    
    for idx in range(len(labels)):
        if not labels == None:
            ax.scatter(x[idx], y[idx], s=15, color=c[idx], label=labels[idx])
        else:
            ax.scatter(x[idx], y[idx], s=15, color=c[idx])
            
    if not title == None:
        plt.title(title)
    
    plt.legend(loc="upper right")
        
    plt.grid()
    if not save_path == None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()