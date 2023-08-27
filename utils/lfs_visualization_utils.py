# based on https://github.com/Daner-Wang/VTC-LFC
# from asyncio import current_task
# from mailbox import mbox
# from tkinter.messagebox import NO
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial import distance
import utils
from timm.models.vision_transformer import PatchEmbed, Attention
from torch import optim as optim
import itertools
import math
import json
from pathlib import Path
import copy
from torch.utils.data.sampler import Sampler
import tqdm

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
# ======================================================================================================================
def count_channels(model: nn.Module):
    total_channels = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            total_channels += m.weight.size(0)
    return int(total_channels)

def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)

def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def filtering(images, L=0.065, padding=0, reverse=False):
    if padding > 0:
        images = F.pad(images, pad=(padding, padding, padding, padding), mode='constant', value=0)

    _, _, H, W = images.shape
    K = min(H, W)
    d0 = (K * L / 2) ** 2
    m0 = (K - 1) / 2.
    x_coord = torch.arange(K).to(images.device)
    x_grid = x_coord.repeat(K).view(K, K)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    kernel = torch.exp(-torch.sum((xy_grid - m0)**2, dim=-1) / (2*d0))

    if IS_HIGH_VERSION:
        fftmaps = torch.fft.fft2(images)
        fftmaps = torch.stack([fftmaps.real, fftmaps.imag], -1)
        pha = torch.atan2(fftmaps[..., 1], fftmaps[..., 0])
        amp = torch.sqrt(fftmaps[..., 0]**2 + fftmaps[..., 1]**2)
    else:
        fftmaps = torch.rfft(images, signal_ndim=2, onesided=False)
        amp, pha = extract_ampl_phase(fftmaps)
    if reverse:
        mask = kernel
    else:
        mask = ifftshift(kernel)
    low_amp = amp.mul(mask)
    a1 = torch.cos(pha) * low_amp
    a2 = torch.sin(pha) * low_amp
    fft_src_ = torch.cat([a1.unsqueeze(-1),a2.unsqueeze(-1)],dim=-1)
    if IS_HIGH_VERSION:
        fft_src_ = torch.complex(fft_src_[..., 0], fft_src_[..., 1])
        outputs = torch.fft.ifft2(fft_src_)
    else:
        outputs = torch.irfft(fft_src_, signal_ndim=2, onesided=False)

    if padding > 0:
        outputs = outputs[:, :, padding:-padding, padding:-padding]

    return outputs

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.as_tensor(range(len(self.indices))))

    def __len__(self):
        return len(self.indices)

class Pruning():
    def __init__(self, model, type='low'):
        super(Pruning, self).__init__()
        self.org_model = model
        self.score = None
        self.w_mgrad = None
        self.cor_mtx = []
        self.i_mask, self.o_mask = [], []
        self.head_cfg, self.head_mask = None, None
        self.actural_cpr = []
        self.data_loader = None

        self.tau, self.alpha = 1.0, 0.1
        self.cutoff = 0.2
        self.type = type
    
    def get_weight_mgrad(self, data_loader, device):
        teacher_model = copy.deepcopy(self.org_model)
        teacher_model.eval()

        criterion = torch.nn.L1Loss()
        optimizer = optim.SGD(self.org_model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
        self.org_model.train()
        grad_batch = {}
        dill_loss = 0
        
        for (batch_id, ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2)) in tqdm.tqdm(enumerate(data_loader)):
            images = degrad_patch_1.to(device, non_blocking=True)
            target = clean_patch_1.to(device, non_blocking=True)

            with torch.no_grad():
                _, teacher_mid = teacher_model.E(images, images) # degradation representation here.

            self.org_model.train()
            optimizer.zero_grad()
            if self.type == 'low':
                images = filtering(images, L=self.cutoff, padding=0, reverse=False).real
            elif self.type == 'high':
                images = images - filtering(images, L=self.cutoff, padding=0, reverse=False).real
            else:
                assert False, 'invalid type.'
                
            
            output, _, _ = self.org_model(images, images) # restored images here.
            _, _, _, student_mid = self.org_model.E(images, images) # degradation representation here.
            loss = criterion(output, target)

            if student_mid.dim() > 1:
                T = self.tau
                distillation_loss = F.kl_div(
                    F.log_softmax(student_mid / T, dim=1),
                    F.log_softmax(teacher_mid / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / student_mid.numel()
                dill_loss += distillation_loss.item()
                loss = loss * self.alpha + distillation_loss * (1 - self.alpha)

            loss.backward()

            if batch_id % 10 == 0:
                print(f'batch-{batch_id}: Counting weight grads... | distillation_loss: {dill_loss/(batch_id+1)}')
            for k, param in self.org_model.named_parameters():
                if param.requires_grad and not param.grad == None:
                    if batch_id == 0:
                        grad_batch[k] = param.grad
                    else:
                        grad_batch[k] = grad_batch[k] + param.grad
                  
        for k, v in grad_batch.items():
            grad_batch[k] = v / (batch_id + 1)

        return grad_batch

    def taylor_score(self, grad, param, name, block_id=None):
        w = (param.mul(grad[name]).view(param.shape[0], -1)**2).sum(-1)
        if 'qkv' in name:
            v_dim = 768
            w_qk, w_v = w[:-v_dim].reshape(2, -1), w[-v_dim:]
            w = torch.cat((w_qk[0]+w_qk[1], w_qk[0]+w_qk[1], w_v), dim=0)
        return w

    def criterion_l_taylor(self, data_loader, device):
        score = []
        block_id = 0
        ori_graph_paras = self.org_model.state_dict()

        grad_dict = self.get_weight_mgrad(data_loader, device)
        self.w_mgrad = grad_dict
        print('Gradient achieved!!!')

        for k, v in ori_graph_paras.items():
            if k in grad_dict:
                w = self.taylor_score(grad=grad_dict, param=v, name=k)
            print(k, torch.mean(w).item())
            score.append(w)
            block_id += 1
        return score

    def get_score(self,  criterion: str, dataset=None, device=None, batch_size=None):
        if criterion == 'lfs':
            sampler = torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(dataset)), batch_size * 100))
            data_loader = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=batch_size, num_workers=10,
                pin_memory=True, drop_last=False)
            self.score = self.criterion_l_taylor(data_loader, device)
        else:
            raise ValueError('Unsupported criterion!')

    def get_mask_and_newcfg(self, criterion='lfs', dataset=None, device=None, batch_size=None):
        
        self.get_score(criterion, dataset, device, batch_size)
        
        self.i_mask, self.o_mask = [], []
        self.actural_cpr = []
        self.head_cfg = []
        print('Scores achieved!!!')

        score = None
        for s in self.score:
            score = s if score is None else torch.cat((score, s), dim=0)
        sorted_score, _ = torch.sort(score)
        print(sorted_score)
        
        ori_graph_paras = self.org_model.state_dict()