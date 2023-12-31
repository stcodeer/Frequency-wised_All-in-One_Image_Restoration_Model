import subprocess
import os
from tqdm import tqdm

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from utils.dataset_utils import TrainDataset, checkout
from utils.visualization_utils import plot_image_grid, plot_loss_curve
from net.model import AirNet
from net.utils.frequency_decompose import FrequencyDecompose
from test import test_by_task

from option import options as opt

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    # random.seed(1234)
    # np.random.seed(1234)
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    checkout(opt.output_path)
    checkout(opt.ckpt_path)

    train_log_file = open(opt.output_path + 'train.log', "w")
    opt_log_file = open(opt.output_path + 'options.log', "w")
    result_log_file = open(os.path.join(opt.output_path, 'results.log'), "w")
    # result_best_log_file = open(os.path.join(opt.output_path, 'results_best.log'), "w")
    # if not opt.frequency_decompose_type == 'none':
    #     lamb_q_log_file = open(opt.output_path + 'lamb_q.log', "w")
    #     lamb_k_log_file = open(opt.output_path + 'lamb_k.log', "w")
    
    opt_log_file.write(f"|{'=' * 151}|")
    opt_log_file.write("\n")
    for key, value in opt.__dict__.items():
        opt_log_file.write(f"|{str(key):>50s}|{str(value):<100s}|")
        opt_log_file.write("\n")
    opt_log_file.write(f"|{'=' * 151}|")
    opt_log_file.write("\n")
    opt_log_file.close()
    
    # print('start loading data...')
    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    print('loading %s data pairs in total.'%(str(trainset.len())))
    
    # print('start init network...')
    # Network Construction
    net = AirNet(opt).cuda()
    net.train()
    startpoint = 0
    if startpoint > 0:
        net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_' + str(startpoint) + '.pth', map_location=torch.device(opt.cuda)))

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()
        
    num_losses = opt.L

    if not opt.num_frequency_bands_l1 == -1:
        decompose = FrequencyDecompose('frequency_decompose', 1./opt.num_frequency_bands_l1, opt.patch_size, opt.patch_size, inverse=False)
    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        if epoch >= startpoint:
            for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):

                degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
                clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

                optimizer.zero_grad()

                if epoch < opt.epochs_encoder:
                    _, output, target, _ = net.E(x_query=degrad_patch_1, x_key=degrad_patch_2)
                    contrast_loss = sum([CE(output[i], target[i]) for i in range(num_losses)]) / num_losses
                    loss = contrast_loss
                else:
                    restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                    contrast_loss = sum([CE(output[i], target[i]) for i in range(num_losses)]) / num_losses
                    l1_loss = l1(restored, clean_patch_1)
                    if not opt.num_frequency_bands_l1 == -1:
                        l1_loss = l1_loss + opt.frequency_l1_loss_weight * l1(decompose(restored), decompose(clean_patch_1))
                    loss = l1_loss + opt.contrast_loss_weight * contrast_loss

                # backward
                loss.backward()
                optimizer.step()

            if epoch < opt.epochs_encoder:
                print(
                    'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                        epoch, contrast_loss.item(),
                    ), '\r', end='')
                train_log_file.write(
                    'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                        epoch, contrast_loss.item(),
                    ))
                train_log_file.flush()
            else:
                print(
                    'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n' % (
                        epoch, l1_loss.item(), contrast_loss.item(),
                    ), '\r', end='')
                train_log_file.write(
                    'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n' % (
                        epoch, l1_loss.item(), contrast_loss.item(),
                    ))
                train_log_file.flush()

            GPUS = 1
            if epoch + 1 == opt.epochs:
                checkpoint = {
                    "net": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                }
                if GPUS == 1:
                    torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
                else:
                    torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            
            if epoch >= opt.epochs_encoder:
                result_log_file.write('%s Epochs Results:\n'%str(epoch + 1))
                
                net.eval()
                for task in opt.test_de_type:
                    result = test_by_task(net, task=task, epochs=epoch + 1)
                    result_log_file.write(task + ': ' + ' ' * (25 - len(task)) + result + '\n')

                result_log_file.flush()
                net.train()

        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 60))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
                
        # if not opt.frequency_decompose_type == 'none':
        #     for i in range(net.E.E.encoder_q.depth):
        #         lamb_q_log_file.write(str(net.E.E.encoder_q.transformer.layers[i][0].fn.lamb.tolist()) + '\n')
        #     lamb_q_log_file.flush()
            
        #     for i in range(net.E.E.encoder_k.depth):
        #         lamb_k_log_file.write(str(net.E.E.encoder_k.transformer.layers[i][0].fn.lamb.tolist()) + '\n')
        #     lamb_k_log_file.flush()

                
    train_log_file.close()
    
    plot_loss_curve(opt.output_path)
