import subprocess
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestDataset, checkout
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model import AirNet

from option import options as opt

def test_by_task(net, task):
    print('starting testing %s...'%(task))

    checkout(opt.output_path)
    output_path = opt.output_path + 'test_' + task + '/'
    checkout(output_path)
        
    test_log_file = open(output_path + 'test.log', "w")
    
    if opt.save_imgs:
        output_imgs_path = output_path + 'output_imgs/'
        checkout(output_imgs_path)
    
    testset = TestDataset(opt, task)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([img_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            if opt.save_imgs:
                save_image_tensor(restored, output_imgs_path + img_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        test_log_file.write("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        test_log_file.close()


if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_' + str(opt.epochs) + '.pth', map_location=torch.device(opt.cuda)))
    
    for task in opt.test_de_type:
        test_by_task(net, task=task)
