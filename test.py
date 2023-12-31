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

def test_by_task(net, task, epochs):
    print('starting testing %s...'%(task))
    
    if opt.save_imgs:
        checkout(opt.output_path)
        
        output_path = opt.output_path + 'epoch_%s_imgs/'%str(epochs)
        checkout(output_path)
        output_path = output_path + 'test_' + task + '/'
        checkout(output_path)
        
        # test_log_file = open(output_path + 'test.log', "w")
    
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
                save_image_tensor(restored, output_path + img_name[0] + '.png')

        print("PSNR/SSIM: %.2f/%.4f" % (psnr.avg, ssim.avg))
        # test_log_file.write("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        # test_log_file.close()

    return "PSNR/SSIM: %.2f/%.4f" % (psnr.avg, ssim.avg)

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))
    
    result_log_file = open(os.path.join(opt.output_path, 'epoch_%s_results.log'%str(opt.epochs)), "w")
    
    for task in opt.test_de_type:
        result = test_by_task(net, task=task, epochs=opt.epochs)
        result_log_file.write(task + ': ' + ' ' * (25 - len(task)) + result + '\n')
