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
        for ([img_name], input_img, clean_img) in tqdm(testloader):
            input_img, clean_img = input_img.cuda(), clean_img.cuda()
            
            _, C, H, W = input_img.shape
            patch_size = opt.crop_test_imgs_size
                
            assert H >= patch_size and W >= patch_size, "invalid test image size (%d, %d)" % (H, W)
            assert patch_size % 8 == 0, "patch size should be a multiple of window_size"
            assert _ == 1
            
            # test the image patch by patch
            h_idx_list = list(range(0, H-patch_size, patch_size)) + [H-patch_size]
            w_idx_list = list(range(0, W-patch_size, patch_size)) + [W-patch_size]
            
            patched_input_img = []
            
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    patched_input_img.append(input_img[..., h_idx:h_idx+patch_size, w_idx:w_idx+patch_size])
            
            patched_input_img = torch.cat(patched_input_img, dim=0)

            patched_restored = net(x_query=patched_input_img, x_key=patched_input_img)
            
            E = torch.zeros(C, H, W).type_as(input_img)
            W = torch.zeros_like(E)
            
            cnt = 0
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    E[..., h_idx:h_idx+patch_size, w_idx:w_idx+patch_size].add_(patched_input_img[cnt])
                    W[..., h_idx:h_idx+patch_size, w_idx:w_idx+patch_size].add_(torch.ones_like(patched_input_img[cnt]))
                    cnt = cnt + 1
            
            restored = E.div_(W).unsqueeze(0)
            
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_img)
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
    # net.load_state_dict(torch.load(opt.ckpt_path + 'epoch_%s.pth'%str(opt.epochs), map_location=torch.device(opt.cuda)))
    
    result_log_file = open(os.path.join(opt.output_path, 'epoch_%s_results.log'%str(opt.epochs)), "w")
    
    for task in opt.test_de_type:
        result = test_by_task(net, task=task, epochs=opt.epochs)
        result_log_file.write(task + ': ' + ' ' * (25 - len(task)) + result + '\n')
