from utils.visualization_utils import plot_curve
import os
import re
import numpy as np


path = 'output/crop_test_imgs_size_128_de_type_denoising_15_denoising_25_denoising_50_deraining_test_de_type_denoising_bsd68_15_denoising_bsd68_25_denoising_bsd68_50_deraining'

f = np.zeros((4, 10))

for i in range(100, 1100, 100):
    file_name = 'epoch_%s_results.log'%str(i)
    file = open(os.path.join(path, file_name), 'r')
    Lines = file.readlines()
    file.close()
    
    for idx, line in enumerate(Lines):
        strings = re.split('[:,\s]', line.strip())
        for pos, string in enumerate(strings):
            if strings[pos] == 'PSNR':
                f[idx][(i - 100) // 100] = float(strings[pos + 2])
                
                

print(f)
                
plot_curve(f, x_range=(100, 1100, 100), 
           labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
           xlabel='Epochs', 
           ylabel='PSNR', 
           save_path=os.path.join(path, 'PSNR_curve.png'))            
    

