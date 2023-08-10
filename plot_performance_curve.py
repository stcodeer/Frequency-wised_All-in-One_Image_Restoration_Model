from utils.visualization_utils import plot_curve
import os
import re
import numpy as np

from option import options as opt

f = np.zeros((len(opt.test_de_type), opt.epochs // 100))

for i in range(100, opt.epochs + 100, 100):
    file_name = 'epoch_%s_results.log'%str(i)
    file = open(os.path.join(opt.output_path, file_name), 'r')
    Lines = file.readlines()
    file.close()
    
    for idx, line in enumerate(Lines):
        strings = re.split('[:,\s]', line.strip())
        for pos, string in enumerate(strings):
            if strings[pos] == 'PSNR':
                f[idx][(i - 100) // 100] = float(strings[pos + 2])
                
plot_curve(f, x_range=(100, opt.epochs + 100, 100), 
           labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
           xlabel='Epochs', 
           ylabel='PSNR', 
           save_path=os.path.join(opt.output_path, 'PSNR_curve.png'))            
    

