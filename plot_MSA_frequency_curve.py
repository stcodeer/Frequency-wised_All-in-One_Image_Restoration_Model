from utils.visualization_utils import plot_curve
import os
import re
import numpy as np

from option import options as opt
from utils.dataset_utils import checkout


s = np.zeros((10, 4, 5))

checkout('output_img')
checkout('output_img/after_msa_denoising_15_25_50_75/')

for idx, sigma in enumerate(['15', '25', '50', '75']):
    file_name = 'after_MSA_%s.log'%sigma
    file = open(file_name, 'r')
    Lines = file.readlines()
    file.close()
    
    for line in Lines:
        strings = re.split('[:,\s]', line.strip())
        print(strings)
        i = int(strings[0])
        j = int(strings[1])
        f = [float(strings[2]), float(strings[3]), float(strings[4]), float(strings[5]), float(strings[6])]
        for t in range(5):
            if i in [0, 1, 4, 5, 8, 9]:
                s[i][idx][t] = s[i][idx][t] + f[t] / 2
            else:
                s[i][idx][t] = s[i][idx][t] + f[t] / 8
                
# proportion
for i in range(10):
    for j in range(4):
        t = np.sum(s[i][j])
        for k in range(5):
            s[i][j][k] = s[i][j][k] / t


for i in range(10):
    plot_curve(s[i], ylim=None,
            labels=('denoising_15', 'denoising_25', 'denoising_50', 'denoising_75'), 
            xlabel='Epochs', 
            ylabel='PSNR', 
            save_path='output_img/after_msa_denoising_15_25_50_75/proportion_layer_%s.png'%i)            
    

