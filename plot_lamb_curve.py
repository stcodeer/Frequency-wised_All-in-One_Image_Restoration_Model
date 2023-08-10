from utils.visualization_utils import plot_curve
import os
import re
import numpy as np

from option import options as opt

depths = 12
num_bands = 5
wised_batch = len(opt.de_type)
heads = 12

lamb_k = np.zeros((depths, num_bands, wised_batch, heads))
lamb_q = np.zeros((depths, num_bands, wised_batch, heads))

k_file_name = 'lamb_k.log'
q_file_name = 'lamb_q.log'

k_file = open(os.path.join(opt.output_path, k_file_name), 'r')
q_file = open(os.path.join(opt.output_path, q_file_name), 'r')

k_lines = k_file.readlines()[-depths:]
q_lines = q_file.readlines()[-depths:]

k_file.close()
q_file.close()

for idx in range(depths):
    strings_k = re.split(',', k_lines[idx].strip())
    strings_q = re.split(',', q_lines[idx].strip())
    iterator = 0
    for i in range(num_bands):
        for j in range(wised_batch):
            for k in range(heads):
                lamb_k[idx][i][j][k] = float(strings_k[iterator].replace('[', '').replace(']', '').replace(' ', ''))
                lamb_q[idx][i][j][k] = float(strings_q[iterator].replace('[', '').replace(']', '').replace(' ', ''))
                iterator = iterator + 1
        
lamb_k = np.mean(lamb_k, axis=3)
lamb_k = np.mean(lamb_k, axis=0)
lamb_k = np.transpose(lamb_k, (1, 0)) # [wised_batch, num_bands]

lamb_q = np.mean(lamb_q, axis=3)
lamb_q = np.mean(lamb_q, axis=0)
lamb_q = np.transpose(lamb_q, (1, 0)) # [wised_batch, num_bands]
    
plot_curve(lamb_k, 
        ylim=(-1, 1),
        labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
        xlabel='Frequency Bands', 
        ylabel='Lamb', 
        save_path=os.path.join(opt.output_path, 'lamb_k_curve.png'))     
 
plot_curve(lamb_q, 
        ylim=(-1, 1),
        labels=('denoising_15', 'denoising_25', 'denoising_50', 'deraining'), 
        xlabel='Frequency Bands', 
        ylabel='Lamb', 
        save_path=os.path.join(opt.output_path, 'lamb_q_curve.png'))         
    

