import os
import re
import numpy as np
import functools

path = '/mnt/d/1AirNet_data/output/h__cuda_0_num_frequency_bands_l1_1_degradation_embedding_method_all_DC_de_type_5tasks'
epochs = 1000
file = open(os.path.join(path, 'results.log'), 'r')
task = 5

perf = []
for i in range(task):
    perf.append([])
    
for i in range(epochs):
    title = file.readline()
    for j in range(task):
        x = file.readline()
        x = re.split('[/:,\s]', x.strip())
        perf[j].append([float(x[-2]), float(x[-1])])

def cmp(x, y):
    if x[0] == y[0]:
        return y[1] - x[1]
    return y[0] - x[0] 

for i in range(task):
    perf[i].sort(key=functools.cmp_to_key(cmp))
    psnr = [perf[i][j][0] for j in range(epochs)]
    ssim = [perf[i][j][1] for j in range(epochs)]
    print('1st: %.2lf/%.4lf'%(psnr[0], ssim[0]))
    print('2nd: %.2lf/%.4lf'%(psnr[1], ssim[1]))
    print('3rd: %.2lf/%.4lf'%(psnr[2], ssim[2]))
    print('avg: %.2lf/%.4lf'%(np.mean(psnr[0: 50]), np.mean(ssim[0: 50])))
    print('var: %.4lf/%.7lf'%(np.std(psnr[0: 50]), np.std(ssim[0: 50])))