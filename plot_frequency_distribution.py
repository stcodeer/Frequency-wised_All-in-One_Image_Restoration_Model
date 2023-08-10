from PIL import Image
from utils.visualization_utils import plot_curve, rgb2gray, plot_image_grid
from utils.dataset_utils import checkout

import numpy as np


def get_frequency_distribution(img):

    ftimage = np.fft.fft2(img)
    ftimage = abs(np.fft.fftshift(ftimage))

    h, w = ftimage.shape

    center = (int(w / 2), int(h / 2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    
    diag = center[0] # np.sqrt(center[0] ** 2 + center[1] ** 2)

    tot_value = [0.] * int(1 / size)
    
    for idx, sz in enumerate(np.linspace(size, 1, int(1 / size))):
        for y in range(h):
            for x in range(w):
                if sz == 1 and diag * (sz - size) <= dist_from_center[y][x] <= diag * sz:
                    tot_value[idx] = tot_value[idx] + ftimage[y][x]
                elif sz < 1 and diag * (sz - size) <= dist_from_center[y][x] < diag * sz:
                    tot_value[idx] = tot_value[idx] + ftimage[y][x]
                    
    return tot_value


size = 0.2

img_list = []

# clean image
img = np.array(Image.open('data/F16_GT.png').convert('RGB'))
img = rgb2gray(img)
img_list.append(np.expand_dims(img, 0))
f_clean = get_frequency_distribution(img)

# noise 15
img = np.random.randn(*img.shape) * 15
img_list.append(np.expand_dims(img, 0))
f_noise_15 = get_frequency_distribution(img)

# noise 25
img = np.random.randn(*img.shape) * 25
img_list.append(np.expand_dims(img, 0))
f_noise_25 = get_frequency_distribution(img)

# noise 50
img = np.random.randn(*img.shape) * 50
img_list.append(np.expand_dims(img, 0))
f_noise_50 = get_frequency_distribution(img)

# rain region
img = np.array(Image.open('data/rainregion-5.png').convert('RGB'))
img = rgb2gray(img)
img_list.append(np.expand_dims(img, 0))
f_rainregion = get_frequency_distribution(img)

# rain streak
img = np.array(Image.open('data/rainstreak-198.png').convert('RGB'))
img = rgb2gray(img)
img_list.append(np.expand_dims(img, 0))
f_rainstreak = get_frequency_distribution(img)

checkout('output_img')

for idx in range(len(img_list)):
    plot_image_grid([img_list[idx]], dpi=1000, nrow=1, save_path='output_img/default_%s.png'%idx)

plot_curve((f_clean, f_noise_15, f_noise_25, f_noise_50, f_rainregion, f_rainstreak),
        labels=('clean', 'noise_15', 'noise_25', 'noise_50', 'rain_region', 'rain_streak'), 
        xlabel='Frequency Bands', 
        ylabel='Gray Value', 
        ylim=(0, 2e9),
        save_path='output_img/frequency_distribution_center0.png')   