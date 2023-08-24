from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random
import copy
import numpy as np
import glob

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation, crop_img

def checkout(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_data_ids(dir, need_synthesize=False):
    input_dir = dir + 'Input/'
    gt_dir = dir + 'GT/'
    
    input_ids = []
    gt_ids = []
    
    if need_synthesize:
        for file in os.listdir(gt_dir):
            gt_ids.append(os.path.join(gt_dir, file))
            input_ids.append('')
    
    else:
        for file in os.listdir(input_dir):
            pre = file.split('.')[0].split('_')[0]
            suf = file.split('.')[-1]
            str = pre + '.' + suf
            
            # # regular expression
            # l = glob.glob(gt_dir + '*[!0-9]' + str)
            # l = l + glob.glob(gt_dir + str)
            # if not len(l) == 1:
            #     print('GT: ', l)
            #     print('Input: ', file)
            #     assert False, 'multi GT image names match given Input image name.'  
            # gt_ids.append(os.path.join(gt_dir, l[0].split('/')[-1]))
            
            gt_ids.append(os.path.join(gt_dir, str))
            input_ids.append(os.path.join(input_dir, file))
        
    return gt_ids, input_ids

def _crop_patch(img_1, img_2, size):
    H = img_1.shape[0]
    W = img_1.shape[1]
    ind_H = random.randint(0, H - size)
    ind_W = random.randint(0, W - size)

    patch_1 = img_1[ind_H:ind_H + size, ind_W:ind_W + size]
    patch_2 = img_2[ind_H:ind_H + size, ind_W:ind_W + size]

    return patch_1, patch_2

# def naive_convert(x):
#     if x.shape[2] == 3:
#         return x
#     elif x.shape[2] == 4:
#         return x[:, :, 0:3]
#     else:
#         assert False, 'unwelcomed image type.'

class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.de_type = self.args.de_type
        
        self.de_type_iterator = 0
        self.de_iterator = [0] * len(self.de_type)
        
        self.gt_ids = [[]] * len(self.de_type)
        self.input_ids = [[]] * len(self.de_type)
        
        self._init_ids()

        self.toTensor = ToTensor()
        
        # self.log_file = open('D:\sutong\AirNet\datasets.log', 'w')

    def _init_ids(self):
        for i in range(len(self.de_type)):
            if 'denoising' in self.de_type[i]:
                length_sigma = len(self.de_type[i].split('_')[-1]) + 1
                data_dir = 'data/' + self.de_type[i][0:-length_sigma] + '_train/'
                self.gt_ids[i], self.input_ids[i] = get_data_ids(data_dir, True)
            else:
                data_dir = 'data/' + self.de_type[i] + '_train/'
                self.gt_ids[i], self.input_ids[i] = get_data_ids(data_dir, False)
                
    def __getitem__(self, _):
        de_num = self.de_type_iterator % len(self.de_type)
        
        if self.de_iterator[de_num] == 0:
            for t in reversed(range(1, len(self.gt_ids[de_num]))):
                j = random.randrange(1, t + 1)
                self.gt_ids[de_num][t], self.gt_ids[de_num][j] = self.gt_ids[de_num][j], self.gt_ids[de_num][t]
                self.input_ids[de_num][t], self.input_ids[de_num][j] = self.input_ids[de_num][j], self.input_ids[de_num][t]
        
        
        gt_id = self.gt_ids[de_num][self.de_iterator[de_num]]
        input_id = self.input_ids[de_num][self.de_iterator[de_num]]
        
        # self.log_file.write('gt_id: ' + gt_id + '\n')

        # print('before RGB: ', np.array(Image.open(gt_id)).shape)
        # print('after RGB: ', np.array(Image.open(gt_id).convert('RGB')).shape)
        
        # assert Image.open(gt_id) == Image.open(gt_id).convert('RGB')
        
        # get clean image
        gt_img = crop_img(np.array(Image.open(gt_id).convert('RGB')), base=16)
        gt_name = gt_id.split("/")[-1].split('.')[0]
        
        # get degradation image
        if 'denoising' in self.de_type[de_num]:
            sigma = int(self.de_type[de_num].split('_')[-1])
            if sigma == 0:
                sigma = np.random.uniform(low=15, high=50)
            input_img = np.clip(gt_img+np.random.randn(*gt_img.shape)*sigma, 0, 255).astype(np.uint8)
        else:
            # self.log_file.write('input_id: ' + input_id + '\n')
            input_img = crop_img(np.array(Image.open(input_id).convert('RGB')), base=16)
         
        degrad_patch_1, clean_patch_1 = random_augmentation(*_crop_patch(input_img, gt_img, size=self.args.patch_size))
        degrad_patch_2, clean_patch_2 = random_augmentation(*_crop_patch(input_img, gt_img, size=self.args.patch_size))
        
        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)
        
        self.de_iterator[de_num] = (self.de_iterator[de_num] + 1) % len(self.gt_ids[de_num])
        
        self.de_type_iterator = (self.de_type_iterator + 1) % len(self.de_type)
        
        return [gt_name, self.de_type[de_num]], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2

    def __len__(self):
        return 400 * len(self.args.de_type)
    
    def len(self):
        return sum([len(self.gt_ids[i]) for i in range(len(self.gt_ids))])
    

class TestDataset(Dataset):
    def __init__(self, args, de_type):
        super(TestDataset, self).__init__()
        self.args = args
        self.de_type = de_type
        
        self._init_ids()
        
        self.toTensor = ToTensor()

    def _init_ids(self):
            if 'denoising' in self.de_type:
                length_sigma = len(self.de_type.split('_')[-1]) + 1
                data_dir = 'data/' + self.de_type[0:-length_sigma] + '_test/'
                self.gt_ids, self.input_ids = get_data_ids(data_dir, True)
            else:
                data_dir = 'data/' + self.de_type + '_test/'
                self.gt_ids, self.input_ids = get_data_ids(data_dir, False)

    def __getitem__(self, idx):
        gt_id = self.gt_ids[idx]
        input_id = self.input_ids[idx]
        
        # get clean image
        gt_img = crop_img(np.array(Image.open(gt_id).convert('RGB')), base=16)
        
        # get degradation image
        if 'denoising' in self.de_type:
            sigma = int(self.de_type.split('_')[-1])
            if sigma == 0:
                sigma = np.random.uniform(low=15, high=50)
            input_img = np.clip(gt_img+np.random.randn(*gt_img.shape)*sigma, 0, 255).astype(np.uint8)
            input_name = gt_id.split("/")[-1].split('.')[0]
        else:
            input_img = crop_img(np.array(Image.open(input_id).convert('RGB')), base=16)
            input_name = input_id.split("/")[-1].split('.')[0]
        
        if input_img.shape[0] > self.args.crop_test_imgs_size and input_img.shape[1] > self.args.crop_test_imgs_size:
            input_img, gt_img = _crop_patch(input_img, gt_img, size=self.args.crop_test_imgs_size)
            
        input_img = self.toTensor(input_img)
        gt_img = self.toTensor(gt_img)
        
        return [input_name], input_img, gt_img

    def __len__(self):
        return len(self.gt_ids)
    