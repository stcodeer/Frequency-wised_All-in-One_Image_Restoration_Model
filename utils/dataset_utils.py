import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation, crop_img

def get_data_ids(dir, need_synthesize=False):
    input_dir = dir + 'Input/'
    gt_dir = dir + 'GT/'
    
    input_ids = []
    gt_ids = []
    
    if need_synthesize:
        for file in gt_dir:
            gt_ids.append(os.path.join(gt_dir, file))
            input_ids.append('')
    
    else:
        for file in input_dir:
            l = file.split('_|.')
            gt_ids.append(os.path.join(gt_dir, l[0] + '.' + l[-1]))
            input_ids.append(os.path.join(input_dir, file))
        
    return gt_ids, input_ids
            
            
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

    def _init_ids(self):
        for i in range(len(self.de_type)):
            if 'denoising' in self.de_type[i]:
                data_dir = 'data/' + self.de_type[i].split('_')[0] + '_train/'
                self.gt_ids[i], self.input_ids[i] = get_data_ids(data_dir, True)
            else:
                data_dir = 'data/' + self.de_type[i] + '_train/'
                self.gt_ids[i], self.input_ids[i] = get_data_ids(data_dir, False)
        
        random.shuffle(self.gt_ids[i])
        random.shuffle(self.input_ids[i])

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def __getitem__(self, _):
        de_num = self.de_type_iterator % len(self.de_type)
        
        gt_id = self.gt_ids[de_num][self.de_iterator[de_num]]
        input_id = self.input_ids[de_num][self.de_iterator[de_num]]
        
        # get clean image
        gt_img = crop_img(np.array(Image.open(gt_id).convert('RGB')), base=16)
        gt_name = gt_id.split("/")[-1].split('.')[0]
        
        # get degradation image
        if 'denoising' in self.de_type[de_num]:
            sigma = int(self.de_type[de_num].split('_')[-1])
            input_img = np.clip(gt_img+np.random.randn(*gt_img.shape)*sigma, 0, 255).astype(np.uint8)
        else:
            input_img = crop_img(np.array(Image.open(input_id).convert('RGB')), base=16)

        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(input_img, gt_img))
        degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(input_img, gt_img))
        
        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)
        
        self.de_iterator[de_num] = (self.de_iterator[de_num] + 1) % len(self.gt_ids[de_num])
        
        self.de_type_iterator = (self.de_type_iterator + 1) % len(self.de_type)
        
        return [gt_name, self.de_type[de_num]], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2

    def __len__(self):
        return 400 * len(self.args.de_type)


# class DenoiseTestDataset(Dataset):
#     def __init__(self, args):
#         super(DenoiseTestDataset, self).__init__()
#         self.args = args
#         self.clean_ids = []
#         self.sigma = 15

#         self._init_clean_ids()

#         self.toTensor = ToTensor()

#     def _init_clean_ids(self):
#         name_list = os.listdir(self.args.denoise_path)
#         self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

#         self.num_clean = len(self.clean_ids)

#     def _add_gaussian_noise(self, clean_patch):
#         noise = np.random.randn(*clean_patch.shape)
#         noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
#         return noisy_patch, clean_patch

#     def set_sigma(self, sigma):
#         self.sigma = sigma

#     def __getitem__(self, clean_id):
#         clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
#         clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

#         noisy_img, _ = self._add_gaussian_noise(clean_img)
#         clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

#         return [clean_name], noisy_img, clean_img

#     def __len__(self):
#         return self.num_clean


# class DerainDehazeDataset(Dataset):
#     def __init__(self, args, task="derain"):
#         super(DerainDehazeDataset, self).__init__()
#         self.ids = []
#         self.task_idx = 0
#         self.args = args

#         self.task_dict = {'derain': 0, 'dehaze': 1}
#         self.toTensor = ToTensor()

#         self.set_dataset(task)

#     def _init_input_ids(self):
#         if self.task_idx == 0:
#             self.ids = []
#             name_list = os.listdir(self.args.derain_path + 'input/')
#             self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
#         elif self.task_idx == 1:
#             self.ids = []
#             name_list = os.listdir(self.args.dehaze_path + 'input/')
#             self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

#         self.length = len(self.ids)

#     def _get_gt_path(self, degraded_name):
#         if self.task_idx == 0:
#             gt_name = degraded_name.replace("input", "target")
#         elif self.task_idx == 1:
#             dir_name = degraded_name.split("input")[0] + 'target/'
#             name = degraded_name.split('/')[-1].split('_')[0] + '.png'
#             gt_name = dir_name + name
#         return gt_name

#     def set_dataset(self, task):
#         self.task_idx = self.task_dict[task]
#         self._init_input_ids()

#     def __getitem__(self, idx):
#         degraded_path = self.ids[idx]
#         clean_path = self._get_gt_path(degraded_path)

#         degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
#         clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

#         clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
#         degraded_name = degraded_path.split('/')[-1][:-4]

#         return [degraded_name], degraded_img, clean_img

#     def __len__(self):
#         return self.length


# class TestSpecificDataset(Dataset):
#     def __init__(self, args):
#         super(TestSpecificDataset, self).__init__()
#         self.args = args
#         self.degraded_ids = []
#         self._init_clean_ids(args.test_path)

#         self.toTensor = ToTensor()

#     def _init_clean_ids(self, root):
#         name_list = os.listdir(root)
#         self.degraded_ids += [root + id_ for id_ in name_list]

#         self.num_img = len(self.degraded_ids)

#     def __getitem__(self, idx):
#         degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
#         name = self.degraded_ids[idx].split('/')[-1][:-4]

#         degraded_img = self.toTensor(degraded_img)

#         return [name], degraded_img

#     def __len__(self):
#         return self.num_img
