import numpy as np
import os
import PIL
from PIL import Image
import cv2
import shutil
import glob

import torch.utils.data.dataset
import torch
from torchvision import transforms


class LayeredDepthImage(torch.utils.data.dataset.Dataset):

    def __init__(self, data_paths, size, apply_positional_encoding=False, apply_transform=False, multiply_data=None,
                 flip=0, take_first=None, ldi_mult_augmentation_prob=0.0, ldi_mult_augmentation_factor=[0.75, 1.25], ignore_wrong_ldi_images=True,
                 num_of_ldi_channels=3):
        super().__init__()
        
        self.size = size
        self.apply_positional_encoding = apply_positional_encoding
        self.apply_transform = apply_transform
        self.flip = flip
        self.ignore_wrong_ldi_images = ignore_wrong_ldi_images
        self.num_of_ldi_channels = num_of_ldi_channels

        self.transforms = transforms.Compose([
            LDIMultiplyAugmentation(ldi_mult_augmentation_prob, ldi_mult_augmentation_factor),
            Resize(self.size, self.flip),
            NormalizeLDI()
        ])

        folder_list = []
        
        for data_path in data_paths:
            folder_list += glob.glob(os.path.join(data_path, "*"))

        self.ldi_files = []
        self.rgb_files = []
        for folder_path in folder_list:
            ldi_path = folder_path + "/ldi.npy"
            rgb_path = folder_path + "/rgb.png"

            if not (os.path.exists(ldi_path) and os.path.exists(rgb_path)):
                # delete folder using shutil
                # shutil.rmtree(folder_path)
                print(f"Can't read {folder_path}")
                continue
            
            ignore_cond = False

            if self.ignore_wrong_ldi_images:

                # check if ldi is all 0
                ldi = np.load(ldi_path)
                ignore_cond = ignore_cond or (np.all(ldi == 0))
                
                # check if there is any pixel lowwer than 0.1
                ignore_cond = ignore_cond or (np.any(ldi[:,:,0] < 0.1))

                # if delete_cond:
                #     # delete folder using shutil
                #     shutil.rmtree(folder_path)
                #     print(f"Deleted {folder_path}")
                #     continue

            if ignore_cond:
                print(f"Ignored {folder_path}")
                continue

            self.ldi_files.append(ldi_path)
            self.rgb_files.append(rgb_path)

        # print ignored file count
        print(f"Ignored {len(folder_list) - len(self.ldi_files)} files!")

        if take_first is not None:
            self.ldi_files = self.ldi_files[:take_first]
            self.rgb_files = self.rgb_files[:take_first]

            print(f"Taking first {take_first} samples!!!!!!!!!!!!!!!!!!!!!!!!")

        if multiply_data is not None:
            self.ldi_files = self.ldi_files * multiply_data
            self.rgb_files = self.rgb_files * multiply_data

            print(f"Multiplying data by {multiply_data}!!!!!!!!!!!!!!!!!!!!!!!!")

    def __len__(self):
        return len(self.ldi_files)

    def __getitem__(self, idx):
        ldi_path = self.ldi_files[idx]
        ldi = np.load(ldi_path)  # shape 1024 x 1024 x 10
        # ldi = np.transpose(ldi, (2, 0, 1))  # shape 10 x 1024 x 1024

        # load rgb as condition
        rgb_path = self.rgb_files[idx]
        rgb = Image.open(rgb_path)
        rgb = np.array(rgb).astype(np.uint8)  # shape 1024 x 1024 x 3

        if self.apply_positional_encoding:
            ldi = ldi + self.positional_encoding(idx)

        sample = {
            'scene_id': idx,
            'ldi': ldi[:, :, :self.num_of_ldi_channels],
            'rgb': rgb
        }

        if self.apply_transform:
            
            sample = self.transforms(sample)

        return sample

    # apply positional encoding along channels to the ldi
    def positional_encoding(self, idx):
        ldi_path = self.ldi_files[idx]
        ldi = np.load(ldi_path)  # shape 1024 x 1024 x 10
        ldi = ldi.transpose((2, 0, 1))  # shape 10 x 1024 x 1024
        ldi = ldi.reshape((10, -1))  # shape 10 x (1024*1024)

        n_position = ldi.shape[0]
        d_hid = ldi.shape[1]

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        sinusoid_table = sinusoid_table.reshape((10, 1024, 1024))

        return sinusoid_table.transpose((2, 0, 1))

class LDIMultiplyAugmentation:

    def __init__(self, prob, factor_range):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, sample):

        if np.random.rand() < self.prob:
            ldi = sample['ldi']
            factor = np.random.uniform(self.factor_range[0], self.factor_range[1])
            ldi = ldi * factor
            sample['ldi'] = ldi

        return sample

class Resize:

    def __init__(self, size, flip_p, interpolation="bicubic"):
        self.size = size
        self.interpolation = {"linear": PIL.Image.Resampling.NEAREST,
                              "bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __call__(self, sample):
        ldi = sample['ldi']
        rgb = sample['rgb']

        ldi = cv2.resize(ldi, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        sample['ldi'] = ldi

        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((self.size, self.size), resample=self.interpolation)
        # rgb = self.flip(rgb)
        rgb = np.array(rgb).astype(np.uint8)
        sample['rgb'] = (rgb / 127.5 - 1.0).astype(np.float32)

        return sample


class NormalizeLDI:

    def __init__(self):
        self.mean = 2.450134528114036
        self.std = 1.1005151495167442

        print(f"Using mean: {self.mean} and std: {self.std}")

    def NormalizeLDI(self, ldi):
        ldi = (ldi - self.mean) / self.std
        return ldi

    def __call__(self, sample):
        sample['ldi'] = self.NormalizeLDI(sample['ldi'])
        return sample


"""
class ToTensor:
    def __call__(self, sample):
        ldi = sample['ldi']
        ldi = np.transpose(ldi, (2, 0, 1))  # shape 10 x 1024 x 1024
        sample['ldi'] = ldi
        return sample
"""


class LayeredDepthImageAllTrain(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_paths=['../../renders/unused', '../../renders/train'], **kwargs)

class LayeredDepthImageTrain(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_paths=['../../renders/train'], **kwargs)

class LayeredDepthImageValidation(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_paths=['../../renders/val'], **kwargs)

class LayeredDepthImageTest(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_paths=['../../renders/test'], **kwargs)
