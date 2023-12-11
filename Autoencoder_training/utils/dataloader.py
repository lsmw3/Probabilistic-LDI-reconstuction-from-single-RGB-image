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

    def __init__(self, data_path, size, apply_positional_encoding=False, apply_transform=False, multiply_data=None,
                 flip=0):
        super().__init__()
        self.size = size
        self.apply_positional_encoding = apply_positional_encoding
        self.apply_transform = apply_transform
        self.flip = flip

        self.transforms = transforms.Compose([
                Resize(self.size, self.flip),
                NormalizeLDI()
            ])

        folder_list = glob.glob(os.path.join(data_path, "*"))
        self.ldi_files = []
        self.rgb_files = []
        for folder_path in folder_list:
            ldi_path = folder_path + "/ldi.npy"
            rgb_path = folder_path + "/rgb.png"

            if not (os.path.exists(ldi_path) and os.path.exists(rgb_path)):
                # delete folder using shutil
                shutil.rmtree(folder_path)
                print(f"Deleted {folder_path}")
                continue
            # else:

            #     # read np
            #     ldi = np.load(ldi_path)

            #     # check if ldi is all zeros
            #     if np.all(ldi == 0):
            #         # delete folder using shutil
            #         shutil.rmtree(folder_path)
            #         print(f"Deleted {folder_path}")
            #         continue

            self.ldi_files.append(ldi_path)
            self.rgb_files.append(rgb_path)

        if multiply_data is not None:
            self.ldi_files = self.ldi_files * multiply_data

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
            'ldi': ldi[:, :, :3],
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
        self.mean = 2.5502266793814417
        self.std = 0.9683916122629385

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


class LayeredDepthImageTrain(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_path='../../renders/train', **kwargs)


class LayeredDepthImageValidation(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_path='../../renders/val', **kwargs)
