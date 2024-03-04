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

    def __init__(self, data_path, size, apply_positional_encoding=False, apply_ldi_augmentation = False, apply_rgb_transform=False, multiply_data=None,
                 flip=0, gray_scale=0, gaussian_blur=0, ldi_mult_augmentation_factor=[0.75, 1.25],
                 ignore_wrong_ldi_images=False):
        super().__init__()
        self.size = size
        self.apply_positional_encoding = apply_positional_encoding
        self.apply_ldi_augmentation = apply_ldi_augmentation
        self.apply_rgb_transform = apply_rgb_transform
        self.flip = flip
        self.gray_scale = gray_scale
        self.gaussian_blur = gaussian_blur
        self.ldi_mult_augmentation_factor = ldi_mult_augmentation_factor
        self.ignore_wrong_ldi_images = ignore_wrong_ldi_images

        self.transforms_1 = transforms.Compose([
                Resize(self.size),
                NormalizeLDI()
            ])
        
        self.transforms_2 = transforms.Compose([
                Transforms_on_RGB(flip, gray_scale, gaussian_blur)
            ])

        folder_list = glob.glob(os.path.join(data_path, "*"))
        self.ldi_files = []
        self.rgb_files = []
        # count = 0
        for folder_path in folder_list:
            ldi_path = folder_path + "/ldi.npy"
            rgb_path = folder_path + "/rgb.png"

            if not (os.path.exists(ldi_path) and os.path.exists(rgb_path)):
                # delete folder using shutil
                # shutil.rmtree(folder_path)
                print(f"Can't read {folder_path}")
                continue

            """
            ignord_cond = False

            if self.ignore_wrong_ldi_images:
                ldi = np.load(ldi_path)
                ignord_cond = ignord_cond or (np.all(ldi == 0))

                ignord_cond = ignord_cond or (np.any(ldi[:, :, 0] < 0.1))

                if ignord_cond:
                    shutil.rmtree(folder_path)
                    print(f"deleted {folder_path}")
                    count += 1
                    continue
            """

            self.ldi_files.append(ldi_path)
            self.rgb_files.append(rgb_path)

        # print(f"delete {count} invalid ldis")

        if multiply_data is not None:
            self.ldi_files = self.ldi_files * multiply_data
            self.rgb_files = self.rgb_files * multiply_data

    def __len__(self):
        return len(self.ldi_files)

    def __getitem__(self, idx):
        ldi_path = self.ldi_files[idx]
        ldi = np.load(ldi_path) # shape 1024 x 1024 x 10

        # load rgb as condition
        rgb_path = self.rgb_files[idx]
        rgb = Image.open(rgb_path)
        rgb = np.array(rgb).astype(np.uint8) # shape 1024 x 1024 x 3

        if self.apply_positional_encoding:
            ldi = ldi + self.positional_encoding(idx)

        if self.apply_ldi_augmentation:
            ldi = self.LDIMultiplyAugmentation(ldi, self.ldi_mult_augmentation_factor)

        sample = {
            'scene_id': idx,
            'ldi': ldi[:, :, :3],
            'rgb': rgb
        }

        sample = self.transforms_1(sample)
        
        if self.apply_rgb_transform: 
            sample = self.transforms_2(sample)

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
     
    def LDIMultiplyAugmentation(self, ldi, factor_range):
        factor = np.random.uniform(factor_range[0], factor_range[1])
        ldi = ldi * factor

        return ldi



class Resize:

    def __init__(self, size, interpolation="bicubic"):
        self.size = size
        self.interpolation = {"linear": PIL.Image.Resampling.NEAREST,
                              "bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]

    def __call__(self, sample):
        ldi = sample['ldi']
        rgb = sample['rgb']

        ldi = cv2.resize(ldi, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        sample['ldi'] = ldi

        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((self.size, self.size), resample=self.interpolation)
        rgb = np.array(rgb).astype(np.uint8)
        sample['rgb'] = (rgb / 127.5 - 1.0).astype(np.float32)

        return sample


class NormalizeLDI:

    def __init__(self):
        self.mean = 2.450134528114036
        self.std = 1.1005151495167442

    def NormalizeLDI(self, ldi):
        ldi = (ldi - self.mean) / self.std
        return ldi

    def __call__(self, sample):
        sample['ldi'] = self.NormalizeLDI(sample['ldi'])
        return sample


class Transforms_on_RGB:

    def __init__(self, flip_p, gray_p, gaussian_p):
        self.to_tensor = transforms.ToTensor()
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.gray_scale = transforms.RandomGrayscale(p=gray_p)
        self.gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=gaussian_p)
    
    def __call__(self, sample):
        rgb = sample['rgb']

        rgb_augmentation = transforms.Compose([
            self.to_tensor,
            self.color_jitter,
            self.flip,
            self.gray_scale,
            self.gaussian_blur
        ])

        augmented_rgb = rgb_augmentation(rgb).numpy()
        augmented_rgb = np.transpose(augmented_rgb, (1, 2, 0))
        sample['rgb'] = augmented_rgb

        return sample


class LayeredDepthImageTrain(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_path='../data/train', **kwargs)


class LayeredDepthImageValidation(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_path='../data/val', **kwargs)

class LayeredDepthImageTest(LayeredDepthImage):
    def __init__(self, **kwargs):
        super().__init__(data_path='../data/test', **kwargs)
