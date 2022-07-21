from configparser import Interpolation
import math
import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


INTERPOLATION_NEAREST = 0
INTERPOLATION_BILINEAR = 2
INTERPOLATION_BICUBIC = 3


# https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size, interpolation=INTERPOLATION_BILINEAR):
        self.size = size
        self.img_resize = T.Resize(size, interpolation=interpolation)
        self.target_resize = T.Resize(size, interpolation=INTERPOLATION_NEAREST)

    def __call__(self, image, target):
        image = self.img_resize(image)
        target = self.target_resize(target)

        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=INTERPOLATION_NEAREST)
        return image, target


class RandomHorizontalFlip:

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


# https://pytorch.org/vision/0.8/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
class RandomResizedCrop:
    def __init__(self, size, 
                 scale=(0.08, 1.0), 
                 ratio=(3. / 4., 4. / 3.), 
                 interpolation=INTERPOLATION_BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(image, scale, ratio):
        width, height = F.get_image_size(image)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, (self.size, self.size), self.interpolation)
        target = F.resized_crop(target, i, j, h, w, (self.size, self.size), INTERPOLATION_NEAREST)

        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class MergeContours:
    def __call__(self, image, target):
        contours = (target == 255)
        target[contours] = 0

        return image, target


class ToTensor:
    def __init__(self):
        self.to_tensor = T.ToTensor()
    
    def __call__(self, image, target):
        image = self.to_tensor(image)
        target = torch.as_tensor(np.array(target))
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToBinaryMask:
    def __call__(self, image, target):
        foreground = (target != 0)
        contours = (target == 255)

        target[foreground] = 1
        target[contours] = 255

        return image, target