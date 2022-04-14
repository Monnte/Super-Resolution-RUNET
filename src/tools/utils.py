"""
Implementation of utilities used for training, validation and evaluating.

:filename utils.py
:date 06.02.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from ignite.metrics import PSNR, SSIM
from PIL import Image
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    GaussianBlur,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)
from torchvision.transforms.functional import resize


def print_config(config):
    """Print configuration file in format -> key : value."""
    print("Config file:")
    for key in config:
        print(f"{key} : {config[key]}")
    print("------------")


def ssim(input, target, data_range=1.0):
    """Structural similarity index measure function."""
    ssim = SSIM(data_range=data_range)
    ssim.update((input, target))
    return ssim.compute()


def psnr(input, target, data_range=1.0):
    """Peak noise signal ratio function."""
    psnr = PSNR(data_range=data_range)
    psnr.update((input, target))
    return psnr.compute()


def isImage(file):
    """Check file name extension."""
    return Path(file).suffix in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def hr_transform_train(crop_size):
    """Image transofrmation for dataset ground truh images in train mode."""
    return Compose(
        [
            RandomCrop(crop_size),
            RandomHorizontalFlip(0.25),
            RandomVerticalFlip(0.25),
            ToTensor(),
        ]
    )


def hr_transform_valid(crop_size):
    """Image transofrmation for dataset ground truth images in validation mode."""
    return Compose(
        [
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )


def lr_transform_train(crop_size, upscale):
    """Image transofrmation for dataset low-resolution images in train mode."""
    return Compose(
        [
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BICUBIC),
            GaussianBlur((3, 3), sigma=(0.1, 0.3)),
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            GaussianNoise(std=0.05),
        ]
    )


def lr_transform_valid(crop_size, upscale):
    """Image transofrmation for dataset low-resolution images in validation mode."""
    return Compose(
        [
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BICUBIC),
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ]
    )


def valid_bilinear_transform(crop_size, upscale):
    """
    Image transofrmation to low-resolution image using bilinear transformation.

    Used for validation metrics for bilinear output.
    """
    return Compose(
        [
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BILINEAR),
            Resize(crop_size, interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )


class GaussianNoise:
    """Gaussian noise transformation for tensor."""

    def __init__(self, mean=0.0, std=1.0):
        """Gaussian noise initialization."""
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """Apply Gausian noise to tensor."""
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0, 1)


class MineDataset(Dataset):
    """Dataset from folder class implementation."""

    def __init__(self, folder, upscale, crop_size, type):
        """Dataset class initialization."""
        super(MineDataset, self).__init__()
        assert type in {"train", "valid"}
        if type == "train":
            assert crop_size % upscale == 0, "Crop size need to by perfectly divisible by scaling factor!"
            self.hr_transform = hr_transform_train(crop_size)
            self.lr_transform = lr_transform_train(crop_size, upscale)
        else:
            self.hr_transform = hr_transform_valid(crop_size)
            self.lr_transform = lr_transform_valid(crop_size, upscale)

        self.filenames = [os.path.join(folder, x) for x in os.listdir(folder) if isImage(x)]

    def __getitem__(self, i):
        """Load high-resolution image and create low-resolution image."""
        in_image = Image.open(self.filenames[i]).convert("RGB")
        hr_image = self.hr_transform(in_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        """Return count of images."""
        return len(self.filenames)


class PerceptualLoss(nn.Module):
    """Perceptual Loss implementation from paper: https://arxiv.org/pdf/1603.08155.pdf."""

    def __init__(self, feature_layers=[0, 1, 2, 3, 4]):
        """Perceptual loss initialization."""
        super(PerceptualLoss, self).__init__()

        # load model
        vgg = torchvision.models.vgg16(pretrained=True).eval()

        # turn off gradient
        for param in vgg.parameters():
            param.requires_grad = False

        # extract blocks
        blocks = []
        blocks.append(vgg.features[:4])
        blocks.append(vgg.features[4:9])
        blocks.append(vgg.features[9:16])
        blocks.append(vgg.features[16:23])
        blocks.append(vgg.features[24:30])

        self.blocks = torch.nn.ModuleList(blocks)

        self.resize = resize
        self.loss_function = mse_loss
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.feature_layers = feature_layers

    def forward(self, input, target):
        """Forward function for perceptual loss."""
        # If image have less then 3 channels, make them 3
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize input
        input = self.normalize(input)
        target = self.normalize(target)

        # Resize to 224x224 as default VGG16 was trained
        input = self.resize(input, (224, 224))
        target = self.resize(target, (224, 224))

        # Calculate loss trought each block
        loss = 0.0
        for i, block in enumerate(self.blocks):
            input = block(input)
            target = block(target)
            if i in self.feature_layers:
                loss += self.loss_function(input, target)

        return loss
