import os
import torchvision
import torch
import torch.nn.functional as f
import torch.nn as nn
from torchvision.transforms.functional import resize
from ignite.metrics import SSIM, PSNR
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    ToPILImage,
    Resize,
    Grayscale,
    GaussianBlur,
    RandomCrop,
    Normalize,
    CenterCrop,
)
from torchvision.transforms import InterpolationMode


def ssim(input, target, data_range):
    ssim = SSIM(data_range=data_range)
    ssim.update((input, target))
    return ssim.compute()


def psnr(input, target, data_range):
    psnr = PSNR(data_range=data_range)
    psnr.update((input, target))
    return psnr.compute()


def isImage(file):
    return Path(file).suffix in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def hr_transform_train(crop_size):
    return Compose(
        [
            # Grayscale(),
            RandomCrop(crop_size),
            ToTensor(),
        ]
    )


def hr_transform_valid(crop_size):
    return Compose(
        [
            # Grayscale(),
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )


def lr_transform_train(crop_size, upscale):
    return Compose(
        [
            # GaussianNoise(std=0.01),
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BICUBIC),
            GaussianBlur((3, 3), sigma=(0.1, 0.3)),
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ]
    )


def lr_transform_valid(crop_size, upscale):
    return Compose(
        [
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BICUBIC),
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ]
    )


def valid_bilinear_transform(crop_size, upscale):
    return Compose(
        [
            ToPILImage(),
            Resize((crop_size // upscale), interpolation=InterpolationMode.BILINEAR),
            Resize(crop_size, interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )


class GaussianNoise:
    """
    Gaussian noise transformation for tensor
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0, 1)


class MineDataset(Dataset):
    """
    Dataset from folder class implementation
    """

    def __init__(self, folder, upscale, crop_size, type):
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
        in_image = Image.open(self.filenames[i]).convert("RGB")
        hr_image = self.hr_transform(in_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.filenames)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss implementation from paper https://arxiv.org/pdf/1603.08155.pdf
    """

    def __init__(self, features=23):
        super(PerceptualLoss, self).__init__()
        # load model
        vgg = torchvision.models.vgg16(pretrained=True).eval()

        # turn off gradient
        for param in vgg.parameters():
            param.requires_grad = False

        self.main = nn.Sequential(*list(vgg.features.children())[:features])

        self.resize = resize
        self.calc_loss = f.mse_loss
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, input, target):

        # prepare inputs as vgg network was trained
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = self.resize(input, (224, 224))
        target = self.resize(target, (224, 224))

        input = self.normalize(input)
        target = self.normalize(target)

        # calc loss
        return self.calc_loss(input, target)
