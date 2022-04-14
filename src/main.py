"""
Script used for model usage.

:filename main.py
:date 25.03.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

"""

import argparse
import os
from os.path import exists

import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor
from torchvision.utils import save_image


def image_transform(image, upscale_factor):
    """Upscale image by given scale factor."""
    return Compose(
        [
            Resize(
                (image.size[1] * upscale_factor, image.size[0] * upscale_factor),
                interpolation=InterpolationMode.BICUBIC,
            ),
            ToTensor(),
        ]
    )(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store", help="Path to image file", required=True)
    parser.add_argument("--model", action="store", help="Path to model file", required=True)
    parser.add_argument("--upscale", action="store", type=int, default=2, help="Upscale factor")
    parser.add_argument("--device", action="store", default="cpu", help="Device CPU or CUDA")

    args = parser.parse_args()

    assert exists(args.image), "Image file doesn't exists."
    assert exists(args.model), "Model file doesn't exists."
    assert args.upscale in [2, 4], "Only [2,4] factors are avaiable."

    model = torch.load(args.model).to(args.device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image = image_transform(image, args.upscale).unsqueeze(0)

    with torch.no_grad():
        out = model(image)

    name = os.path.basename(args.image)
    save_image(out, f"{args.upscale}x_{name}")
    