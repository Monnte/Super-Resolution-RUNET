"""
Script used for model validation.

:filename valid.py
:date 10.02.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

"""

import argparse
import json
from os.path import exists

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from tools.utils import MineDataset, print_config, psnr, ssim, valid_bilinear_transform

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", default="config_valid.json", help="Config file path", required=True)
    args = parser.parse_args()
    assert exists(args.config), "Config file doesn't exists. Create 'config.json' file"

    with open(args.config) as config_file:
        config = json.load(config_file)
        print_config(config)

    assert exists(config["model"]), "Model file doesn't exists."

    # Debug mode
    if "debug" in config.keys():
        debug = config["debug"]
    else:
        debug = False

    # Load model
    model = torch.jit.load(config["model"]).to(device)
    model.eval()

    # Dataset for validation
    valid_data = MineDataset(config["dataset_valid"], config["upscale_factor"], config["crop_size"], "valid")
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    model_psnr = []
    model_ssim = []
    interpolation_psnr = []
    interpolation_ssim = []

    with torch.no_grad():
        for i, (lr_image, hr_image) in enumerate(tqdm(valid_loader)):
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            outputs = model(lr_image)

            bi_image = (
                valid_bilinear_transform(config["crop_size"], config["upscale_factor"])(hr_image.squeeze())
                .unsqueeze(0)
                .to(device)
            )

            # Calculate metrics results
            model_psnr.append(psnr(hr_image, outputs))
            model_ssim.append(ssim(hr_image, outputs))
            interpolation_psnr.append(psnr(hr_image, bi_image))
            interpolation_ssim.append(ssim(hr_image, bi_image))

            if debug:
                save_image(outputs, f"./e{i}_out.jpg")
                save_image(bi_image, f"./e{i}_bi.jpg")
                save_image(hr_image, f"./e{i}_hr.jpg")

    # Python array transform to numpy array
    model_psnr = np.array(model_psnr)
    model_ssim = np.array(model_ssim)

    # Print statistics
    print(f"PSNR Interpolation: {np.average(interpolation_psnr):.3f}")
    print(f"SSIM Interpolation: {np.average(interpolation_ssim):.3f}")
    print("----------")
    print(f"PSNR Model: {np.average(model_psnr):.3f}")
    print(f"SSIM Model: {np.average(model_ssim):.3f}")
