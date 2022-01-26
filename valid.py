import torch
from utils import ssim, psnr
from tqdm import tqdm
from models.UNet import UNet
from models.RUNet import RUNet
from utils import MineDataset, valid_bilinear_transform
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torchvision.utils import save_image


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--upscale", action="store", type=int, default=2, help="Upscale value")
    parser.add_argument("--crop", action="store", type=int, default=1024, help="Crop size")
    parser.add_argument("--model", action="store", default="UNET", help="Model to use, UNET or RUNET")
    parser.add_argument("--dataset", action="store", default="./datasets/comics_valid", help="Path to dataset folder")
    parser.add_argument("--name", action="store", default="./trainedModel.pth", help="Path to pretrained model")

    args = parser.parse_args()
    assert args.model in {"UNET", "RUNET"}, "Available models {UNET,RUNET}"

    print(
        f"model = {args.model}"
        f"\nname = {args.name}"
        f"\ndataset = {args.dataset}"
        f"\nupscale = {args.upscale}"
        f"\ncrop size = {args.crop}"
    )

    if args.model == "UNET":
        model = UNet().to(device)
    if args.model == "RUNET":
        model = RUNet().to(device)

    model.load_state_dict(torch.load(args.name))

    valid_data = MineDataset(args.dataset, args.upscale, args.crop, "valid")
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    model_psnr = []
    model_ssim = []
    interpolation_psnr = []
    interpolation_ssim = []

    model.load_state_dict(torch.load(args.name))
    model.eval()
    with torch.no_grad():
        for i, (lr_image, hr_image) in enumerate(tqdm(valid_loader)):
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            outputs = model(lr_image)

            bi_image = valid_bilinear_transform(args.crop, args.upscale)(hr_image.squeeze()).unsqueeze(0).to(device)
            save_image(outputs, f"./outputs/{i}_r_out.jpg")
            save_image(bi_image, f"./outputs/{i}_r_lr.jpg")
            save_image(hr_image, f"./outputs/{i}_r_hr.jpg")
            model_psnr.append(psnr(hr_image, outputs, 1.0))
            model_ssim.append(ssim(hr_image, outputs, 1.0))
            interpolation_psnr.append(psnr(hr_image, bi_image, 1.0))
            interpolation_ssim.append(ssim(hr_image, bi_image, 1.0))

    model_psnr = np.array(model_psnr)
    model_ssim = np.array(model_ssim)
    print("--OUTPUT--")
    print(f"PSNR Interpolation: {np.average(interpolation_psnr):.3f}")
    print(f"SSIM Interpolation: {np.average(interpolation_ssim):.3f}")
    print("")
    print(f"PSNR Model: {np.average(model_psnr):.3f}")
    print(f"SSIM Model: {np.average(model_ssim):.3f}")
