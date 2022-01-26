import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import MineDataset, PerceptualLoss
from models.UNet import UNet
from models.RUNet import RUNet
from tqdm import tqdm
import seaborn as sns
import argparse
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, ecpohs, model, loss, optimizer):
    trainLoss = []
    model.train()
    for epoch in tqdm(range(ecpohs)):
        loss_value = 0.0

        for lr_image, hr_image in train_loader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            outputs = model(lr_image)
            loss = loss_function(outputs, hr_image)

            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            loss_value += loss.item()

            save_image(lr_image, f"./outputs/{epoch}_lr.jpg")
            save_image(hr_image, f"./outputs/{epoch}_hr.jpg")
            save_image(outputs, f"./outputs/{epoch}_out.jpg")

        print(f"Epoch {epoch} loss: {loss_value / train_loader.batch_size:.4f}")
        trainLoss.append(loss_value / train_loader.batch_size)

    return trainLoss


def save_train_loss_plot(data, path):
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    p = sns.lineplot(data=data)
    p.set_xlabel("Epochs")
    p.set_ylabel("Loss")
    fig = p.get_figure()
    fig.tight_layout()
    fig.savefig(f"{path}_train_loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--epochs", action="store", type=int, default=100, help="Count of epochs")
    parser.add_argument("--upscale", action="store", type=int, default=2, help="Upscale value")
    parser.add_argument("--crop", action="store", type=int, default=128, help="Crop size")
    parser.add_argument("--batch_size", action="store", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", action="store", type=float, default=0.001, help="Learning rate value")
    parser.add_argument("--model", action="store", default="UNET", help="Model to use, UNET or RUNET")
    parser.add_argument("--loss", action="store", default="MSE", help="Loss function to use, MSE or PERCEPTUAL")
    parser.add_argument("--dataset", action="store", default="./datasets/comics_train", help="Path to dataset folder")
    parser.add_argument("--name", action="store", default="./trainedModel", help="Path + Name for saved model")

    args = parser.parse_args()

    assert args.model in {"UNET", "RUNET"}, "Available models {UNET,RUNET}"
    assert args.loss in {"MSE", "PERCEPTUAL"}, "Available loss functions {MSE,PERCEPTUAL}"
    print(
        f"model = {args.model}"
        f"\nloss = {args.loss}"
        f"\nname = {args.name}"
        f"\ndataset = {args.dataset}"
        f"\nbatch size = {args.batch_size}"
        f"\nepochs = {args.epochs}"
        f"\nupscale = {args.upscale}"
        f"\ncrop size = {args.crop}"
        f"\nlr = {args.lr}"
    )

    train_data = MineDataset(args.dataset, args.upscale, args.crop, "train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    if args.model == "UNET":
        model = UNet().to(device)
    if args.model == "RUNET":
        model = RUNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    if args.loss == "MSE":
        loss_function = nn.MSELoss().to(device)
    if args.loss == "PERCEPTUAL":
        loss_function = PerceptualLoss().to(device)

    loss_data = train(train_loader, args.epochs, model, loss_function, optimizer)

    # Save model
    torch.save(model.state_dict(), f"{args.name}_dict.pth")
    torch.save(model.to("cpu"), f"{args.name}_model.pt")
    model_scripted = torch.jit.script(model)
    model_scripted.save(f"{args.name}_scripted.pt")

    # traced_foo = torch.jit.trace(model, torch.rand(1, 1, 1024, 1024))
    # traced_foo.save(f"{args.name}_scripted.pt")

    save_train_loss_plot(loss_data, args.name)
    print("Model sucesfully trained and saved.")
