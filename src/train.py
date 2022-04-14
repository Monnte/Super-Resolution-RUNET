"""
Script used for model training.

:filename train.py
:date 10.02.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

"""

import argparse
import json
from os.path import exists, join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.RUNet import RUNet
from models.UNet import UNet
from tools.pytorchtools import EarlyStopping
from tools.utils import MineDataset, PerceptualLoss, print_config

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, valid_loader, epochs, model, loss_function, optimizer, scheduler, early_stopping, debug):
    """Train model function."""
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(epochs):
        train_losses = []
        valid_losses = []

        # Training phase
        model.train()
        for lr_image, hr_image in train_loader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            optimizer.zero_grad()

            outputs = model(lr_image)

            # Calculate loss
            loss = loss_function(outputs, hr_image)
            loss.backward()
            train_losses.append(loss.item())

            # Update weights
            optimizer.step()

        # Validation phase
        model.eval()
        for lr_image, hr_image in valid_loader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            outputs = model(lr_image)
            # Calculate loss
            loss = loss_function(outputs, hr_image)
            valid_losses.append(loss.item())

        # Statistics
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(
            f"Epoch [{epoch}/{epochs}]"
            f"\nTraining Loss: {train_loss:.6f}"
            f"\nValidation Loss:{valid_loss:.6f}"
            f"\nLR:{optimizer.param_groups[0]['lr']}"
        )

        # Scheduler and Early stopping check
        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load last best model
    model.load_state_dict(torch.load("checkpoint.pt"))
    return model, avg_train_losses, avg_valid_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", default="config_train.json", help="config file path", required=True)
    args = parser.parse_args()

    assert exists(args.config), "Config file doesn't exists. Create 'config.json' file"
    with open(args.config) as config_file:
        config = json.load(config_file)
        print_config(config)

    # Basic checks
    assert config["model"] in {"UNET", "RUNET"}, "Available models {UNET,RUNET}"
    assert config["loss"] in {"MSE", "PERCEPTUAL"}, "Available loss functions {MSE,PERCEPTUAL}"

    # Debug mode
    if "debug" in config.keys():
        debug = config["debug"]
    else:
        debug = False

    # Dataset for training
    train_data = MineDataset(config["dataset_train"], config["upscale_factor"], config["crop_size"], "train")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], num_workers=4, shuffle=True, pin_memory=True)

    # Dataset for validation
    valid_data = MineDataset(config["dataset_valid"], config["upscale_factor"], config["crop_size"], "valid")
    valid_loader = DataLoader(valid_data, batch_size=config["batch_size"], num_workers=4, shuffle=True, pin_memory=True)

    # Define model
    if config["model"] == "UNET":
        model = UNet().to(device)
    if config["model"] == "RUNET":
        model = RUNet().to(device)

    # Dfine loss function
    if config["loss"] == "MSE":
        loss_function = nn.MSELoss().to(device)
    if config["loss"] == "PERCEPTUAL":
        if "loss_layers" in config.keys():
            loss_function = PerceptualLoss(config["loss_layers"]).to(device)
        else:
            loss_function = PerceptualLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=5)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    model, train_loss, valid_loss = train(
        train_loader, valid_loader, config["epochs"], model, loss_function, optimizer, scheduler, early_stopping, debug
    )

    # Save model data
    path = config["save_path"]
    name = config["save_name"]

    model_save_path = join(path, f"{name}.pt")
    model_scripted = torch.jit.script(model.to("cpu"))
    model_scripted.save(model_save_path)

    # If debug mode is enabled save train and validation losses durning training
    if debug:
        np.save(join(path, f"{name}_train_loss.npy"), train_loss)
        np.save(join(path, f"{name}_valid_loss.npy"), valid_loss)

    print(f"Model sucesfully trained and saved. -> {model_save_path}")
