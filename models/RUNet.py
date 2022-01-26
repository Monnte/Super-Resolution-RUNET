import torch
import torch.nn as nn
from torchvision.transforms.functional import resize


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x_old):
        x = self.blocks(x_old)

        if x.shape != x_old.shape:
            return x + torch.cat((x_old, x_old), dim=1)

        return x + x_old


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * upscale_factor ** 2, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return self.blocks(x)


class RUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, upscale_factor=2):
        super().__init__()

        self.features = [64, 128, 256, 512, 1024]

        self.start = nn.Sequential(
            nn.Conv2d(in_channels, self.features[0], kernel_size=(7, 7), padding="same"),
            nn.BatchNorm2d(self.features[0]),
            nn.ReLU(),
        )

        self.e1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            down_block(self.features[0], self.features[0]),
            down_block(self.features[0], self.features[0]),
            down_block(self.features[0], self.features[0]),
            down_block(self.features[0], self.features[1]),
        )

        self.e2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            down_block(self.features[1], self.features[1]),
            down_block(self.features[1], self.features[1]),
            down_block(self.features[1], self.features[1]),
            down_block(self.features[1], self.features[2]),
        )

        self.e3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[2]),
            down_block(self.features[2], self.features[3]),
        )

        self.e4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            down_block(self.features[3], self.features[3]),
            down_block(self.features[3], self.features[3]),
            nn.BatchNorm2d(self.features[3]),
            nn.ReLU(),
        )

        self.b_neck = nn.Sequential(
            nn.Conv2d(self.features[3], self.features[4], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(self.features[4], self.features[3] * upscale_factor ** 2, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor),
        )

        self.d1 = up_block(512 + self.features[3], 512, upscale_factor)
        self.d2 = up_block(512 + self.features[3], 384, upscale_factor)
        self.d3 = up_block(384 + self.features[2], 256, upscale_factor)
        self.d4 = up_block(256 + self.features[1], 96, upscale_factor)

        self.end = nn.Sequential(
            nn.Conv2d(96 + self.features[0], 99, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(99, 99, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(99, out_channels, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        x1 = self.start(x)

        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)
        x5 = self.e4(x4)

        bn = self.b_neck(x5)

        bn = resize(bn, size=x5.shape[2:])
        bn = torch.cat([bn, x5], dim=1)
        u1 = self.d1(bn)

        u1 = resize(u1, size=x4.shape[2:])
        u1 = torch.cat([u1, x4], dim=1)
        u2 = self.d2(u1)

        u2 = resize(u2, size=x3.shape[2:])
        u2 = torch.cat([u2, x3], dim=1)
        u3 = self.d3(u2)

        u3 = resize(u3, size=x2.shape[2:])
        u3 = torch.cat([u3, x2], dim=1)
        u4 = self.d4(u3)

        u4 = resize(u4, size=x1.shape[2:])
        u4 = torch.cat([u4, x1], dim=1)

        output = self.end(u4)

        return output
