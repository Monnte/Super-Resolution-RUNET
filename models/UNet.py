import torch
import torch.nn as nn
from torchvision.transforms.functional import resize


class double_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = double_conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        return p, x


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False)
        self.conv1 = double_conv_block(in_channels, out_channels)
        self.conv2 = double_conv_block(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1(x)
        if x.shape != skip.shape:
            x = resize(x, size=skip.shape[2:])
        x = torch.cat((x, skip), dim=1)
        x = self.conv2(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = double_conv_block(512, 1024)

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        self.outputs = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1, s1 = self.e1(x)
        x2, s2 = self.e2(x1)
        x3, s3 = self.e3(x2)
        x4, s4 = self.e4(x3)

        x = self.b(x4)

        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)

        outputs = self.outputs(x)

        return outputs
