"""
U-Net architecutre implementation.

The original U-Net paper: https://arxiv.org/abs/1505.04597.

:filename UNet.py
:date 26.01.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

"""

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize


class DoubleConvolutionBlock(nn.Module):
    """
    Double convolution block, base block of architecture.

    Convolution -> ReLU -> Convolution -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        """Double convolution block initialization."""
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward function for double convolution block."""
        return self.blocks(x)


class EncoderBlock(nn.Module):
    """
    Base block of contractig path.

    Max pooling -> Double convolution block
    """

    def __init__(self, in_channels, out_channels):
        """Encoder block initialization."""
        super().__init__()

        self.blocks = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConvolutionBlock(in_channels, out_channels)
        )

    def forward(self, x):
        """Forward function for encoder block."""
        return self.blocks(x)


class DecoderBlock(nn.Module):
    """
    Base block of expadning path.

    Upsample(bilinear) -> Convolution (halfs number of features) -> Concat skip connection -> Double convolution block
    """

    def __init__(self, in_channels, out_channels):
        """Decoder Block initialization."""
        super().__init__()
        self.up = nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = DoubleConvolutionBlock(2 * out_channels, out_channels)

    def forward(self, x, skip):
        """Forward function for decoder block."""
        x = self.up(x)
        x = self.conv1(x)

        # Resize skip coonection if needed
        if x.shape != skip.shape:
            x = resize(x, size=skip.shape[2:])
        x = torch.cat((x, skip), dim=1)

        x = self.conv2(x)

        return x


class UNet(nn.Module):
    """
    Original U-Net implementation.

    Parameters:
        in_channels (int): Number of channels for input image
        out_channels (int): Number of channels for output image
    """

    def __init__(self, in_channels=3, out_channels=3):
        """U-Net model initialization."""
        super().__init__()

        # Define features for convolutions in encoding/decoding path
        self.features = [64, 128, 256, 512, 1024]

        self.start = DoubleConvolutionBlock(in_channels, self.features[0])

        # Encoder parts
        self.e1 = EncoderBlock(self.features[0], self.features[1])
        self.e2 = EncoderBlock(self.features[1], self.features[2])
        self.e3 = EncoderBlock(self.features[2], self.features[3])

        # Bottleneck
        self.bottleneck = DoubleConvolutionBlock(self.features[3], self.features[3] * 2)

        # Decoder parts
        self.d1 = DecoderBlock(self.features[4], self.features[3])
        self.d2 = DecoderBlock(self.features[3], self.features[2])
        self.d3 = DecoderBlock(self.features[2], self.features[1])
        self.d4 = DecoderBlock(self.features[1], self.features[0])

        self.end = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Forward function for U-Net."""
        x1 = self.start(x)

        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)

        x = self.bottleneck(x4)

        x = self.d1(x, x4)
        x = self.d2(x, x3)
        x = self.d3(x, x2)
        x = self.d4(x, x1)

        output = self.end(x)
        return output


if __name__ == "__main__":
    print(UNet()(torch.rand(1, 3, 128, 128)))
