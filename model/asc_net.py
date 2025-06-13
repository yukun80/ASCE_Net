# model/asc_net.py (升级版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexConvTranspose2d


class ComplexDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class ComplexDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # MaxPool operates on magnitude, but for simplicity we apply it on real/imag separately
            ComplexDoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        # A simple way to apply MaxPool to complex data
        x_real = F.max_pool2d(x.real, 2)
        x_imag = F.max_pool2d(x.imag, 2)
        x_pooled = torch.complex(x_real, x_imag)
        return self.maxpool_conv[1](x_pooled)


class ComplexUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ASCNetComplex(nn.Module):
    def __init__(self, n_channels=1, n_params=2):
        super().__init__()
        # Note: input channel is 1 (complex), but internally layers handle real/imag parts
        self.inc = ComplexDoubleConv(n_channels, 64)
        self.down1 = ComplexDown(64, 128)
        self.down2 = ComplexDown(128, 256)
        self.down3 = ComplexDown(256, 512)
        self.down4 = ComplexDown(512, 1024)
        self.up1 = ComplexUp(1024, 512)
        self.up2 = ComplexUp(512, 256)
        self.up3 = ComplexUp(256, 128)
        self.up4 = ComplexUp(128, 64)

        # The output must be real-valued (2 channels for A and alpha)
        # So the last layer is a standard real convolution
        self.outc = nn.Conv2d(64, n_params, kernel_size=1)

    def forward(self, x):  # x is a complex tensor
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Take the magnitude of the final complex feature map before the output layer
        x_mag = x.abs()

        params_out = self.outc(x_mag)
        return params_out
