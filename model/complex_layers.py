# model/complex_layers.py (新增)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):  # x is a complex tensor
        real_part = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag_part = self.conv_real(x.imag) + self.conv_imag(x.real)
        return torch.complex(real_part, imag_part)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real_part = self.bn_real(x.real)
        imag_part = self.bn_imag(x.imag)
        return torch.complex(real_part, imag_part)


class ComplexReLU(nn.Module):
    def forward(self, x):
        return F.relu(x.real) + 1j * F.relu(x.imag)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.tconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.tconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        real_part = self.tconv_real(x.real) - self.tconv_imag(x.imag)
        imag_part = self.tconv_real(x.imag) + self.tconv_imag(x.real)
        return torch.complex(real_part, imag_part)
