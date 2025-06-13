# model/asc_net.py (v2 优化版)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保 complex_layers.py 在同一目录下
from .complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexConvTranspose2d

# ... ComplexDoubleConv, ComplexDown, ComplexUp 类代码保持不变 ...
# ... 为保持代码完整性，这里全部列出 ...


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
        self.maxpool_conv = nn.Sequential(ComplexDoubleConv(in_channels, out_channels))

    def forward(self, x):
        x_real = F.max_pool2d(x.real, 2)
        x_imag = F.max_pool2d(x.imag, 2)
        x_pooled = torch.complex(x_real, x_imag)
        return self.maxpool_conv(x_pooled)


class ComplexUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- 模型优化：创建一个更轻量化的版本 ---
class ASCNetComplex_v2(nn.Module):
    def __init__(self, n_channels=1, n_params=2):
        super().__init__()
        # 相比原版，减少了通道数，让模型更轻量
        self.inc = ComplexDoubleConv(n_channels, 32)
        self.down1 = ComplexDown(32, 64)
        self.down2 = ComplexDown(64, 128)
        self.down3 = ComplexDown(128, 256)
        self.up1 = ComplexUp(256, 128)
        self.up2 = ComplexUp(128, 64)
        self.up3 = ComplexUp(64, 32)

        # 输出层前的最后一层卷积
        self.final_conv = ComplexDoubleConv(32, 32)

        self.outc = nn.Conv2d(32, n_params, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.final_conv(x)

        x_mag = x.abs()

        # --- CHANGED: Apply selective activation to output ---
        raw_params = self.outc(x_mag)

        # For Amplitude (A), we can use ReLU to ensure non-negativity.
        # Since we log-transformed the label, the network output will be log(1+A).
        # We can revert this during prediction/inference.
        pred_A = torch.relu(raw_params[:, 0:1, :, :])

        # For Alpha (α), use tanh to constrain the output to [-1, 1].
        pred_alpha = torch.tanh(raw_params[:, 1:2, :, :])

        # Concatenate the processed channels back together.
        params_out = torch.cat([pred_A, pred_alpha], dim=1)

        return params_out
