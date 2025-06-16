import torch
import torch.nn as nn
import torch.nn.functional as F
from .complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexConvTranspose2d


# --- Complex U-Net Building Blocks (No changes needed here) ---
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


# --- FINAL MODEL IMPLEMENTATION ---
# This version replaces your ASCNetComplex_v2_Full
class ASCNet_v3_5param(nn.Module):
    """
    Final model implementing the 5-parameter prediction strategy:
    1. Heatmap (for localization)
    2. Amplitude (A)
    3. Alpha (α)
    4. X-offset (dx)
    5. Y-offset (dy)
    """

    def __init__(self, n_channels=1, n_params=5):  # n_params is now 5
        super().__init__()
        # A slightly deeper U-Net structure for better feature extraction
        self.inc = ComplexDoubleConv(n_channels, 64)
        self.down1 = ComplexDown(64, 128)
        self.down2 = ComplexDown(128, 256)
        self.down3 = ComplexDown(256, 512)
        self.up1 = ComplexUp(512, 256)
        self.up2 = ComplexUp(256, 128)
        self.up3 = ComplexUp(128, 64)
        self.final_conv = ComplexDoubleConv(64, 64)

        # The output layer is a standard 2D convolution that maps the
        # magnitude of the complex features to the 5 real-valued parameter maps.
        self.outc = nn.Conv2d(64, n_params, kernel_size=1)

    def forward(self, x):
        # U-Net Encoder-Decoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.final_conv(x)

        # Transition from complex feature space to real parameter space
        x_mag = x.abs()
        raw_params = self.outc(x_mag)

        # --- Apply appropriate activations to each parameter channel ---
        # 1. Heatmap: Sigmoid to get a probability-like map in [0, 1]
        pred_heatmap = torch.sigmoid(raw_params[:, 0:1, :, :])
        # 2. Amplitude (A): ReLU to ensure it's non-negative. We predict log(1+A).
        pred_A_log1p = torch.relu(raw_params[:, 1:2, :, :])
        # 3. Alpha (α): Tanh to constrain it to its physical range of [-1, 1]
        pred_alpha = torch.tanh(raw_params[:, 2:3, :, :])
        # 4 & 5. Offsets (dx, dy): Tanh to constrain them to [-1, 1] pixel range.
        pred_dx = torch.tanh(raw_params[:, 3:4, :, :])
        pred_dy = torch.tanh(raw_params[:, 4:5, :, :])

        # Concatenate the processed channels for the final output
        params_out = torch.cat([pred_heatmap, pred_A_log1p, pred_alpha, pred_dx, pred_dy], dim=1)
        return params_out
