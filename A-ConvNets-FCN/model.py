from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BN + ReLU with configurable kernel size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBranch(nn.Module):
    """Reduce channels then upsample to the target spatial size."""

    def __init__(self, in_channels: int, out_channels: int, target_size: Tuple[int, int]) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        return F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)


class AConvFCN(nn.Module):
    """A-ConvNets backbone with FCN-style multi-branch aggregation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 8, kernel_size)
        self.conv2 = ConvBlock(8, 16, kernel_size)
        self.conv3 = ConvBlock(16, 32, kernel_size)
        self.conv4 = ConvBlock(32, 64, kernel_size)
        self.conv5 = ConvBlock(64, 64, kernel_size)
        self.pool = nn.MaxPool2d(2)

        self.branch_deep = UpsampleBranch(64, 16, target_size=(128, 128))
        self.branch_mid = UpsampleBranch(64, 16, target_size=(128, 128))
        self.branch_shallow = UpsampleBranch(32, 16, target_size=(128, 128))

        self.fuse = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.conv1(x)
        x = self.pool(feat1)

        feat2 = self.conv2(x)
        x = self.pool(feat2)

        feat3 = self.conv3(x)
        x = self.pool(feat3)

        feat4 = self.conv4(x)
        x = self.pool(feat4)

        feat5 = self.conv5(x)
        deep = self.pool(feat5)

        up_deep = self.branch_deep(deep)
        up_mid = self.branch_mid(feat4)
        up_shallow = self.branch_shallow(feat3)

        fused = self.fuse(up_deep + up_mid + up_shallow)
        return self.output_conv(fused)