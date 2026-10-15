"""Compact 2D U-Net with configurable width."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net whose channel widths are base_channels * (1, 2, 4, ...)."""

    def __init__(self, in_channels=1, n_classes=4, base_channels=16, depth=4):
        super().__init__()
        chs = [base_channels * 2**i for i in range(depth + 1)]  # e.g. 16, 32, 64, 128, 256

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.encoders.append(ConvBlock(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-2], chs[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(c * 2, c))  # c (upsampled) + c (skip)

        self.head = nn.Conv2d(chs[0], n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)  # saved for the decoder
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)  # raw scores (logits), one map per class


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)