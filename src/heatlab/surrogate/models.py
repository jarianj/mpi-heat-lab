"""Fourier Neural Operator and U-Net that predict the temperature change over a fixed time."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_inputs(initial: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    batch, height, width = initial.shape
    y = (torch.arange(height, device=initial.device) + 0.5) / height
    x = (torch.arange(width, device=initial.device) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack(
        [
            initial,
            torch.log(alpha),
            grid_x.expand(batch, -1, -1),
            grid_y.expand(batch, -1, -1),
        ],
        dim=1,
    )


def conserve_energy(change: torch.Tensor) -> torch.Tensor:
    return change - change.mean(dim=(-2, -1), keepdim=True)


class SpectralConv(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        shape = (2, width, width, modes, modes)
        self.weights = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros_like(spectrum)
        m = self.modes
        out[..., :m, :m] = torch.einsum("bixy,ioxy->boxy", spectrum[..., :m, :m], self.weights[0])
        out[..., -m:, :m] = torch.einsum("bixy,ioxy->boxy", spectrum[..., -m:, :m], self.weights[1])
        return torch.fft.irfft2(out, s=(height, width), norm="ortho")


class FNO(nn.Module):
    def __init__(self, width: int = 32, modes: int = 12, layers: int = 4, padding: int = 8):
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv2d(4, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, initial: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        h = self.lift(build_inputs(initial, alpha))
        h = F.pad(h, (0, self.padding, 0, self.padding))
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local, strict=True)):
            h = spectral(h) + local(h)
            if index < len(self.spectral) - 1:
                h = F.gelu(h)
        h = h[..., : -self.padding, : -self.padding]
        return self.project(h).squeeze(1)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class UNet(nn.Module):
    def __init__(self, base: int = 32, levels: int = 3):
        super().__init__()
        channels = [base * 2**i for i in range(levels + 1)]
        self.down = nn.ModuleList()
        previous = 4
        for count in channels[:-1]:
            self.down.append(ConvBlock(previous, count))
            previous = count
        self.bottom = ConvBlock(channels[-2], channels[-1])
        self.up = nn.ModuleList(
            ConvBlock(high + low, low)
            for high, low in zip(channels[:0:-1], channels[-2::-1], strict=True)
        )
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, initial: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        h = build_inputs(initial, alpha)
        skips = []
        for block in self.down:
            h = block(h)
            skips.append(h)
            h = F.avg_pool2d(h, 2)
        h = self.bottom(h)
        for block, skip in zip(self.up, reversed(skips), strict=True):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1))
        return self.head(h).squeeze(1)


MODELS = {"fno": FNO, "unet": UNet}


def count_parameters(model: nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * (2 if parameter.is_complex() else 1)
    return total
