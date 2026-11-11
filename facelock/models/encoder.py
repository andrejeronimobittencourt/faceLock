"""Convolutional face encoder → fixed-size embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
  """Conv → BatchNorm → ReLU."""

  def __init__(self, inChannels: int, outChannels: int, stride: int = 1) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(
        inChannels,
        outChannels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
      ),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class FaceEncoder(nn.Module):
  """Maps an RGB face crop to a latent embedding vector."""

  def __init__(self, embedDim: int = 128) -> None:
    super().__init__()
    self.features = nn.Sequential(
      ConvBlock(3, 32, stride=2),
      ConvBlock(32, 64, stride=2),
      ConvBlock(64, 128, stride=2),
      ConvBlock(128, 256, stride=2),
      ConvBlock(256, 512, stride=2),
    )
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, embedDim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.pool(x).flatten(1)
    return self.fc(x)

  def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Return L2-normalized embedding."""
    z = self.forward(x)
    if normalize:
      z = F.normalize(z, p=2, dim=1)
    return z
