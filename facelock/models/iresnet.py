"""IResNet face encoders (IR-18 / IR-34 / IR-50 / IR-100)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
  """Pre-norm residual unit with PReLU (face IResNet block)."""

  def __init__(
    self,
    channelsIn: int,
    channelsOut: int,
    stride: int = 1,
  ) -> None:
    super().__init__()
    self.normIn = nn.BatchNorm2d(channelsIn, eps=1e-5)
    self.expand = nn.Conv2d(channelsIn, channelsOut, kernel_size=3, padding=1, bias=False)
    self.normMid = nn.BatchNorm2d(channelsOut, eps=1e-5)
    self.act = nn.PReLU(channelsOut)
    self.project = nn.Conv2d(
      channelsOut,
      channelsOut,
      kernel_size=3,
      stride=stride,
      padding=1,
      bias=False,
    )
    self.normOut = nn.BatchNorm2d(channelsOut, eps=1e-5)
    if stride != 1 or channelsIn != channelsOut:
      self.skip = nn.Sequential(
        nn.Conv2d(channelsIn, channelsOut, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(channelsOut, eps=1e-5),
      )
    else:
      self.skip = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x if self.skip is None else self.skip(x)
    y = self.normIn(x)
    y = self.expand(y)
    y = self.normMid(y)
    y = self.act(y)
    y = self.project(y)
    y = self.normOut(y)
    return y + residual


class IResNet(nn.Module):
  """112×112 RGB face crop → fixed-size embedding."""

  def __init__(
    self,
    stageDepths: tuple[int, int, int, int],
    embedDim: int = 512,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.stem = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(64, eps=1e-5),
      nn.PReLU(64),
    )
    self.stage1 = self._buildStage(64, 64, stageDepths[0], stride=2)
    self.stage2 = self._buildStage(64, 128, stageDepths[1], stride=2)
    self.stage3 = self._buildStage(128, 256, stageDepths[2], stride=2)
    self.stage4 = self._buildStage(256, 512, stageDepths[3], stride=2)
    self.tailNorm = nn.BatchNorm2d(512, eps=1e-5)
    self.drop = nn.Dropout(p=dropout, inplace=True)
    # 112 → /16 spatial → 7×7 before flatten
    self.embed = nn.Linear(512 * 7 * 7, embedDim)
    self.embedNorm = nn.BatchNorm1d(embedDim, eps=1e-5)
    nn.init.constant_(self.embedNorm.weight, 1.0)
    self.embedNorm.weight.requires_grad = False
    self._initWeights()

  def _buildStage(
    self,
    channelsIn: int,
    channelsOut: int,
    depth: int,
    stride: int,
  ) -> nn.Sequential:
    units = [ResidualUnit(channelsIn, channelsOut, stride=stride)]
    for _ in range(1, depth):
      units.append(ResidualUnit(channelsOut, channelsOut, stride=1))
    return nn.Sequential(*units)

  def _initWeights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.1)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.tailNorm(x)
    x = torch.flatten(x, 1)
    x = self.drop(x)
    x = self.embed(x)
    return self.embedNorm(x)


_STAGE_DEPTHS: dict[str, tuple[int, int, int, int]] = {
  'ir18': (2, 2, 2, 2),
  'ir34': (3, 4, 6, 3),
  'ir50': (3, 4, 14, 3),
  'ir100': (3, 13, 30, 3),
}

BACKBONE_NAMES = tuple(_STAGE_DEPTHS.keys())


def buildBackbone(name: str, numFeatures: int = 512, dropout: float = 0.0) -> IResNet:
  """Build an IResNet by short name (ir18 / ir34 / ir50 / ir100)."""
  key = name.lower().strip()
  if key not in _STAGE_DEPTHS:
    raise ValueError(f'Unknown backbone {name!r}. Choose from {list(BACKBONE_NAMES)}')
  return IResNet(
    stageDepths=_STAGE_DEPTHS[key],
    embedDim=numFeatures,
    dropout=dropout,
  )


def iresnet18(numFeatures: int = 512, dropout: float = 0.0) -> IResNet:
  """IResNet-18 face encoder."""
  return buildBackbone('ir18', numFeatures=numFeatures, dropout=dropout)


def iresnet34(numFeatures: int = 512, dropout: float = 0.0) -> IResNet:
  """IResNet-34 face encoder."""
  return buildBackbone('ir34', numFeatures=numFeatures, dropout=dropout)


def iresnet50(numFeatures: int = 512, dropout: float = 0.0) -> IResNet:
  """IResNet-50 face encoder."""
  return buildBackbone('ir50', numFeatures=numFeatures, dropout=dropout)


def iresnet100(numFeatures: int = 512, dropout: float = 0.0) -> IResNet:
  """IResNet-100 face encoder."""
  return buildBackbone('ir100', numFeatures=numFeatures, dropout=dropout)


BACKBONE_BUILDERS = {
  'ir18': iresnet18,
  'ir34': iresnet34,
  'ir50': iresnet50,
  'ir100': iresnet100,
}
