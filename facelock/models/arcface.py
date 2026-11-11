"""ArcFace margin head + IResNet classification model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from facelock.models.iresnet import buildBackbone


class ArcMarginProduct(nn.Module):
  """Additive angular margin on the unit sphere (ArcFace)."""

  def __init__(
    self,
    inFeatures: int,
    outFeatures: int,
    scale: float = 64.0,
    margin: float = 0.50,
  ) -> None:
    super().__init__()
    self.inFeatures = inFeatures
    self.outFeatures = outFeatures
    self.scale = float(scale)
    self.margin = float(margin)
    self.weight = nn.Parameter(torch.empty(outFeatures, inFeatures))
    nn.init.xavier_uniform_(self.weight)
    self.cosMargin = math.cos(margin)
    self.sinMargin = math.sin(margin)
    self.threshold = math.cos(math.pi - margin)
    self.marginMm = math.sin(math.pi - margin) * margin

  def forward(self, embed: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return scaled logits with angular margin on the true class."""
    cosine = F.linear(F.normalize(embed), F.normalize(self.weight))
    if self.margin <= 0.0:
      return cosine * self.scale
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
    phi = cosine * self.cosMargin - sine * self.sinMargin
    phi = torch.where(cosine > self.threshold, phi, cosine - self.marginMm)
    oneHot = torch.zeros_like(cosine)
    oneHot.scatter_(1, labels.view(-1, 1), 1.0)
    return (oneHot * phi + (1.0 - oneHot) * cosine) * self.scale


class ArcFaceModel(nn.Module):
  """IResNet encoder + ArcFace margin head."""


  def __init__(
    self,
    numClasses: int,
    backbone: str = 'ir50',
    embedDim: int = 512,
    scale: float = 64.0,
    margin: float = 0.50,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.backboneName = backbone
    self.encoder = buildBackbone(backbone, numFeatures=embedDim, dropout=dropout)
    self.margin = ArcMarginProduct(embedDim, numClasses, scale=scale, margin=margin)

  def forward(
    self,
    images: torch.Tensor,
    labels: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    embed = self.encoder(images)
    logits = self.margin(embed, labels)
    return logits, embed
