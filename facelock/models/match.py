"""Shared encoder + MATCH / NOT MATCH training head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from facelock.models.encoder import FaceEncoder


class MatchHead(nn.Module):
  """Binary match head on a pair of embeddings (training signal only)."""

  def __init__(self, embedDim: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(embedDim * 3, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 1),
    )

  def forward(self, embedA: torch.Tensor, embedB: torch.Tensor) -> torch.Tensor:
    """Return logits [N] for BCEWithLogits (1=match, 0=not)."""
    embedA = F.normalize(embedA, p=2, dim=1)
    embedB = F.normalize(embedB, p=2, dim=1)
    features = torch.cat([embedA, embedB, torch.abs(embedA - embedB)], dim=1)
    return self.net(features).squeeze(1)


class MatchModel(nn.Module):
  """Shared FaceEncoder + match head used during training."""

  def __init__(self, embedDim: int = 128) -> None:
    super().__init__()
    self.encoder = FaceEncoder(embedDim=embedDim)
    self.head = MatchHead(embedDim)

  def forward(
    self,
    imageA: torch.Tensor,
    imageB: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedA = self.encoder(imageA)
    embedB = self.encoder(imageB)
    logits = self.head(embedA, embedB)
    return logits, embedA, embedB
