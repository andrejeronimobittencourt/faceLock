"""Encoder checkpoint export and rebuild."""


from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from facelock.models.encoder import FaceEncoder
from facelock.models.iresnet import BACKBONE_BUILDERS, buildBackbone


def exportEncoder(
  encoder: nn.Module,
  path: Path,
  embedDim: int,
  imageSize: int,
  extraMeta: dict[str, Any] | None = None,
) -> None:
  """Save encoder weights and metadata."""

  path.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, Any] = {
    'encoderState': encoder.state_dict(),
    'latentDim': embedDim,
    'imageSize': imageSize,
  }
  if extraMeta:
    payload.update(extraMeta)
  torch.save(payload, path)


def buildEncoderFromCheckpoint(checkpoint: dict[str, Any]) -> nn.Module:
  """Rebuild the encoder described by an exported checkpoint."""
  embedDim = int(checkpoint['latentDim'])
  backbone = checkpoint.get('backbone')
  pipeline = checkpoint.get('pipeline', 'pair')
  if backbone:
    if backbone not in BACKBONE_BUILDERS:
      raise ValueError(f'Unknown backbone in checkpoint: {backbone!r}')
    return buildBackbone(str(backbone), numFeatures=embedDim)
  if pipeline == 'arcface':
    return buildBackbone('ir50', numFeatures=embedDim)
  return FaceEncoder(embedDim=embedDim)
