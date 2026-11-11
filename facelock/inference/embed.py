"""Load a trained encoder and embed face images / BGR crops."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from facelock.models.checkpoint import buildEncoderFromCheckpoint


def buildTransform(
  imageSize: int,
  mean: Sequence[float] | None = None,
  std: Sequence[float] | None = None,
) -> transforms.Compose:
  """Resize → tensor; optional mean/std normalize."""
  steps: list = [
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor(),
  ]
  if mean is not None and std is not None:
    steps.append(transforms.Normalize(mean=list(mean), std=list(std)))
  return transforms.Compose(steps)


def loadEncoder(
  encoderPath: Path,
  device: torch.device,
) -> tuple[nn.Module, dict]:
  """Rebuild encoder from an exported checkpoint."""
  checkpoint = torch.load(encoderPath, map_location=device, weights_only=False)
  encoder = buildEncoderFromCheckpoint(checkpoint).to(device)
  encoder.load_state_dict(checkpoint['encoderState'])
  encoder.eval()
  return encoder, checkpoint


def transformFromCheckpoint(checkpoint: dict, imageSize: int) -> transforms.Compose:
  """Build preprocess from checkpoint metadata."""
  normalize = checkpoint.get('inputNormalize')
  if isinstance(normalize, dict) and 'mean' in normalize and 'std' in normalize:
    return buildTransform(imageSize, mean=normalize['mean'], std=normalize['std'])
  return buildTransform(imageSize)


@torch.no_grad()
def embed(encoder: nn.Module, imageTensor: torch.Tensor) -> torch.Tensor:
  """Return L2-normalized embedding."""
  return F.normalize(encoder(imageTensor), p=2, dim=1)


def imagePathToTensor(
  path: Path,
  imageSize: int,
  device: torch.device,
  transform: transforms.Compose | None = None,
) -> torch.Tensor:
  """Load one RGB image as a batched encoder input."""
  image = Image.open(path).convert('RGB')
  tf = transform or buildTransform(imageSize)
  return tf(image).unsqueeze(0).to(device)


def bgrCropToTensor(
  faceBgr: np.ndarray,
  imageSize: int,
  device: torch.device,
  transform: transforms.Compose | None = None,
) -> torch.Tensor:
  """Convert an OpenCV BGR face crop into a batched encoder input."""
  if faceBgr.size == 0:
    raise ValueError('Face crop is empty')
  rgb = faceBgr[:, :, ::-1]
  image = Image.fromarray(rgb)
  tf = transform or buildTransform(imageSize)
  return tf(image).unsqueeze(0).to(device)


def embedImagePath(
  encoderPath: Path,
  imagePath: Path,
  device: torch.device,
) -> np.ndarray:
  """Embed one image file; return float32 L2-normalized vector."""
  encoder, checkpoint = loadEncoder(encoderPath, device)
  imageSize = int(checkpoint.get('imageSize', 112))
  transform = transformFromCheckpoint(checkpoint, imageSize)
  tensor = imagePathToTensor(imagePath, imageSize, device, transform=transform)
  return embed(encoder, tensor)[0].detach().cpu().numpy().astype(np.float32)
