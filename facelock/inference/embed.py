"""Load a trained encoder and embed face images / BGR crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from facelock.models import FaceEncoder


def buildTransform(imageSize: int) -> transforms.Compose:
  """Match training eval preprocessing: resize → tensor in [0, 1]."""
  return transforms.Compose(
    [
      transforms.Resize((imageSize, imageSize)),
      transforms.ToTensor(),
    ],
  )


def loadEncoder(
  encoderPath: Path,
  device: torch.device,
) -> tuple[FaceEncoder, dict]:
  """Rebuild FaceEncoder from an exported checkpoint."""
  checkpoint = torch.load(encoderPath, map_location=device, weights_only=False)
  embedDim = int(checkpoint['latentDim'])
  encoder = FaceEncoder(embedDim=embedDim).to(device)
  encoder.load_state_dict(checkpoint['encoderState'])
  encoder.eval()
  return encoder, checkpoint


@torch.no_grad()
def embed(encoder: FaceEncoder, imageTensor: torch.Tensor) -> torch.Tensor:
  """L2-normalized embedding suitable for cosine similarity."""
  return F.normalize(encoder(imageTensor), p=2, dim=1)


def imagePathToTensor(
  path: Path,
  imageSize: int,
  device: torch.device,
) -> torch.Tensor:
  """Load one RGB image as a batched encoder input."""
  image = Image.open(path).convert('RGB')
  return buildTransform(imageSize)(image).unsqueeze(0).to(device)


def bgrCropToTensor(
  faceBgr: np.ndarray,
  imageSize: int,
  device: torch.device,
) -> torch.Tensor:
  """Convert an OpenCV BGR face crop into a batched encoder input."""
  if faceBgr.size == 0:
    raise ValueError('Face crop is empty')
  rgb = faceBgr[:, :, ::-1]
  image = Image.fromarray(rgb)
  return buildTransform(imageSize)(image).unsqueeze(0).to(device)


def embedImagePath(
  encoderPath: Path,
  imagePath: Path,
  device: torch.device,
) -> np.ndarray:
  """Embed one image file; return float32 L2-normalized vector."""
  encoder, checkpoint = loadEncoder(encoderPath, device)
  imageSize = int(checkpoint.get('imageSize', 112))
  tensor = imagePathToTensor(imagePath, imageSize, device)
  return embed(encoder, tensor)[0].detach().cpu().numpy().astype(np.float32)
