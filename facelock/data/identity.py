"""Per-image identity classification dataset (ArcFace)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from facelock.data.index import FaceIndex
from facelock.data.pairs import AddGaussianNoise


CENTERED_NORMALIZE = {
  'mean': [0.5, 0.5, 0.5],
  'std': [0.5, 0.5, 0.5],
}


def buildArcFaceTrainTransform(imageSize: int, augSize: int) -> transforms.Compose:
  """Train augs with centered normalize to [-1, 1]."""
  return transforms.Compose(
    [
      transforms.Resize((augSize, augSize)),
      transforms.RandomResizedCrop(
        imageSize,
        scale=(0.7, 1.0),
        ratio=(0.9, 1.1),
      ),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.05,
      ),
      transforms.ToTensor(),
      AddGaussianNoise(std=0.03),
      transforms.Normalize(
        mean=CENTERED_NORMALIZE['mean'],
        std=CENTERED_NORMALIZE['std'],
      ),
    ],
  )


def buildArcFaceEvalTransform(imageSize: int) -> transforms.Compose:
  """Eval resize + centered normalize."""
  return transforms.Compose(
    [
      transforms.Resize((imageSize, imageSize)),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=CENTERED_NORMALIZE['mean'],
        std=CENTERED_NORMALIZE['std'],
      ),
    ],
  )


class IdentityImageDataset(Dataset):
  """One image + remapped class id."""

  def __init__(
    self,
    index: FaceIndex,
    idIndexes: np.ndarray,
    labelById: dict[int, int],
    transform: transforms.Compose,
  ) -> None:
    self.samples: list[tuple[str, int]] = []
    for idIndex in idIndexes.tolist():
      label = labelById[int(idIndex)]
      for path in index.pathsById[int(idIndex)]:
        self.samples.append((path, label))
    if not self.samples:
      raise RuntimeError('IdentityImageDataset got zero samples')
    self.transform = transform

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    path, label = self.samples[index]
    image = self.transform(Image.open(path).convert('RGB'))
    return image, torch.tensor(label, dtype=torch.long)
