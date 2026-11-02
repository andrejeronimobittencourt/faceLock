"""On-the-fly MATCH / NOT MATCH image pairs."""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from facelock.data.index import FaceIndex


class AddGaussianNoise:
  """Add small Gaussian noise to a float image tensor in [0, 1]."""

  def __init__(self, std: float = 0.04) -> None:
    self.std = std

  def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
    if self.std <= 0:
      return tensor
    return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def buildTrainTransform(imageSize: int, augSize: int) -> transforms.Compose:
  """Augment: resize to augSize, then crop/affine into imageSize."""
  return transforms.Compose(
    [
      transforms.Resize((augSize, augSize)),
      transforms.RandomResizedCrop(
        imageSize,
        scale=(0.65, 1.0),
        ratio=(0.85, 1.15),
      ),
      transforms.RandomHorizontalFlip(),
      transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        fill=0,
      ),
      transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05,
      ),
      transforms.ToTensor(),
      AddGaussianNoise(std=0.04),
    ],
  )


def buildEvalTransform(imageSize: int) -> transforms.Compose:
  """Plain resize used at validation and inference."""
  return transforms.Compose(
    [
      transforms.Resize((imageSize, imageSize)),
      transforms.ToTensor(),
    ],
  )


class PairDataset(Dataset):
  """Lazy pairs from anchor positions; partner chosen in __getitem__."""

  def __init__(
    self,
    index: FaceIndex,
    anchorIndexes,
    allowedIdIndexes,
    imageSize: int,
    pairSeed: int,
    augment: bool,
    augSize: int = 144,
  ) -> None:
    self.index = index
    self.anchorIndexes = anchorIndexes
    self.allowedIdIndexes = allowedIdIndexes
    if len(self.allowedIdIndexes) < 2:
      raise ValueError('Need at least 2 identities for different-person pairs')
    self.pairSeed = pairSeed
    self.transform = (
      buildTrainTransform(imageSize, augSize)
      if augment
      else buildEvalTransform(imageSize)
    )

  def __len__(self) -> int:
    return int(len(self.anchorIndexes))

  def _load(self, path: str) -> torch.Tensor:
    return self.transform(Image.open(path).convert('RGB'))

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = int(self.anchorIndexes[index])
    idIndex = int(self.index.anchorId[anchor])
    localIndex = int(self.index.anchorLocal[anchor])
    paths = self.index.pathsById[idIndex]
    pathA = paths[localIndex]

    rng = random.Random((self.pairSeed << 20) ^ anchor)
    canSame = len(paths) >= 2
    wantSame = (anchor % 2 == 0) and canSame

    if wantSame:
      localB = localIndex
      while localB == localIndex:
        localB = rng.randrange(len(paths))
      pathB = paths[localB]
      label = 1.0
    else:
      idB = idIndex
      while idB == idIndex:
        idB = int(self.allowedIdIndexes[rng.randrange(len(self.allowedIdIndexes))])
      pathB = self.index.pathsById[idB][rng.randrange(len(self.index.pathsById[idB]))]
      label = 0.0

    return (
      self._load(pathA),
      self._load(pathB),
      torch.tensor(label, dtype=torch.float32),
    )
