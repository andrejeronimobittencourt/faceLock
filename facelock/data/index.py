"""Index identity folders on disk (one folder per person)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


class FaceIndex:
  """Lazy folder index — image paths stay on disk."""

  def __init__(self, root: Path) -> None:
    self.idNames: list[str] = []
    self.pathsById: list[list[str]] = []

    folders = sorted([path for path in root.iterdir() if path.is_dir()])
    if not folders:
      raise FileNotFoundError(f'No identity folders under {root.resolve()}')

    anchorId: list[int] = []
    anchorLocal: list[int] = []

    for folder in folders:
      images = sorted(
        [
          str(path)
          for path in folder.iterdir()
          if path.suffix.lower() in IMAGE_SUFFIXES
        ],
      )
      if not images:
        continue
      idIndex = len(self.idNames)
      self.idNames.append(folder.name)
      self.pathsById.append(images)
      for localIndex, _ in enumerate(images):
        anchorId.append(idIndex)
        anchorLocal.append(localIndex)

    if len(self.idNames) < 2:
      raise RuntimeError('Need at least 2 identities with images')

    self.anchorId = np.asarray(anchorId, dtype=np.int32)
    self.anchorLocal = np.asarray(anchorLocal, dtype=np.int32)
    self.numImages = int(self.anchorId.shape[0])
    self.numIdentities = len(self.idNames)

    multi = [i for i, paths in enumerate(self.pathsById) if len(paths) >= 2]
    if not multi:
      raise RuntimeError('Need identities with at least 2 images each for match pairs')
    self.multiIdIndexes = np.asarray(multi, dtype=np.int32)


def splitByIdentity(
  index: FaceIndex,
  valImages: int,
  seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Hold out whole people for validation (no shared identities)."""
  rng = np.random.default_rng(seed)
  idOrder = rng.permutation(index.numIdentities).astype(np.int32)

  valIds: list[int] = []
  valImageCount = 0
  targetValImages = min(max(valImages, 1), max(1, index.numImages // 20))

  for idIndex in idOrder:
    if len(valIds) >= 2 and valImageCount >= targetValImages:
      break
    valIds.append(int(idIndex))
    valImageCount += len(index.pathsById[int(idIndex)])

  valIdSet = set(valIds)
  trainIds = [i for i in range(index.numIdentities) if i not in valIdSet]
  if len(trainIds) < 2:
    raise RuntimeError('Train split needs at least 2 identities')

  trainIdIndexes = np.asarray(trainIds, dtype=np.int32)
  valIdIndexes = np.asarray(valIds, dtype=np.int32)
  trainAnchors = np.flatnonzero(np.isin(index.anchorId, trainIdIndexes)).astype(np.int32)
  valAnchors = np.flatnonzero(np.isin(index.anchorId, valIdIndexes)).astype(np.int32)
  return trainAnchors, valAnchors, trainIdIndexes, valIdIndexes
