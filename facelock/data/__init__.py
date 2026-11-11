"""Training data: identity folders → pairs or classification samples."""

from facelock.data.index import FaceIndex, splitByIdentity
from facelock.data.identity import IdentityImageDataset
from facelock.data.pairs import PairDataset

__all__ = [
  'FaceIndex',
  'IdentityImageDataset',
  'PairDataset',
  'splitByIdentity',
]
