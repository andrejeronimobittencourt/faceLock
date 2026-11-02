"""Training data: identity folders → match/not-match pairs."""

from facelock.data.index import FaceIndex, splitByIdentity
from facelock.data.pairs import PairDataset

__all__ = ['FaceIndex', 'PairDataset', 'splitByIdentity']
