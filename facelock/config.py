"""Shared paths and defaults for the face-lock pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model / image geometry (pair-match defaults)
IMAGE_SIZE = 112
AUG_SIZE = 144
EMBED_DIM = 128

# ArcFace / IResNet defaults
ARCFACE_EMBED_DIM = 512
ARCFACE_AUG_SIZE = 128
ARCFACE_BACKBONE = 'ir50'
ARCFACE_SCALE = 64.0
ARCFACE_MARGIN = 0.50

# Gallery match
MATCH_THRESHOLD = 0.65
VERIFY_THRESHOLD = 0.50
VERIFY_MISSES = 2
CONFIRM_HITS = 2
LOST_GRACE_FRAMES = 20
SESSION_MEMORY_FRAMES = 90

# Access levels: user < operator < admin
ACCESS_USER = 'user'
ACCESS_OPERATOR = 'operator'
ACCESS_ADMIN = 'admin'


@dataclass(frozen=True)
class Paths:
  """Default filesystem paths (relative to the process cwd / repo root)."""

  dataDir: Path = Path('data') / 'identities'
  modelsDir: Path = Path('models')
  encoderBest: Path = Path('models') / 'encoder_best.pt'
  galleryDb: Path = Path('gallery') / 'gallery.db'
  galleryFaces: Path = Path('gallery') / 'faces'
  cascade: Path = PROJECT_ROOT / 'assets' / 'haarcascade_frontalface_default.xml'
  assetsDir: Path = PROJECT_ROOT / 'assets'


@dataclass(frozen=True)
class TrainConfig:
  """Identity match/not-match training defaults."""

  imageSize: int = IMAGE_SIZE
  augSize: int = AUG_SIZE
  embedDim: int = EMBED_DIM
  batchSize: int = 64
  epochs: int = 20
  learningRate: float = 1e-3
  weightDecay: float = 5e-4
  numWorkers: int = 4
  seed: int = 42
  valImages: int = 10_000
  saveEvery: int = 5


@dataclass(frozen=True)
class ArcFaceTrainConfig:
  """IResNet + ArcFace training defaults."""

  backbone: str = ARCFACE_BACKBONE
  imageSize: int = IMAGE_SIZE
  augSize: int = ARCFACE_AUG_SIZE
  embedDim: int = ARCFACE_EMBED_DIM
  batchSize: int = 64
  epochs: int = 20
  learningRate: float = 0.1
  weightDecay: float = 5e-4
  momentum: float = 0.9
  warmupEpochs: int = 1
  gradClip: float = 5.0
  numWorkers: int = 4
  seed: int = 42
  valImages: int = 10_000
  saveEvery: int = 5
  arcScale: float = ARCFACE_SCALE
  arcMargin: float = ARCFACE_MARGIN
  dropout: float = 0.0
