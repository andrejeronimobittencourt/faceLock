"""Device and RNG helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolveDevice(choice: str = 'auto') -> torch.device:
  """Pick CUDA when available unless forced otherwise."""
  if choice == 'cpu':
    return torch.device('cpu')
  if choice == 'cuda':
    if not torch.cuda.is_available():
      raise RuntimeError('CUDA requested but unavailable')
    return torch.device('cuda')
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setSeed(seed: int) -> None:
  """Seed Python, NumPy, and Torch RNGs."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
