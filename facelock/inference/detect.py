"""OpenCV Haar cascade face detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facelock.config import Paths

DEFAULT_CASCADE = Paths().cascade


class FaceDetector:
  """Detects frontal faces in BGR video frames."""

  def __init__(
    self,
    cascadePath: Path | None = None,
    scaleFactor: float = 1.3,
    minNeighbors: int = 5,
    minSize: tuple[int, int] = (60, 60),
  ) -> None:
    path = Path(cascadePath) if cascadePath is not None else DEFAULT_CASCADE
    self._cascade = cv2.CascadeClassifier(str(path))
    if self._cascade.empty():
      raise FileNotFoundError(f'Failed to load face cascade: {path}')
    self._scaleFactor = scaleFactor
    self._minNeighbors = minNeighbors
    self._minSize = minSize

  def detect(self, frame: np.ndarray) -> np.ndarray:
    """Return bounding boxes as (x, y, width, height) arrays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return self._cascade.detectMultiScale(
      gray,
      scaleFactor=self._scaleFactor,
      minNeighbors=self._minNeighbors,
      minSize=self._minSize,
    )


def expandBox(
  x: int,
  y: int,
  width: int,
  height: int,
  frameWidth: int,
  frameHeight: int,
  margin: float = 0.25,
) -> tuple[int, int, int, int]:
  """Expand a Haar box by margin fraction and clamp to frame bounds."""
  padX = int(width * margin)
  padY = int(height * margin)
  x1 = max(0, x - padX)
  y1 = max(0, y - padY)
  x2 = min(frameWidth, x + width + padX)
  y2 = min(frameHeight, y + height + padY)
  return x1, y1, x2 - x1, y2 - y1


def cropBgr(
  frame: np.ndarray,
  x: int,
  y: int,
  width: int,
  height: int,
) -> np.ndarray:
  """Crop a BGR face region from a frame."""
  return frame[y:y + height, x:x + width].copy()
