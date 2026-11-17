"""YOLO face detection."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from facelock.config import Paths

DEFAULT_FACE_MODEL = Paths().faceModel
FACE_MODEL_URL = (
  'https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt'
)


def ensureFaceModel(path: Path) -> Path:
  """Download face weights to ``path`` when the file is missing."""
  path = Path(path)
  if path.is_file():
    return path
  path.parent.mkdir(parents=True, exist_ok=True)
  print(f'Downloading YOLO face weights → {path}')
  tmpPath = path.with_suffix(f'{path.suffix}.partial')
  try:
    urllib.request.urlretrieve(FACE_MODEL_URL, tmpPath)
    tmpPath.replace(path)
  except Exception:
    if tmpPath.is_file():
      tmpPath.unlink()
    raise
  return path


class FaceDetector:
  """Detects faces in BGR frames with Ultralytics YOLO-face."""

  def __init__(
    self,
    modelPath: Path | None = None,
    minSize: tuple[int, int] = (1, 1),
    minDetectionConfidence: float = 0.25,
    device: str | None = None,
  ) -> None:
    try:
      from ultralytics import YOLO
    except ImportError as exc:
      raise ImportError(
        'ultralytics is required for face detection. Install with: '
        "pip install 'ultralytics>=8.3.0'"
      ) from exc

    path = ensureFaceModel(Path(modelPath) if modelPath is not None else DEFAULT_FACE_MODEL)
    self._minSize = (int(minSize[0]), int(minSize[1]))
    self._confidence = float(minDetectionConfidence)
    self._device = device
    self._model = YOLO(str(path))

  def detect(self, frame: np.ndarray) -> np.ndarray:
    """Return (N, 4) boxes as (x, y, width, height)."""
    if frame is None or frame.size == 0 or self._model is None:
      return np.zeros((0, 4), dtype=np.int32)

    frameH, frameW = frame.shape[:2]
    results = self._model.predict(
      source=frame,
      conf=self._confidence,
      verbose=False,
      device=self._device,
    )
    resultBoxes = results[0].boxes if results else None
    if resultBoxes is None or len(resultBoxes) == 0:
      return np.zeros((0, 4), dtype=np.int32)

    minW, minH = self._minSize
    boxesOut: list[tuple[int, int, int, int]] = []
    for x1f, y1f, x2f, y2f in resultBoxes.xyxy.detach().cpu().numpy():
      x1 = max(0, min(frameW - 1, int(round(float(x1f)))))
      y1 = max(0, min(frameH - 1, int(round(float(y1f)))))
      x2 = max(0, min(frameW, int(round(float(x2f)))))
      y2 = max(0, min(frameH, int(round(float(y2f)))))
      width = x2 - x1
      height = y2 - y1
      if width < minW or height < minH:
        continue
      boxesOut.append((x1, y1, width, height))

    if not boxesOut:
      return np.zeros((0, 4), dtype=np.int32)
    return np.asarray(boxesOut, dtype=np.int32)

  def close(self) -> None:
    """Release detector resources."""
    self._model = None


def expandBox(
  x: int,
  y: int,
  width: int,
  height: int,
  frameWidth: int,
  frameHeight: int,
  margin: float = 0.0,
) -> tuple[int, int, int, int]:
  """Expand a face box by margin fraction and clamp to frame bounds."""
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
