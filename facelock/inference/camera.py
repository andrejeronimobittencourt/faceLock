"""Live webcam: detect → track → identify → HUD."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from facelock.config import (
  CONFIRM_HITS,
  LOST_GRACE_FRAMES,
  MATCH_THRESHOLD,
  SESSION_MEMORY_FRAMES,
  VERIFY_MISSES,
  VERIFY_THRESHOLD,
  Paths,
)
from facelock.device import resolveDevice
from facelock.gallery import FaceGallery, parseAccessLevel
from facelock.inference.detect import FaceDetector, cropBgr, expandBox
from facelock.inference.embed import (
  bgrCropToTensor,
  embed,
  loadEncoder,
  transformFromCheckpoint,
)
from facelock.inference.track import (
  FaceTrack,
  appearanceStillMatches,
  associateBoxes,
  normalizeEmbedding,
  pruneSessionMemory,
  retireTrack,
  trackLabel,
  tryIdentifyTrack,
)
from facelock.viz import drawFaceHud


def loadReferenceBgr(imagePath: str) -> np.ndarray | None:
  """Load a gallery reference face as BGR."""
  path = Path(imagePath)
  if not path.exists():
    return None
  return cv2.imread(str(path), cv2.IMREAD_COLOR)


def runCamera(
  cameraIndex: int = 0,
  encoderPath: Path | None = None,
  dbPath: Path | None = None,
  facesDir: Path | None = None,
  cascadePath: Path | None = None,
  required: str = 'admin',
  matchThreshold: float = MATCH_THRESHOLD,
  verifyThreshold: float = VERIFY_THRESHOLD,
  verifyMisses: int = VERIFY_MISSES,
  confirmHits: int = CONFIRM_HITS,
  lostGraceFrames: int = LOST_GRACE_FRAMES,
  sessionMemoryFrames: int = SESSION_MEMORY_FRAMES,
  deviceChoice: str = 'auto',
  inferEvery: int = 2,
  cropMargin: float = 0.0,
  mirror: bool = True,
) -> None:
  """Open the webcam and run recognition with the JARVIS HUD."""
  paths = Paths()
  encoderPath = Path(encoderPath or paths.encoderBest)
  dbPath = Path(dbPath or paths.galleryDb)
  facesDir = Path(facesDir or paths.galleryFaces)
  cascadePath = Path(cascadePath or paths.cascade)
  requiredLevel = parseAccessLevel(required)

  if not encoderPath.exists():
    raise FileNotFoundError(f'Encoder not found: {encoderPath}')

  device = resolveDevice(deviceChoice)
  encoder, checkpoint = loadEncoder(encoderPath, device)
  imageSize = int(checkpoint.get('imageSize', 112))
  preprocess = transformFromCheckpoint(checkpoint, imageSize)

  detector = FaceDetector(cascadePath)
  gallery = FaceGallery(dbPath, facesDir)
  if not gallery.listIdentities():
    print('Warning: gallery is empty. Register faces with gallery_cli.py first.')

  capture = cv2.VideoCapture(cameraIndex)
  if not capture.isOpened():
    gallery.close()
    raise RuntimeError(f'Could not open camera at index {cameraIndex}')

  windowTitle = 'FaceLock'
  cv2.namedWindow(windowTitle, cv2.WINDOW_NORMAL)

  tracks: list[FaceTrack] = []
  sessionMemory = []
  nextTrackId = 1
  frameIndex = 0
  startedAt = time.perf_counter()
  frameW = frameH = 0

  def embedBox(x: int, y: int, width: int, height: int) -> np.ndarray | None:
    ex, ey, ew, eh = expandBox(x, y, width, height, frameW, frameH, margin=cropMargin)
    face = cropBgr(frame, ex, ey, ew, eh)
    if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
      return None
    tensor = bgrCropToTensor(face, imageSize, device, transform=preprocess)
    vector = embed(encoder, tensor)[0].detach().cpu().numpy().astype(np.float32)
    return normalizeEmbedding(vector)

  try:
    while True:
      success, frame = capture.read()
      if not success:
        break
      if mirror:
        frame = cv2.flip(frame, 1)
      animT = time.perf_counter() - startedAt
      frameH, frameW = frame.shape[:2]

      boxes = [
        (int(x), int(y), int(w), int(h))
        for (x, y, w, h) in detector.detect(frame)
      ]
      shouldInfer = frameIndex % max(1, inferEvery) == 0

      matched, newBoxes, lostTracks = associateBoxes(tracks, boxes)
      active: list[FaceTrack] = []
      hijackedBoxes: list[tuple[int, int, int, int]] = []

      for track, (x, y, width, height) in matched:
        if shouldInfer and track.identity is not None:
          embedding = embedBox(x, y, width, height)
          if embedding is not None and not appearanceStillMatches(
            track,
            embedding,
            verifyThreshold,
            verifyMisses,
          ):
            retireTrack(track, sessionMemory, frameIndex)
            hijackedBoxes.append((x, y, width, height))
            continue
          if embedding is not None:
            track.lastEmbedding = embedding

        track.x, track.y, track.width, track.height = x, y, width, height
        track.lastSeenFrame = frameIndex
        track.lostFrames = 0
        active.append(track)

      for track in lostTracks:
        track.lostFrames += 1
        if track.lostFrames <= lostGraceFrames:
          active.append(track)
        else:
          retireTrack(track, sessionMemory, frameIndex)

      for (x, y, width, height) in list(newBoxes) + hijackedBoxes:
        active.append(
          FaceTrack(
            trackId=nextTrackId,
            x=x,
            y=y,
            width=width,
            height=height,
            lastSeenFrame=frameIndex,
          ),
        )
        nextTrackId += 1

      tracks = active
      sessionMemory = pruneSessionMemory(
        sessionMemory,
        frameIndex,
        sessionMemoryFrames,
      )

      if shouldInfer:
        for track in tracks:
          if track.lostFrames > 0 or track.identity is not None:
            continue
          embedding = embedBox(track.x, track.y, track.width, track.height)
          if embedding is None:
            continue
          track.lastEmbedding = embedding
          tryIdentifyTrack(
            track,
            embedding,
            gallery,
            requiredLevel,
            matchThreshold,
            confirmHits,
            sessionMemory,
            frameIndex,
            sessionMemoryFrames,
            loadReferenceBgr,
          )

      for track in tracks:
        nameLine, metaLine, score, decision = trackLabel(track)
        drawFaceHud(
          frame,
          track.x,
          track.y,
          track.width,
          track.height,
          nameLine,
          metaLine,
          score,
          decision,
          track.referenceBgr if track.identity is not None else None,
          animT + track.trackId * 0.37,
          locked=track.identity is not None,
        )

      cv2.imshow(windowTitle, frame)
      frameIndex += 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  finally:
    capture.release()
    cv2.destroyAllWindows()
    gallery.close()
