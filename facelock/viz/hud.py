"""JARVIS-style HUD overlays for the live camera.

All chrome is sized from the face bounding square (w×h), not fixed pixels:
text, padding, stroke, and gallery ref scale with that box.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from facelock.gallery import (
  DECISION_GRANTED,
  DECISION_INSUFFICIENT,
  DECISION_STRANGER,
)


WHITE = (235, 235, 235)
CYAN = (230, 200, 120)
CYAN_SOFT = (170, 145, 80)
MINT = (190, 230, 160)
AMBER = (60, 170, 230)
CRIMSON = (90, 90, 230)
PANEL = (16, 12, 10)


def decisionAccent(decision: str | None) -> tuple[int, int, int]:
  """Accent color for clearance state."""
  if decision == DECISION_GRANTED:
    return MINT
  if decision == DECISION_INSUFFICIENT:
    return AMBER
  if decision == DECISION_STRANGER:
    return CRIMSON
  return CYAN


def ease(t: float, speed: float = 1.2) -> float:
  """Slow sine phase in [0, 1]."""
  return 0.5 + 0.5 * math.sin(t * speed)


def faceSide(faceW: int, faceH: int) -> float:
  """Primary size unit: shorter side of the face box."""
  return float(max(1, min(faceW, faceH)))


def strokeFor(side: float) -> int:
  """Line thickness proportional to the face square."""
  return max(1, int(round(side / 140.0)))


def alphaRect(
  frame: np.ndarray,
  x1: int,
  y1: int,
  x2: int,
  y2: int,
  color: tuple[int, int, int],
  alpha: float,
) -> None:
  """Fill a rectangle with translucency."""
  h, w = frame.shape[:2]
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(w, x2), min(h, y2)
  if x2 <= x1 or y2 <= y1:
    return
  roi = frame[y1:y2, x1:x2]
  wash = np.empty_like(roi)
  wash[:] = color
  cv2.addWeighted(wash, alpha, roi, 1.0 - alpha, 0, dst=roi)


def drawBrackets(
  frame: np.ndarray,
  x: int,
  y: int,
  w: int,
  h: int,
  color: tuple[int, int, int],
  t: float,
  locked: bool,
  side: float,
) -> None:
  """Corner brackets sized from the face square."""
  arm = max(int(side * 0.18), strokeFor(side) * 6)
  inset = 0 if locked else int(side * 0.012 * ease(t, 2.0))
  thickness = strokeFor(side)
  x1, y1 = x - inset, y - inset
  x2, y2 = x + w + inset, y + h + inset

  lines = [
    ((x1, y1), (x1 + arm, y1)),
    ((x1, y1), (x1, y1 + arm)),
    ((x2, y1), (x2 - arm, y1)),
    ((x2, y1), (x2, y1 + arm)),
    ((x1, y2), (x1 + arm, y2)),
    ((x1, y2), (x1, y2 - arm)),
    ((x2, y2), (x2 - arm, y2)),
    ((x2, y2), (x2, y2 - arm)),
  ]
  for p1, p2 in lines:
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def drawReticle(
  frame: np.ndarray,
  cx: int,
  cy: int,
  radius: int,
  color: tuple[int, int, int],
  t: float,
  locked: bool,
  side: float,
) -> None:
  """Circular reticle; radius already comes from the face box."""
  if radius < 10:
    return
  thickness = strokeFor(side)
  gap = 28 if not locked else 14
  start = int((t * 40) % 360)
  cv2.ellipse(
    frame,
    (cx, cy),
    (radius, radius),
    0,
    start + gap,
    start + 360,
    color,
    thickness,
    cv2.LINE_AA,
  )
  inner = max(6, radius - int(side * 0.09))
  cv2.circle(frame, (cx, cy), inner, CYAN_SOFT, thickness, cv2.LINE_AA)
  tick = max(4, radius // 10)
  for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    x0 = cx + dx * (inner - tick)
    y0 = cy + dy * (inner - tick)
    x1 = cx + dx * (inner + 1)
    y1 = cy + dy * (inner + 1)
    cv2.line(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def drawScanLine(
  frame: np.ndarray,
  x: int,
  y: int,
  w: int,
  h: int,
  color: tuple[int, int, int],
  t: float,
  side: float,
) -> None:
  """Horizontal sweep across the face box."""
  if w < 8 or h < 8:
    return
  phase = (t * 0.35) % 1.0
  scanY = y + int(phase * max(1, h - 1))
  band = max(2, int(side * 0.015))
  y0 = max(y, scanY - band)
  y1 = min(y + h, scanY + band + 1)
  if y1 > y0:
    alphaRect(frame, x, y0, x + w, y1, color, 0.12)
  cv2.line(
    frame,
    (x + 2, scanY),
    (x + w - 2, scanY),
    color,
    strokeFor(side),
    cv2.LINE_AA,
  )


def drawReadout(
  frame: np.ndarray,
  x: int,
  y: int,
  nameLine: str,
  metaLine: str,
  color: tuple[int, int, int],
  side: float,
) -> None:
  """Identity chip sized as a fraction of the face square."""
  font = cv2.FONT_HERSHEY_SIMPLEX
  # OpenCV font scale ≈ pixel height / ~22 at thickness 1.
  scaleName = (side * 0.11) / 22.0
  scaleMeta = (side * 0.085) / 22.0
  thickness = strokeFor(side)
  nameSize = cv2.getTextSize(nameLine, font, scaleName, thickness)[0]
  metaSize = cv2.getTextSize(metaLine, font, scaleMeta, thickness)[0]
  padX = max(4, int(side * 0.05))
  padY = max(4, int(side * 0.04))
  gap = max(4, int(side * 0.035))
  boxW = max(nameSize[0], metaSize[0]) + padX * 2
  boxH = nameSize[1] + metaSize[1] + padY * 2 + gap
  top = max(0, y - boxH - max(4, int(side * 0.04)))
  left = max(0, x)
  frameH, frameW = frame.shape[:2]
  if left + boxW >= frameW:
    left = max(0, frameW - boxW - 1)

  alphaRect(frame, left, top, left + boxW, top + boxH, PANEL, 0.55)
  cv2.rectangle(
    frame,
    (left, top),
    (left + boxW, top + boxH),
    CYAN_SOFT,
    thickness,
    cv2.LINE_AA,
  )
  cv2.line(
    frame,
    (left, top),
    (left, top + boxH),
    color,
    max(2, thickness + 1),
    cv2.LINE_AA,
  )
  cv2.putText(
    frame,
    nameLine,
    (left + padX, top + padY + nameSize[1]),
    font,
    scaleName,
    WHITE,
    thickness,
    cv2.LINE_AA,
  )
  cv2.putText(
    frame,
    metaLine,
    (left + padX, top + padY + nameSize[1] + metaSize[1] + gap),
    font,
    scaleMeta,
    color,
    thickness,
    cv2.LINE_AA,
  )


def drawMatchMeter(
  frame: np.ndarray,
  x: int,
  y: int,
  w: int,
  score: int | None,
  color: tuple[int, int, int],
  side: float,
) -> None:
  """Match meter under the face box."""
  if score is None:
    return
  barY = y + max(4, int(side * 0.045))
  thickness = max(2, strokeFor(side))
  if barY + thickness >= frame.shape[0] or w < 20:
    return
  cv2.line(
    frame,
    (x, barY),
    (x + w, barY),
    CYAN_SOFT,
    max(1, thickness - 1),
    cv2.LINE_AA,
  )
  fill = int(w * max(0, min(100, score)) / 100.0)
  if fill > 0:
    cv2.line(frame, (x, barY), (x + fill, barY), color, thickness, cv2.LINE_AA)


def drawReferencePanel(
  frame: np.ndarray,
  referenceBgr: np.ndarray,
  anchorX: int,
  anchorY: int,
  color: tuple[int, int, int],
  side: float,
) -> None:
  """Gallery ref ≈ 42% of the face square side."""
  frameH, frameW = frame.shape[:2]
  panelSide = max(48, int(side * 0.42))
  panel = cv2.resize(referenceBgr, (panelSide, panelSide))

  gap = max(8, int(side * 0.08))
  x1 = min(frameW - panelSide - 4, max(0, anchorX + gap))
  y1 = min(frameH - panelSide - 4, max(0, anchorY))
  x2 = x1 + panelSide
  y2 = y1 + panelSide
  frame[y1:y2, x1:x2] = panel

  thickness = strokeFor(side)
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
  mark = max(6, int(side * 0.06))
  cv2.line(frame, (x1, y1), (x1 + mark, y1), WHITE, thickness, cv2.LINE_AA)
  cv2.line(frame, (x1, y1), (x1, y1 + mark), WHITE, thickness, cv2.LINE_AA)
  cv2.line(frame, (x2, y2), (x2 - mark, y2), WHITE, thickness, cv2.LINE_AA)
  cv2.line(frame, (x2, y2), (x2, y2 - mark), WHITE, thickness, cv2.LINE_AA)

  linkY = y1 + max(12, int(side * 0.12))
  cv2.line(
    frame,
    (max(0, anchorX), min(frameH - 1, anchorY + max(12, int(side * 0.12)))),
    (x1, linkY),
    CYAN_SOFT,
    thickness,
    cv2.LINE_AA,
  )


def drawFaceHud(
  frame: np.ndarray,
  x: int,
  y: int,
  w: int,
  h: int,
  nameLine: str,
  metaLine: str,
  score: int | None,
  decision: str | None,
  referenceBgr: np.ndarray | None,
  t: float,
  locked: bool,
) -> None:
  """JARVIS-style overlay for one tracked face."""
  color = decisionAccent(decision)
  side = faceSide(w, h)
  cx, cy = x + w // 2, y + h // 2
  radius = int(max(w, h) * 0.55)

  drawBrackets(frame, x, y, w, h, color, t, locked=locked, side=side)
  drawReticle(frame, cx, cy, radius, color, t, locked=locked, side=side)
  drawScanLine(frame, x, y, w, h, color, t, side=side)
  drawReadout(frame, x, y, nameLine, metaLine, color, side=side)
  drawMatchMeter(frame, x, y + h, w, score, color, side=side)

  if referenceBgr is not None and locked:
    drawReferencePanel(frame, referenceBgr, x + w, y, color, side=side)
