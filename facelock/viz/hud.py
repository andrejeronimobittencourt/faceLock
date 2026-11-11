"""JARVIS-style HUD overlays for the live camera.

All chrome is sized from the face bounding square (w×h), not fixed pixels:
text, padding, stroke, and gallery ref scale with that box.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

_FONT_CANDIDATES = (
  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
  '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
  '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
  '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
  '/usr/share/fonts/TTF/DejaVuSans.ttf',
)


@lru_cache(maxsize=8)
def _loadHudFont(pixelHeight: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  """Load a TTF that supports Latin accents; fall back to PIL default."""
  size = max(8, int(pixelHeight))
  for candidate in _FONT_CANDIDATES:
    path = Path(candidate)
    if path.is_file():
      return ImageFont.truetype(str(path), size=size)
  return ImageFont.load_default()


def _textSizeUtf8(text: str, pixelHeight: int) -> tuple[int, int]:
  """Width/height of UTF-8 text at the given pixel height."""
  font = _loadHudFont(pixelHeight)
  left, top, right, bottom = font.getbbox(text)
  return int(right - left), int(bottom - top)


def _putTextUtf8(
  frame: np.ndarray,
  text: str,
  org: tuple[int, int],
  pixelHeight: int,
  colorBgr: tuple[int, int, int],
) -> None:
  """Draw UTF-8 text with baseline org (x, baselineY)."""
  if not text:
    return
  font = _loadHudFont(pixelHeight)
  left, top, right, bottom = font.getbbox(text)
  textW = max(1, int(right - left))
  textH = max(1, int(bottom - top))
  overlay = Image.new('RGBA', (textW, textH), (0, 0, 0, 0))
  draw = ImageDraw.Draw(overlay)
  colorRgb = (int(colorBgr[2]), int(colorBgr[1]), int(colorBgr[0]), 255)
  draw.text((-left, -top), text, font=font, fill=colorRgb)

  x, baselineY = org
  y = int(baselineY - textH)
  frameH, frameW = frame.shape[:2]
  x1 = max(0, x)
  y1 = max(0, y)
  x2 = min(frameW, x + textW)
  y2 = min(frameH, y + textH)
  if x1 >= x2 or y1 >= y2:
    return
  srcX1 = x1 - x
  srcY1 = y1 - y
  srcX2 = srcX1 + (x2 - x1)
  srcY2 = srcY1 + (y2 - y1)
  patch = np.asarray(overlay)[srcY1:srcY2, srcX1:srcX2]
  alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
  rgb = patch[:, :, :3][:, :, ::-1].astype(np.float32)
  roi = frame[y1:y2, x1:x2].astype(np.float32)
  frame[y1:y2, x1:x2] = np.clip(rgb * alpha + roi * (1.0 - alpha), 0, 255).astype(
    np.uint8,
  )



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
  namePx = max(10, int(side * 0.11))
  metaPx = max(9, int(side * 0.085))
  thickness = strokeFor(side)
  nameW, nameH = _textSizeUtf8(nameLine, namePx)
  metaW, metaH = _textSizeUtf8(metaLine, metaPx)
  padX = max(4, int(side * 0.05))
  padY = max(4, int(side * 0.04))
  gap = max(4, int(side * 0.035))
  boxW = max(nameW, metaW) + padX * 2
  boxH = nameH + metaH + padY * 2 + gap
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
  _putTextUtf8(
    frame,
    nameLine,
    (left + padX, top + padY + nameH),
    namePx,
    WHITE,
  )
  _putTextUtf8(
    frame,
    metaLine,
    (left + padX, top + padY + nameH + metaH + gap),
    metaPx,
    color,
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
