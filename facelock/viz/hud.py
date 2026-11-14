"""sci-fi-style HUD overlays for the live camera."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def hudGeometry(x: int, y: int, w: int, h: int) -> tuple[int, int, float]:
  """Center and side length of the square HUD frame for a detector box."""
  side = float(max(1, w, h))
  return x + w // 2, y + h // 2, side


def strokeFor(side: float) -> int:
  """Hairline strokes; only very large faces get 2px."""
  return 1 if side < 260 else 2


def _clampInt(value: float, low: int, high: int) -> int:
  return max(low, min(high, int(round(value))))


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


def _blendColor(
  a: tuple[int, int, int],
  b: tuple[int, int, int],
  weight: float,
) -> tuple[int, int, int]:
  """Linear blend between two BGR colors; weight in [0, 1] toward b."""
  w = max(0.0, min(1.0, float(weight)))
  return (
    int(round(a[0] + (b[0] - a[0]) * w)),
    int(round(a[1] + (b[1] - a[1]) * w)),
    int(round(a[2] + (b[2] - a[2]) * w)),
  )


def _dim(color: tuple[int, int, int], amount: float = 0.55) -> tuple[int, int, int]:
  """Blend an accent toward the panel tone."""
  return _blendColor(color, PANEL, amount)


def drawBrackets(
  frame: np.ndarray,
  cx: int,
  cy: int,
  frameRadius: int,
  color: tuple[int, int, int],
  t: float,
  locked: bool,
  side: float,
) -> None:
  """Corner brackets around the HUD square."""
  breathe = 0 if locked else int(round(side * 0.01 * ease(t, 2.4)))
  r = frameRadius + breathe
  arm = max(8, int(side * 0.12))
  rail = max(4, int(arm * 0.45))
  offset = 4 if side >= 140 else 3
  thickness = strokeFor(side)
  soft = _dim(color)
  x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
  for px, py, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
    cv2.line(frame, (px, py), (px + sx * arm, py), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px, py + sy * arm), color, thickness, cv2.LINE_AA)
    cv2.line(
      frame,
      (px + sx * (arm - rail), py + sy * offset),
      (px + sx * arm, py + sy * offset),
      soft,
      1,
      cv2.LINE_AA,
    )


def ringRadii(half: float) -> tuple[int, int, int]:
  """Inner, mid, and outer reticle radii for half-side ``half``."""
  return int(half * 1.02), int(half * 1.09), int(half * 1.15)


def drawRings(
  frame: np.ndarray,
  cx: int,
  cy: int,
  half: float,
  color: tuple[int, int, int],
  t: float,
  locked: bool,
  side: float,
) -> None:
  """Draw the concentric reticle around the face."""
  if half < 20:
    return
  rInner, rMid, rOuter = ringRadii(half)
  thickness = strokeFor(side)
  soft = _dim(color)
  speed = 9.0 if locked else 36.0

  base = (t * speed) % 360.0
  for k in range(3):
    start = base + k * 120.0
    cv2.ellipse(
      frame,
      (cx, cy),
      (rOuter, rOuter),
      0,
      start,
      start + 76.0,
      color,
      thickness,
      cv2.LINE_AA,
    )

  counter = (-t * speed * 0.6) % 360.0
  for k in range(4):
    start = counter + k * 90.0
    cv2.ellipse(frame, (cx, cy), (rMid, rMid), 0, start, start + 12.0, soft, 1, cv2.LINE_AA)

  cv2.circle(frame, (cx, cy), rInner, soft, 1, cv2.LINE_AA)
  for deg in range(0, 360, 15):
    major = deg % 90 == 0
    length = max(2, int(half * (0.05 if major else 0.02)))
    angle = math.radians(deg)
    ca, sa = math.cos(angle), math.sin(angle)
    p0 = (int(round(cx + ca * rInner)), int(round(cy + sa * rInner)))
    p1 = (
      int(round(cx + ca * (rInner + length))),
      int(round(cy + sa * (rInner + length))),
    )
    cv2.line(frame, p0, p1, color if major else soft, 1, cv2.LINE_AA)


def scanLineY(cy: int, radius: int, t: float) -> int:
  """Scan-line Y inside a ring of the given radius."""
  phase = (t * 0.35) % 1.0
  return int(round(cy - radius + phase * 2 * radius))


def drawScanLine(
  frame: np.ndarray,
  cx: int,
  cy: int,
  radius: int,
  color: tuple[int, int, int],
  t: float,
  side: float,
) -> None:
  """Draw the scan chord and fading wake inside the inner ring."""
  scanY = scanLineY(cy, radius, t)
  if radius < 8:
    return
  frameH, frameW = frame.shape[:2]
  colorArr = np.asarray(color, dtype=np.float32)
  wake = max(4, int(side * 0.07))

  for i in range(1, wake + 1):
    row = scanY - i
    dy = row - cy
    if row < 0 or row >= frameH or abs(dy) >= radius:
      continue
    halfChord = int(math.sqrt(radius * radius - dy * dy))
    xa = max(0, cx - halfChord)
    xb = min(frameW, cx + halfChord + 1)
    if xb <= xa:
      continue
    alpha = 0.16 * (1.0 - i / float(wake)) ** 2
    segment = frame[row, xa:xb].astype(np.float32)
    frame[row, xa:xb] = (segment * (1.0 - alpha) + colorArr * alpha).astype(np.uint8)

  dy = scanY - cy
  if abs(dy) < radius:
    halfChord = int(math.sqrt(radius * radius - dy * dy))
    left = cx - halfChord
    right = cx + halfChord
    cv2.line(frame, (left, scanY), (right, scanY), color, 1, cv2.LINE_AA)
    tick = max(2, int(side * 0.012))
    cv2.line(frame, (left, scanY - tick), (left, scanY + tick), color, 1, cv2.LINE_AA)
    cv2.line(frame, (right, scanY - tick), (right, scanY + tick), color, 1, cv2.LINE_AA)


MATCH_LABEL = 'MATCH'


@dataclass(frozen=True)
class ReadoutMetrics:
  """Laid-out sizes for name and clearance text."""

  namePx: int
  metaPx: int
  nameH: int
  metaH: int
  gap: int
  width: int
  height: int


def measureReadout(nameLine: str, metaLine: str, side: float) -> ReadoutMetrics:
  """Measure name/clearance text for a HUD of ``side`` pixels."""
  namePx = max(11, int(round(side * 0.055)))
  metaPx = max(9, int(round(side * 0.042)))
  nameW, nameH = _textSizeUtf8(nameLine, namePx)
  metaW, metaH = _textSizeUtf8(metaLine, metaPx)
  gap = max(2, int(round(side * 0.015)))
  return ReadoutMetrics(
    namePx=namePx,
    metaPx=metaPx,
    nameH=nameH,
    metaH=metaH,
    gap=gap,
    width=max(nameW, metaW),
    height=nameH + gap + metaH,
  )


def drawReadout(
  frame: np.ndarray,
  left: int,
  top: int,
  nameLine: str,
  metaLine: str,
  color: tuple[int, int, int],
  metrics: ReadoutMetrics,
) -> None:
  """Draw name and clearance text at ``(left, top)``."""
  nameBaseline = top + metrics.nameH
  _putTextUtf8(frame, nameLine, (left, nameBaseline), metrics.namePx, WHITE)
  _putTextUtf8(
    frame,
    metaLine,
    (left, nameBaseline + metrics.gap + metrics.metaH),
    metrics.metaPx,
    color,
  )


def drawMatchMeter(
  frame: np.ndarray,
  left: int,
  right: int,
  top: int,
  score: int | None,
  color: tuple[int, int, int],
  side: float,
) -> None:
  """Draw a thin match bar (dim track, accent fill)."""
  if score is None:
    return
  frameH, frameW = frame.shape[:2]
  thickness = strokeFor(side)
  if top < 0 or top + thickness > frameH:
    return
  left = max(0, left)
  right = min(frameW - 1, right)
  if right <= left:
    return
  bottom = top + thickness - 1
  cv2.rectangle(frame, (left, top), (right, bottom), _dim(color, 0.7), -1)
  fraction = max(0, min(100, score)) / 100.0
  filledRight = left + int(round((right - left) * fraction))
  if filledRight > left:
    cv2.rectangle(frame, (left, top), (filledRight, bottom), color, -1)


def _drawReferenceImage(
  frame: np.ndarray,
  referenceBgr: np.ndarray,
  left: int,
  top: int,
  size: int,
  color: tuple[int, int, int],
) -> None:
  """Draw a gallery thumbnail with border and corner ticks."""
  thumbnail = cv2.resize(referenceBgr, (size, size), interpolation=cv2.INTER_AREA)
  frame[top:top + size, left:left + size] = thumbnail

  right = left + size - 1
  bottom = top + size - 1
  cv2.rectangle(frame, (left, top), (right, bottom), _dim(color), 1)
  mark = max(4, int(size * 0.12))
  for px, py, sx, sy in (
    (left, top, 1, 1),
    (right, top, -1, 1),
    (left, bottom, 1, -1),
    (right, bottom, -1, -1),
  ):
    cv2.line(frame, (px, py), (px + sx * mark, py), color, 1)
    cv2.line(frame, (px, py), (px, py + sy * mark), color, 1)


def drawMatchCard(
  frame: np.ndarray,
  referenceBgr: np.ndarray | None,
  cx: int,
  cy: int,
  frameRadius: int,
  nameLine: str,
  metaLine: str,
  color: tuple[int, int, int],
  side: float,
) -> None:
  """Draw the match card beside the HUD (thumbnail when locked, always text)."""
  frameH, frameW = frame.shape[:2]
  metrics = measureReadout(nameLine, metaLine, side)
  size = max(44, int(round(side * 0.24))) if referenceBgr is not None else 0
  pad = max(4, int(round(side * 0.025)))
  inner = max(3, int(round(side * 0.02)))
  bar = max(2, int(round(side * 0.008)))
  labelPx = max(9, int(round(side * 0.038)))
  labelH = _textSizeUtf8(MATCH_LABEL, labelPx)[1] if size else 0

  contentW = max(size, metrics.width)
  contentH = metrics.height
  if size:
    contentH += labelH + inner + size + inner
  cardW = bar + pad * 2 + contentW
  cardH = pad * 2 + contentH

  columnGap = max(8, int(round(side * 0.06)))
  left = cx + frameRadius + columnGap
  if left + cardW > frameW - 2:
    left = cx - frameRadius - columnGap - cardW
  left = max(2, min(frameW - cardW - 2, left))
  top = max(2, min(frameH - cardH - 2, cy - frameRadius))

  alphaRect(frame, left, top, left + cardW, top + cardH, PANEL, 0.6)
  cv2.rectangle(frame, (left, top), (left + bar - 1, top + cardH - 1), color, -1)
  accentEnd = left + int(cardW * 0.4)
  cv2.line(frame, (left, top), (accentEnd, top), color, 1)
  cv2.line(frame, (accentEnd, top), (left + cardW - 1, top), _dim(color, 0.7), 1)

  x0 = left + bar + pad
  y = top + pad
  if size:
    _putTextUtf8(frame, MATCH_LABEL, (x0, y + labelH), labelPx, _dim(color, 0.35))
    y += labelH + inner
    _drawReferenceImage(frame, referenceBgr, x0, y, size, color)
    y += size + inner
  drawReadout(frame, x0, y, nameLine, metaLine, color, metrics)


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
  """sci-fi-style overlay for one tracked face."""
  color = decisionAccent(decision)
  cx, cy, side = hudGeometry(x, y, w, h)
  half = side / 2.0
  innerRadius, _midRadius, outerRadius = ringRadii(half)
  frameRadius = int(round(outerRadius + max(3.0, side * 0.015)))
  spacing = max(5, int(round(side * 0.03)))

  drawRings(frame, cx, cy, half, color, t, locked=locked, side=side)
  drawScanLine(frame, cx, cy, innerRadius, color, t, side=side)
  drawBrackets(frame, cx, cy, frameRadius, color, t, locked=locked, side=side)

  meterTop = cy + frameRadius + spacing
  drawMatchMeter(frame, cx - frameRadius, cx + frameRadius, meterTop, score, color, side=side)
  drawMatchCard(
    frame,
    referenceBgr if locked else None,
    cx,
    cy,
    frameRadius,
    nameLine,
    metaLine,
    color,
    side=side,
  )
