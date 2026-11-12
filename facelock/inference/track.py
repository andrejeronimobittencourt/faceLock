"""IoU face tracking + gallery identity binding with appearance verify."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facelock.gallery import (
  ACCESS_LEVEL_NAMES,
  DECISION_GRANTED,
  DECISION_INSUFFICIENT,
  DECISION_STRANGER,
  AccessLevel,
  FaceGallery,
  IdentityRecord,
  MatchResult,
  alikeScore,
)


@dataclass
class SessionMemory:
  """Short-term memory of a face that just left the frame."""

  embedding: np.ndarray
  identity: IdentityRecord
  referenceBgr: np.ndarray | None
  decision: str
  matchScore: int
  diedAtFrame: int


@dataclass
class FaceTrack:
  """One physical face continuously associated across frames."""

  trackId: int
  x: int
  y: int
  width: int
  height: int
  lostFrames: int = 0
  lastSeenFrame: int = 0
  identity: IdentityRecord | None = None
  decision: str | None = None
  referenceBgr: np.ndarray | None = None
  matchScore: int | None = None
  pendingIdentityId: int | None = None
  confirmHits: int = 0
  lastEmbedding: np.ndarray | None = None
  verifyMisses: int = 0


def smoothTrackBox(
  track: FaceTrack,
  x: int,
  y: int,
  width: int,
  height: int,
  positionAlpha: float = 0.45,
  sizeAlpha: float = 0.22,
  sizeDeadband: float = 0.06,
) -> None:
  """EMA-smooth box; ignore tiny size jitter so the HUD does not flicker."""
  track.x = int(round(positionAlpha * x + (1.0 - positionAlpha) * track.x))
  track.y = int(round(positionAlpha * y + (1.0 - positionAlpha) * track.y))
  if track.width > 0 and track.height > 0:
    relW = abs(width - track.width) / float(track.width)
    relH = abs(height - track.height) / float(track.height)
    if relW < sizeDeadband and relH < sizeDeadband:
      return
  track.width = int(round(sizeAlpha * width + (1.0 - sizeAlpha) * track.width))
  track.height = int(round(sizeAlpha * height + (1.0 - sizeAlpha) * track.height))
  track.width = max(1, track.width)
  track.height = max(1, track.height)


def boxIou(
  a: tuple[int, int, int, int],
  b: tuple[int, int, int, int],
) -> float:
  """Intersection-over-union for (x, y, w, h) boxes."""

  ax1, ay1, aw, ah = a
  bx1, by1, bw, bh = b
  ax2, ay2 = ax1 + aw, ay1 + ah
  bx2, by2 = bx1 + bw, by1 + bh
  interX1 = max(ax1, bx1)
  interY1 = max(ay1, by1)
  interX2 = min(ax2, bx2)
  interY2 = min(ay2, by2)
  interW = max(0, interX2 - interX1)
  interH = max(0, interY2 - interY1)
  inter = interW * interH
  if inter <= 0:
    return 0.0
  union = aw * ah + bw * bh - inter
  return inter / union if union > 0 else 0.0


def normalizeEmbedding(embedding: np.ndarray) -> np.ndarray:
  """Return an L2-normalized float32 vector."""
  vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
  norm = float(np.linalg.norm(vector))
  if norm < 1e-8:
    return vector
  return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
  """Cosine between two embeddings."""
  return float(np.dot(normalizeEmbedding(a), normalizeEmbedding(b)))


def accessDecisionForIdentity(
  identity: IdentityRecord,
  cosineValue: float,
  requiredLevel: AccessLevel,
) -> MatchResult:
  """Build a MatchResult for an already-chosen identity."""
  score = alikeScore(cosineValue)
  if identity.accessLevel < requiredLevel:
    return MatchResult(
      decision=DECISION_INSUFFICIENT,
      identity=identity,
      cosine=cosineValue,
      score=score,
      requiredLevel=requiredLevel,
      message=(
        f'PERMISSION DENIED — {identity.name} clearance '
        f'{identity.accessName} < required {ACCESS_LEVEL_NAMES[requiredLevel]}'
      ),
    )
  return MatchResult(
    decision=DECISION_GRANTED,
    identity=identity,
    cosine=cosineValue,
    score=score,
    requiredLevel=requiredLevel,
    message=f'ACCESS GRANTED — {identity.name} ({identity.accessName})',
  )


def associateBoxes(
  tracks: list[FaceTrack],
  boxes: list[tuple[int, int, int, int]],
  iouThreshold: float = 0.2,
) -> tuple[
  list[tuple[FaceTrack, tuple[int, int, int, int]]],
  list[tuple[int, int, int, int]],
  list[FaceTrack],
]:
  """Greedy IoU association: tracks ↔ current detections."""
  unmatchedBoxes = set(range(len(boxes)))
  unmatchedTracks = set(range(len(tracks)))
  pairs: list[tuple[int, int, float]] = []
  for trackIndex, track in enumerate(tracks):
    trackBox = (track.x, track.y, track.width, track.height)
    for boxIndex, box in enumerate(boxes):
      score = boxIou(trackBox, box)
      if score >= iouThreshold:
        pairs.append((trackIndex, boxIndex, score))
  pairs.sort(key=lambda item: item[2], reverse=True)

  matched: list[tuple[FaceTrack, tuple[int, int, int, int]]] = []
  for trackIndex, boxIndex, _score in pairs:
    if trackIndex not in unmatchedTracks or boxIndex not in unmatchedBoxes:
      continue
    unmatchedTracks.remove(trackIndex)
    unmatchedBoxes.remove(boxIndex)
    matched.append((tracks[trackIndex], boxes[boxIndex]))

  newBoxes = [boxes[index] for index in sorted(unmatchedBoxes)]
  lostTracks = [tracks[index] for index in sorted(unmatchedTracks)]
  return matched, newBoxes, lostTracks


def bindIdentity(
  track: FaceTrack,
  identity: IdentityRecord,
  cosineValue: float,
  requiredLevel: AccessLevel,
  referenceBgr: np.ndarray | None,
) -> None:
  """Attach a gallery identity to this track (score frozen at bind)."""
  result = accessDecisionForIdentity(identity, cosineValue, requiredLevel)
  track.identity = identity
  track.decision = result.decision
  track.matchScore = result.score
  track.referenceBgr = referenceBgr
  track.pendingIdentityId = None
  track.confirmHits = 0
  track.verifyMisses = 0


def appearanceStillMatches(
  track: FaceTrack,
  embedding: np.ndarray,
  verifyThreshold: float,
  verifyMissesNeeded: int,
) -> bool:
  """True while the crop still looks like the bound identity."""
  if track.identity is None:
    return True
  value = cosine(embedding, track.identity.embedding)
  if value >= verifyThreshold:
    track.verifyMisses = 0
    return True
  track.verifyMisses += 1
  return track.verifyMisses < verifyMissesNeeded


def tryIdentifyTrack(
  track: FaceTrack,
  embedding: np.ndarray,
  gallery: FaceGallery,
  requiredLevel: AccessLevel,
  matchThreshold: float,
  confirmHitsNeeded: int,
  sessionMemory: list[SessionMemory],
  frameIndex: int,
  sessionMemoryFrames: int,
  loadReference,
) -> None:
  """Identify an unbound track; bind once confirmed."""
  if track.identity is not None:
    return

  bestMemory: SessionMemory | None = None
  bestMemoryCosine = -1.0
  for memory in sessionMemory:
    if frameIndex - memory.diedAtFrame > sessionMemoryFrames:
      continue
    value = cosine(embedding, memory.embedding)
    if value > bestMemoryCosine:
      bestMemoryCosine = value
      bestMemory = memory
  if bestMemory is not None and bestMemoryCosine >= matchThreshold:
    bindIdentity(
      track,
      bestMemory.identity,
      bestMemoryCosine,
      requiredLevel,
      bestMemory.referenceBgr,
    )
    track.decision = bestMemory.decision
    return

  identity, galleryCosine = gallery.bestMatch(
    embedding,
    matchThreshold=matchThreshold,
  )
  if identity is None:
    track.pendingIdentityId = None
    track.confirmHits = 0
    track.decision = DECISION_STRANGER
    track.matchScore = None
    track.referenceBgr = None
    return

  if track.pendingIdentityId == identity.id:
    track.confirmHits += 1
  else:
    track.pendingIdentityId = identity.id
    track.confirmHits = 1

  if track.confirmHits >= confirmHitsNeeded:
    bindIdentity(
      track,
      identity,
      galleryCosine,
      requiredLevel,
      loadReference(identity.imagePath),
    )


def retireTrack(
  track: FaceTrack,
  sessionMemory: list[SessionMemory],
  frameIndex: int,
) -> None:
  """Drop a track; keep short memory if it had a gallery identity."""
  if track.identity is not None and track.lastEmbedding is not None:
    sessionMemory.append(
      SessionMemory(
        embedding=track.lastEmbedding.copy(),
        identity=track.identity,
        referenceBgr=track.referenceBgr,
        decision=track.decision or DECISION_GRANTED,
        matchScore=track.matchScore or 0,
        diedAtFrame=frameIndex,
      ),
    )


def pruneSessionMemory(
  sessionMemory: list[SessionMemory],
  frameIndex: int,
  sessionMemoryFrames: int,
) -> list[SessionMemory]:
  """Forget departed faces that have been gone too long."""
  return [
    memory
    for memory in sessionMemory
    if frameIndex - memory.diedAtFrame <= sessionMemoryFrames
  ]


def trackLabel(track: FaceTrack) -> tuple[str, str, int | None, str | None]:
  """HUD lines for a track. Score only when a gallery identity is bound."""
  if track.identity is not None:
    score = track.matchScore if track.matchScore is not None else 0
    decision = track.decision or DECISION_GRANTED
    return (
      track.identity.name.upper(),
      f'{track.identity.accessName.upper()}  //  {score:03d}',
      score,
      decision,
    )
  return 'UNKNOWN', 'STRANGER', None, DECISION_STRANGER
