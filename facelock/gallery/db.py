"""SQLite face gallery: identities, embeddings, and access levels."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from PIL import Image


class AccessLevel(IntEnum):
  """Ordered clearance. Higher number = more access."""

  USER = 1
  OPERATOR = 2
  ADMIN = 3


ACCESS_LEVEL_NAMES = {
  AccessLevel.USER: 'user',
  AccessLevel.OPERATOR: 'operator',
  AccessLevel.ADMIN: 'admin',
}
ACCESS_NAME_TO_LEVEL = {name: level for level, name in ACCESS_LEVEL_NAMES.items()}


DECISION_STRANGER = 'stranger_denied'
DECISION_INSUFFICIENT = 'insufficient_clearance'
DECISION_GRANTED = 'granted'


@dataclass
class IdentityRecord:
  """One registered person in the gallery."""

  id: int
  name: str
  accessLevel: AccessLevel
  imagePath: str
  embedding: np.ndarray
  createdAt: str
  updatedAt: str

  @property
  def accessName(self) -> str:
    """Human-readable access level name."""
    return ACCESS_LEVEL_NAMES[self.accessLevel]


@dataclass
class MatchResult:
  """Best gallery match for a probe embedding, plus clearance decision."""

  decision: str
  identity: IdentityRecord | None
  cosine: float
  score: int
  requiredLevel: AccessLevel
  message: str


def parseAccessLevel(value: str) -> AccessLevel:
  """Parse access level from name or int string."""
  cleaned = value.strip().lower()
  if cleaned in ACCESS_NAME_TO_LEVEL:
    return ACCESS_NAME_TO_LEVEL[cleaned]
  try:
    level = AccessLevel(int(cleaned))
  except ValueError as error:
    allowed = ', '.join(ACCESS_NAME_TO_LEVEL.keys())
    raise ValueError(f'Unknown access level {value!r}. Use: {allowed}') from error
  if level not in ACCESS_LEVEL_NAMES:
    raise ValueError(f'Access level int out of range: {value}')
  return level


def alikeScore(cosine: float) -> int:
  """Map cosine [-1, 1] to a 0–100 score."""
  return int(round(((cosine + 1.0) / 2.0) * 100.0))


def embeddingToBlob(embedding: np.ndarray) -> bytes:
  """Serialize a float32 embedding vector."""
  vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
  return vector.tobytes()


def blobToEmbedding(blob: bytes, dim: int) -> np.ndarray:
  """Deserialize a float32 embedding vector."""
  vector = np.frombuffer(blob, dtype=np.float32)
  if vector.size != dim:
    raise ValueError(f'Embedding size {vector.size} != expected dim {dim}')
  return vector.copy()


class FaceGallery:
  """SQLite-backed registry of faces, embeds, and clearance levels."""

  def __init__(self, dbPath: Path, facesDir: Path) -> None:
    self.dbPath = Path(dbPath)
    self.facesDir = Path(facesDir)
    self.facesDir.mkdir(parents=True, exist_ok=True)
    self.dbPath.parent.mkdir(parents=True, exist_ok=True)
    self._conn = sqlite3.connect(self.dbPath)
    self._conn.row_factory = sqlite3.Row
    self._initSchema()

  def close(self) -> None:
    """Close the database connection."""
    self._conn.close()

  def __enter__(self) -> FaceGallery:
    return self

  def __exit__(self, excType, exc, tb) -> None:
    self.close()

  def _initSchema(self) -> None:
    """Create tables if missing."""
    self._conn.execute(
      """
      CREATE TABLE IF NOT EXISTS identities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        access_level INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        embedding BLOB NOT NULL,
        embedding_dim INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
      """
    )
    self._conn.execute(
      """
      CREATE INDEX IF NOT EXISTS idx_identities_access
      ON identities(access_level)
      """
    )
    self._conn.commit()

  def _normalizeName(self, name: str) -> str:
    """Validate and clean an identity name."""
    cleanedName = name.strip()
    if not cleanedName:
      raise ValueError('Name must not be empty')
    if cleanedName.lower() == 'stranger':
      raise ValueError('Name "stranger" is reserved')
    return cleanedName

  def _normalizeEmbedding(self, embedding: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize an embedding vector."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size < 1:
      raise ValueError('Embedding is empty')
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
      raise ValueError('Embedding norm is ~0')
    return vector / norm

  def _now(self) -> str:
    """UTC-ish local timestamp string for created/updated fields."""
    return time.strftime('%Y-%m-%dT%H:%M:%S')

  def create(
    self,
    name: str,
    accessLevel: AccessLevel,
    sourceImage: Path,
    embedding: np.ndarray,
  ) -> IdentityRecord:
    """Create a new identity. Fails if the name already exists."""
    cleanedName = self._normalizeName(name)
    if self.getByName(cleanedName) is not None:
      raise ValueError(f'Identity already exists: {cleanedName}')

    vector = self._normalizeEmbedding(embedding)
    now = self._now()
    cursor = self._conn.execute(
      """
      INSERT INTO identities (
        name, access_level, image_path, embedding, embedding_dim, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
      """,
      (
        cleanedName,
        int(accessLevel),
        '',
        embeddingToBlob(vector),
        int(vector.size),
        now,
        now,
      ),
    )
    identityId = int(cursor.lastrowid)
    imagePath = self.facesDir / f'{identityId}.jpg'
    self._saveFaceImage(sourceImage, imagePath)
    self._conn.execute(
      'UPDATE identities SET image_path = ? WHERE id = ?',
      (str(imagePath), identityId),
    )
    self._conn.commit()
    record = self.getById(identityId)
    if record is None:
      raise RuntimeError('Failed to reload created identity')
    return record

  def register(
    self,
    name: str,
    accessLevel: AccessLevel,
    sourceImage: Path,
    embedding: np.ndarray,
  ) -> IdentityRecord:
    """Create, or fully replace face/access if the name already exists."""
    existing = self.getByName(name)
    if existing is None:
      return self.create(name, accessLevel, sourceImage, embedding)
    return self.update(
      identityId=existing.id,
      accessLevel=accessLevel,
      sourceImage=sourceImage,
      embedding=embedding,
    )

  def update(
    self,
    identityId: int | None = None,
    name: str | None = None,
    newName: str | None = None,
    accessLevel: AccessLevel | None = None,
    sourceImage: Path | None = None,
    embedding: np.ndarray | None = None,
  ) -> IdentityRecord:
    """Update fields on an identity looked up by id or name."""
    if identityId is not None:
      record = self.getById(identityId)
    elif name is not None:
      record = self.getByName(name)
    else:
      raise ValueError('Provide identityId or name')

    if record is None:
      raise ValueError('Identity not found')

    if sourceImage is not None and embedding is None:
      raise ValueError('Updating the face image requires a new embedding')
    if embedding is not None and sourceImage is None:
      raise ValueError('Updating the embedding requires a face image')

    nextName = record.name
    if newName is not None:
      cleanedNewName = self._normalizeName(newName)
      clash = self.getByName(cleanedNewName)
      if clash is not None and clash.id != record.id:
        raise ValueError(f'Name already taken: {cleanedNewName}')
      nextName = cleanedNewName

    nextAccess = record.accessLevel if accessLevel is None else accessLevel
    nextImagePath = record.imagePath
    nextEmbedding = record.embedding
    nextDim = int(record.embedding.size)

    if sourceImage is not None and embedding is not None:
      nextEmbedding = self._normalizeEmbedding(embedding)
      nextDim = int(nextEmbedding.size)
      imagePath = Path(record.imagePath)
      if not imagePath.as_posix():
        imagePath = self.facesDir / f'{record.id}.jpg'
      self._saveFaceImage(sourceImage, imagePath)
      nextImagePath = str(imagePath)

    self._conn.execute(
      """
      UPDATE identities
      SET name = ?, access_level = ?, image_path = ?, embedding = ?,
          embedding_dim = ?, updated_at = ?
      WHERE id = ?
      """,
      (
        nextName,
        int(nextAccess),
        nextImagePath,
        embeddingToBlob(nextEmbedding),
        nextDim,
        self._now(),
        record.id,
      ),
    )
    self._conn.commit()
    updated = self.getById(record.id)
    if updated is None:
      raise RuntimeError('Failed to reload updated identity')
    return updated

  def _saveFaceImage(self, sourceImage: Path, destPath: Path) -> None:
    """Copy/normalize registered face image into gallery storage."""
    image = Image.open(sourceImage).convert('RGB')
    destPath.parent.mkdir(parents=True, exist_ok=True)
    image.save(destPath, format='JPEG', quality=92)

  def delete(
    self,
    identityId: int | None = None,
    name: str | None = None,
  ) -> bool:
    """Delete by id or name. Returns True if a row was removed."""
    if identityId is not None:
      record = self.getById(identityId)
    elif name is not None:
      record = self.getByName(name)
    else:
      raise ValueError('Provide identityId or name')
    if record is None:
      return False
    imagePath = Path(record.imagePath)
    self._conn.execute('DELETE FROM identities WHERE id = ?', (record.id,))
    self._conn.commit()
    if imagePath.exists():
      imagePath.unlink()
    return True

  def listIdentities(self) -> list[IdentityRecord]:
    """Return all registered identities ordered by name."""
    rows = self._conn.execute(
      'SELECT * FROM identities ORDER BY name COLLATE NOCASE ASC'
    ).fetchall()
    return [self._rowToRecord(row) for row in rows]

  def getById(self, identityId: int) -> IdentityRecord | None:
    """Fetch one identity by primary key."""
    row = self._conn.execute(
      'SELECT * FROM identities WHERE id = ?',
      (identityId,),
    ).fetchone()
    return self._rowToRecord(row) if row is not None else None

  def getByName(self, name: str) -> IdentityRecord | None:
    """Fetch one identity by name (case-insensitive)."""
    row = self._conn.execute(
      'SELECT * FROM identities WHERE name = ? COLLATE NOCASE',
      (name.strip(),),
    ).fetchone()
    return self._rowToRecord(row) if row is not None else None

  def _rowToRecord(self, row: sqlite3.Row) -> IdentityRecord:
    """Convert a DB row into an IdentityRecord."""
    dim = int(row['embedding_dim'])
    return IdentityRecord(
      id=int(row['id']),
      name=str(row['name']),
      accessLevel=AccessLevel(int(row['access_level'])),
      imagePath=str(row['image_path']),
      embedding=blobToEmbedding(row['embedding'], dim),
      createdAt=str(row['created_at']),
      updatedAt=str(row['updated_at']),
    )

  def bestMatch(
    self,
    probeEmbedding: np.ndarray,
    matchThreshold: float = 0.65,
  ) -> tuple[IdentityRecord | None, float]:
    """Return (best identity or None, cosine). None if below threshold."""
    vector = np.asarray(probeEmbedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
      raise ValueError('Probe embedding norm is ~0')
    vector = vector / norm

    identities = self.listIdentities()
    if not identities:
      return None, -1.0

    best: IdentityRecord | None = None
    bestCosine = -1.0
    for identity in identities:
      cosine = float(np.dot(vector, identity.embedding))
      if cosine > bestCosine:
        bestCosine = cosine
        best = identity

    if best is None or bestCosine < matchThreshold:
      return None, bestCosine
    return best, bestCosine

  def evaluateAccess(
    self,
    probeEmbedding: np.ndarray,
    requiredLevel: AccessLevel = AccessLevel.ADMIN,
    matchThreshold: float = 0.65,
  ) -> MatchResult:
    """Match probe embed, then decide stranger / insufficient / granted."""
    identity, cosine = self.bestMatch(probeEmbedding, matchThreshold=matchThreshold)
    score = alikeScore(cosine)

    if identity is None:
      return MatchResult(
        decision=DECISION_STRANGER,
        identity=None,
        cosine=cosine,
        score=score,
        requiredLevel=requiredLevel,
        message='ACCESS DENIED — stranger (not registered)',
      )

    if identity.accessLevel < requiredLevel:
      return MatchResult(
        decision=DECISION_INSUFFICIENT,
        identity=identity,
        cosine=cosine,
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
      cosine=cosine,
      score=score,
      requiredLevel=requiredLevel,
      message=(
        f'ACCESS GRANTED — {identity.name} '
        f'({identity.accessName})'
      ),
    )
