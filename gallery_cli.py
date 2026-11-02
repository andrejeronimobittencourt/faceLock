#!/usr/bin/env python3
"""CRUD for the face gallery (name, access, face image, embedding)."""

from __future__ import annotations

import argparse
from pathlib import Path

from facelock.config import MATCH_THRESHOLD, Paths
from facelock.device import resolveDevice
from facelock.gallery import (
  ACCESS_LEVEL_NAMES,
  DECISION_GRANTED,
  DECISION_INSUFFICIENT,
  DECISION_STRANGER,
  FaceGallery,
  IdentityRecord,
  parseAccessLevel,
)
from facelock.inference.embed import embedImagePath


def printIdentity(record: IdentityRecord) -> None:
  """Print one identity record."""
  print(f'id: {record.id}')
  print(f'name: {record.name}')
  print(f'access: {record.accessName}')
  print(f'face: {record.imagePath}')
  print(f'embed dim: {record.embedding.size}')


def requireRef(args: argparse.Namespace) -> None:
  """Require --id or --name."""
  if args.id is None and not args.name:
    raise SystemExit('Provide --id or --name')


def cmdCreate(args: argparse.Namespace) -> None:
  """Create identity (error if name exists)."""
  imagePath = Path(args.image)
  embedding = embedImagePath(
    Path(args.encoder),
    imagePath,
    resolveDevice(args.device),
  )
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    record = gallery.create(
      name=args.name,
      accessLevel=parseAccessLevel(args.access),
      sourceImage=imagePath,
      embedding=embedding,
    )
  print('Created:')
  printIdentity(record)


def cmdRegister(args: argparse.Namespace) -> None:
  """Create or replace identity by name."""
  imagePath = Path(args.image)
  embedding = embedImagePath(
    Path(args.encoder),
    imagePath,
    resolveDevice(args.device),
  )
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    record = gallery.register(
      name=args.name,
      accessLevel=parseAccessLevel(args.access),
      sourceImage=imagePath,
      embedding=embedding,
    )
  print('Registered:')
  printIdentity(record)


def cmdGet(args: argparse.Namespace) -> None:
  """Read one identity."""
  requireRef(args)
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    record = (
      gallery.getById(args.id) if args.id is not None else gallery.getByName(args.name)
    )
  if record is None:
    raise SystemExit('Not found.')
  printIdentity(record)


def cmdList(args: argparse.Namespace) -> None:
  """List identities."""
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    identities = gallery.listIdentities()
  if not identities:
    print('Gallery empty.')
    return
  print(f'{"ID":>4}  {"ACCESS":<10}  NAME')
  for identity in identities:
    print(f'{identity.id:>4}  {identity.accessName:<10}  {identity.name}')
  print(f'Total: {len(identities)}')


def cmdUpdate(args: argparse.Namespace) -> None:
  """Update name / access / face."""
  requireRef(args)
  if not args.new_name and not args.access and not args.image:
    raise SystemExit('Pass --new-name and/or --access and/or --image')

  embedding = None
  imagePath = None
  if args.image:
    imagePath = Path(args.image)
    embedding = embedImagePath(
      Path(args.encoder),
      imagePath,
      resolveDevice(args.device),
    )

  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    record = gallery.update(
      identityId=args.id,
      name=args.name,
      newName=args.new_name,
      accessLevel=parseAccessLevel(args.access) if args.access else None,
      sourceImage=imagePath,
      embedding=embedding,
    )
  print('Updated:')
  printIdentity(record)


def cmdDelete(args: argparse.Namespace) -> None:
  """Delete by id or name."""
  requireRef(args)
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    removed = gallery.delete(identityId=args.id, name=args.name)
  if not removed:
    raise SystemExit('Not found.')
  print(f'Deleted: {args.id if args.id is not None else args.name}')


def cmdCheck(args: argparse.Namespace) -> None:
  """Match an image against the gallery and evaluate clearance."""
  embedding = embedImagePath(
    Path(args.encoder),
    Path(args.image),
    resolveDevice(args.device),
  )
  with FaceGallery(Path(args.db), Path(args.faces_dir)) as gallery:
    result = gallery.evaluateAccess(
      embedding,
      requiredLevel=parseAccessLevel(args.required),
      matchThreshold=args.match_threshold,
    )
  print(result.message)
  print(f'  decision: {result.decision}')
  print(f'  score: {result.score}/100  cosine: {result.cosine:.4f}')
  print(f'  required: {ACCESS_LEVEL_NAMES[result.requiredLevel]}')
  if result.identity is not None:
    print(f'  matched: {result.identity.name} ({result.identity.accessName})')
  codes = {
    DECISION_GRANTED: 0,
    DECISION_INSUFFICIENT: 2,
    DECISION_STRANGER: 3,
  }
  raise SystemExit(codes.get(result.decision, 1))


def buildParser() -> argparse.ArgumentParser:
  """Gallery CRUD CLI."""
  paths = Paths()
  parser = argparse.ArgumentParser(description='Face gallery CRUD')
  parser.add_argument('--db', type=str, default=str(paths.galleryDb))
  parser.add_argument('--faces-dir', type=str, default=str(paths.galleryFaces))
  parser.add_argument('--encoder', type=str, default=str(paths.encoderBest))
  parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])

  sub = parser.add_subparsers(dest='command', required=True)

  createCmd = sub.add_parser('create', help='Create identity')
  createCmd.add_argument('--name', required=True)
  createCmd.add_argument('--access', required=True, help='user | operator | admin')
  createCmd.add_argument('--image', required=True)
  createCmd.set_defaults(func=cmdCreate)

  registerCmd = sub.add_parser('register', help='Create or replace by name')
  registerCmd.add_argument('--name', required=True)
  registerCmd.add_argument('--access', required=True)
  registerCmd.add_argument('--image', required=True)
  registerCmd.set_defaults(func=cmdRegister)

  getCmd = sub.add_parser('get', help='Get by --id or --name')
  getCmd.add_argument('--id', type=int, default=None)
  getCmd.add_argument('--name', type=str, default='')
  getCmd.set_defaults(func=cmdGet)

  listCmd = sub.add_parser('list', help='List identities')
  listCmd.set_defaults(func=cmdList)

  updateCmd = sub.add_parser('update', help='Update name / access / face')
  updateCmd.add_argument('--id', type=int, default=None)
  updateCmd.add_argument('--name', type=str, default='')
  updateCmd.add_argument('--new-name', default='')
  updateCmd.add_argument('--access', default='')
  updateCmd.add_argument('--image', default='')
  updateCmd.set_defaults(func=cmdUpdate)

  deleteCmd = sub.add_parser('delete', help='Delete by --id or --name')
  deleteCmd.add_argument('--id', type=int, default=None)
  deleteCmd.add_argument('--name', type=str, default='')
  deleteCmd.set_defaults(func=cmdDelete)

  checkCmd = sub.add_parser('check', help='Match image + clearance')
  checkCmd.add_argument('--image', required=True)
  checkCmd.add_argument('--required', default='admin')
  checkCmd.add_argument('--match-threshold', type=float, default=MATCH_THRESHOLD)
  checkCmd.set_defaults(func=cmdCheck)

  return parser


def main() -> None:
  """Dispatch gallery commands."""
  args = buildParser().parse_args()
  for key in ('new_name', 'access', 'image', 'name'):
    if getattr(args, key, None) == '':
      setattr(args, key, None)
  args.func(args)


if __name__ == '__main__':
  main()
