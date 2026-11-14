#!/usr/bin/env python3
"""Live webcam face recognition against the gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from facelock.config import Paths
from facelock.inference import runCamera


def parseArgs() -> argparse.Namespace:
  """CLI for the camera demo."""
  paths = Paths()
  parser = argparse.ArgumentParser(description='Webcam face recognition')
  parser.add_argument('--camera', type=int, default=0)
  parser.add_argument('--required', type=str, default='admin')
  parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
  parser.add_argument('--encoder', type=str, default=str(paths.encoderBest))
  parser.add_argument('--db', type=str, default=str(paths.galleryDb))
  parser.add_argument('--faces-dir', type=str, default=str(paths.galleryFaces))
  parser.add_argument('--no-mirror', action='store_true')
  return parser.parse_args()


def main() -> None:
  """Start the camera recognition loop."""
  args = parseArgs()
  runCamera(
    cameraIndex=args.camera,
    encoderPath=Path(args.encoder),
    dbPath=Path(args.db),
    facesDir=Path(args.faces_dir),
    required=args.required,
    deviceChoice=args.device,
    mirror=not args.no_mirror,
  )


if __name__ == '__main__':
  main()
