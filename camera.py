#!/usr/bin/env python3
"""Live webcam face recognition against the gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from facelock.config import (
  CONFIRM_HITS,
  LOST_GRACE_FRAMES,
  MATCH_THRESHOLD,
  SESSION_MEMORY_FRAMES,
  VERIFY_MISSES,
  VERIFY_THRESHOLD,
  Paths,
)
from facelock.inference import runCamera


def parseArgs() -> argparse.Namespace:
  """CLI for the camera demo."""
  paths = Paths()
  parser = argparse.ArgumentParser(description='Webcam face recognition')
  parser.add_argument('--camera', type=int, default=0)
  parser.add_argument('--encoder', type=str, default=str(paths.encoderBest))
  parser.add_argument('--db', type=str, default=str(paths.galleryDb))
  parser.add_argument('--faces-dir', type=str, default=str(paths.galleryFaces))
  parser.add_argument('--cascade', type=str, default=str(paths.cascade))
  parser.add_argument('--required', type=str, default='admin')
  parser.add_argument('--match-threshold', type=float, default=MATCH_THRESHOLD)
  parser.add_argument('--verify-threshold', type=float, default=VERIFY_THRESHOLD)
  parser.add_argument('--verify-misses', type=int, default=VERIFY_MISSES)
  parser.add_argument('--confirm-hits', type=int, default=CONFIRM_HITS)
  parser.add_argument('--lost-grace-frames', type=int, default=LOST_GRACE_FRAMES)
  parser.add_argument(
    '--session-memory-frames',
    type=int,
    default=SESSION_MEMORY_FRAMES,
  )
  parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
  parser.add_argument('--infer-every', type=int, default=2)
  parser.add_argument(
    '--crop-margin',
    type=float,
    default=0.0,
    help='Expand Haar box before embedding (0 = tight box)',
  )
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
    cascadePath=Path(args.cascade),
    required=args.required,
    matchThreshold=args.match_threshold,
    verifyThreshold=args.verify_threshold,
    verifyMisses=max(1, args.verify_misses),
    confirmHits=max(1, args.confirm_hits),
    lostGraceFrames=max(0, args.lost_grace_frames),
    sessionMemoryFrames=max(0, args.session_memory_frames),
    deviceChoice=args.device,
    inferEvery=max(1, args.infer_every),
    cropMargin=args.crop_margin,
    mirror=not args.no_mirror,
  )


if __name__ == '__main__':
  main()
