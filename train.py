#!/usr/bin/env python3
"""Train the face encoder (pair-match or ArcFace IResNet)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from facelock.config import ArcFaceTrainConfig, Paths, TrainConfig
from facelock.models.iresnet import BACKBONE_NAMES
from facelock.training import runArcFaceTraining, runTraining


def parseArgs() -> argparse.Namespace:
  """CLI for FaceLock encoder training."""
  pairDefaults = TrainConfig()
  arcDefaults = ArcFaceTrainConfig()
  paths = Paths()
  parser = argparse.ArgumentParser(description='Train FaceLock encoder')
  parser.add_argument(
    '--method',
    type=str,
    default='pair',
    choices=['pair', 'arcface'],
    help='Training method (default: pair)',
  )
  parser.add_argument(
    '--dataset-dir',
    type=str,
    default=str(paths.dataDir),
    help='Root of identity folders (one folder per person)',
  )
  parser.add_argument(
    '--output-dir',
    type=str,
    default=str(paths.modelsDir),
    help='Where to write checkpoints and history',
  )
  parser.add_argument('--batch-size', type=int, default=None)
  parser.add_argument('--epochs', type=int, default=None)
  parser.add_argument('--lr', type=float, default=None)
  parser.add_argument('--num-workers', type=int, default=None)
  parser.add_argument('--seed', type=int, default=pairDefaults.seed)
  parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
  parser.add_argument('--resume', type=str, default='')
  parser.add_argument(
    '--backbone',
    type=str,
    default=arcDefaults.backbone,
    choices=list(BACKBONE_NAMES),
    help='IResNet size when --method arcface (default: ir50)',
  )
  return parser.parse_args()


def _orDefault(value, default):
  return default if value is None else value


def main() -> None:
  """Run training and publish encoder_best.pt under models/."""
  args = parseArgs()
  outputDir = Path(args.output_dir)
  resume = Path(args.resume) if args.resume else None

  if args.method == 'pair':
    defaults = TrainConfig()
    config = TrainConfig(
      batchSize=_orDefault(args.batch_size, defaults.batchSize),
      epochs=_orDefault(args.epochs, defaults.epochs),
      learningRate=_orDefault(args.lr, defaults.learningRate),
      numWorkers=_orDefault(args.num_workers, defaults.numWorkers),
      seed=args.seed,
    )
    best = runTraining(
      datasetDir=Path(args.dataset_dir),
      outputDir=outputDir,
      config=config,
      deviceChoice=args.device,
      resume=resume,
    )
  else:
    defaults = ArcFaceTrainConfig()
    config = ArcFaceTrainConfig(
      backbone=args.backbone,
      batchSize=_orDefault(args.batch_size, defaults.batchSize),
      epochs=_orDefault(args.epochs, defaults.epochs),
      learningRate=_orDefault(args.lr, defaults.learningRate),
      numWorkers=_orDefault(args.num_workers, defaults.numWorkers),
      seed=args.seed,
    )
    best = runArcFaceTraining(
      datasetDir=Path(args.dataset_dir),
      outputDir=outputDir,
      config=config,
      deviceChoice=args.device,
      resume=resume,
    )

  published = Paths().encoderBest
  published.parent.mkdir(parents=True, exist_ok=True)
  if best.resolve() != published.resolve():
    shutil.copy2(best, published)
    print(f'Published: {published.resolve()}')


if __name__ == '__main__':
  main()
