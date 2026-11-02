#!/usr/bin/env python3
"""Train the face encoder (MATCH / NOT MATCH pairs → cosine-ready embeds)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from facelock.config import Paths, TrainConfig
from facelock.training import runTraining


def parseArgs() -> argparse.Namespace:
  """CLI for identity encoder training."""
  defaults = TrainConfig()
  paths = Paths()
  parser = argparse.ArgumentParser(description='Train FaceLock encoder')
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
  parser.add_argument('--image-size', type=int, default=defaults.imageSize)
  parser.add_argument('--aug-size', type=int, default=defaults.augSize)
  parser.add_argument('--embed-dim', type=int, default=defaults.embedDim)
  parser.add_argument('--batch-size', type=int, default=defaults.batchSize)
  parser.add_argument('--epochs', type=int, default=defaults.epochs)
  parser.add_argument('--lr', type=float, default=defaults.learningRate)
  parser.add_argument('--weight-decay', type=float, default=defaults.weightDecay)
  parser.add_argument('--num-workers', type=int, default=defaults.numWorkers)
  parser.add_argument('--seed', type=int, default=defaults.seed)
  parser.add_argument('--val-images', type=int, default=defaults.valImages)
  parser.add_argument('--save-every', type=int, default=defaults.saveEvery)
  parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
  parser.add_argument('--resume', type=str, default='')
  return parser.parse_args()


def main() -> None:
  """Run training and publish encoder_best.pt under models/."""
  args = parseArgs()
  config = TrainConfig(
    imageSize=args.image_size,
    augSize=args.aug_size,
    embedDim=args.embed_dim,
    batchSize=args.batch_size,
    epochs=args.epochs,
    learningRate=args.lr,
    weightDecay=args.weight_decay,
    numWorkers=args.num_workers,
    seed=args.seed,
    valImages=args.val_images,
    saveEvery=args.save_every,
  )
  outputDir = Path(args.output_dir)
  best = runTraining(
    datasetDir=Path(args.dataset_dir),
    outputDir=outputDir,
    config=config,
    deviceChoice=args.device,
    resume=Path(args.resume) if args.resume else None,
  )
  # Convenience copy at models/encoder_best.pt when training into models/.
  published = Paths().encoderBest
  published.parent.mkdir(parents=True, exist_ok=True)
  if best.resolve() != published.resolve():
    shutil.copy2(best, published)
    print(f'Published: {published.resolve()}')


if __name__ == '__main__':
  main()
