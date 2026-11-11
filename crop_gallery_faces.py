#!/usr/bin/env python3
"""Haar-crop face cutouts from a folder of full photos (mugshots, etc.)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from facelock.inference.detect import DEFAULT_CASCADE, FaceDetector, cropBgr, expandBox

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def listImages(folder: Path) -> list[Path]:
  """Return image files in folder (non-recursive), sorted by name."""
  return sorted(
    [
      path
      for path in folder.iterdir()
      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
  )


def pickLargestBox(boxes) -> tuple[int, int, int, int] | None:
  """Choose the largest Haar box by area."""
  if boxes is None or len(boxes) == 0:
    return None
  areas = [int(w) * int(h) for (_x, _y, w, h) in boxes]
  index = int(max(range(len(areas)), key=lambda i: areas[i]))
  x, y, w, h = boxes[index]
  return int(x), int(y), int(w), int(h)


def cropOne(
  imagePath: Path,
  detector: FaceDetector,
  margin: float,
) -> tuple[object | None, str]:
  """Detect + crop one image. Returns (bgr_crop_or_None, status)."""
  bgr = cv2.imread(str(imagePath), cv2.IMREAD_COLOR)
  if bgr is None:
    return None, 'unreadable'
  height, width = bgr.shape[:2]
  boxes = detector.detect(bgr)
  box = pickLargestBox(boxes)
  if box is None:
    return None, 'no_face'
  x, y, w, h = box
  ex, ey, ew, eh = expandBox(x, y, w, h, width, height, margin=margin)
  crop = cropBgr(bgr, ex, ey, ew, eh)
  if crop.size == 0:
    return None, 'empty_crop'
  return crop, 'ok'


def parseArgs() -> argparse.Namespace:
  """CLI for batch face cropping."""
  parser = argparse.ArgumentParser(
    description='Haar-crop faces from a folder of full photos',
  )
  parser.add_argument('--input-dir', type=str, required=True)
  parser.add_argument('--output-dir', type=str, default='')
  parser.add_argument('--in-place', action='store_true')
  parser.add_argument(
    '--crop-margin',
    type=float,
    default=0.0,
    help='Expand Haar box by this fraction (default 0 = tight Haar box)',
  )
  parser.add_argument('--cascade', type=str, default=str(DEFAULT_CASCADE))
  parser.add_argument('--min-size', type=int, default=60)
  return parser.parse_args()


def main() -> None:
  """Crop every image in --input-dir."""
  args = parseArgs()
  inputDir = Path(args.input_dir)
  if not inputDir.is_dir():
    raise SystemExit(f'Not a directory: {inputDir}')

  if args.in_place:
    outputDir = inputDir
  else:
    if not args.output_dir:
      raise SystemExit('Pass --output-dir or --in-place')
    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

  images = listImages(inputDir)
  if not images:
    raise SystemExit(f'No images in {inputDir}')

  detector = FaceDetector(
    cascadePath=Path(args.cascade),
    minSize=(args.min_size, args.min_size),
  )

  ok = 0
  failed: list[tuple[str, str]] = []
  for imagePath in images:
    crop, status = cropOne(imagePath, detector, margin=float(args.crop_margin))
    if crop is None:
      failed.append((imagePath.name, status))
      print(f'  skip {imagePath.name} ({status})')
      continue
    dest = outputDir / imagePath.name
    if not cv2.imwrite(str(dest), crop):
      failed.append((imagePath.name, 'write_failed'))
      print(f'  fail {imagePath.name} (write_failed)')
      continue
    ok += 1
    print(f'  ok   {imagePath.name} → {dest}  ({crop.shape[1]}x{crop.shape[0]})')

  print(f'Done. cropped={ok}  skipped={len(failed)}  out={outputDir.resolve()}')
  if failed:
    raise SystemExit(1)


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
