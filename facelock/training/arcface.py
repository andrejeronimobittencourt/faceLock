"""Train IResNet + ArcFace."""


from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from facelock.config import ArcFaceTrainConfig
from facelock.data import FaceIndex, PairDataset, splitByIdentity
from facelock.data.identity import (
  CENTERED_NORMALIZE,
  IdentityImageDataset,
  buildArcFaceEvalTransform,
  buildArcFaceTrainTransform,
)
from facelock.device import resolveDevice, setSeed
from facelock.models.arcface import ArcFaceModel
from facelock.models.checkpoint import exportEncoder
from facelock.models.iresnet import BACKBONE_NAMES


def setEpochLearningRate(
  optimizer: torch.optim.Optimizer,
  baseLr: float,
  epoch: int,
  warmupEpochs: int,
  totalEpochs: int,
) -> float:
  """Linear warmup then cosine decay."""
  if warmupEpochs > 0 and epoch <= warmupEpochs:
    lr = baseLr * float(epoch) / float(warmupEpochs)
  else:
    progress = (epoch - warmupEpochs) / max(totalEpochs - warmupEpochs, 1)
    lr = baseLr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
  for group in optimizer.param_groups:
    group['lr'] = lr
  return lr


def runClassificationEpoch(
  model: ArcFaceModel,
  loader: DataLoader,
  device: torch.device,
  optimizer: torch.optim.Optimizer | None,
  gradClip: float = 0.0,
) -> dict[str, float]:
  """One ArcFace train epoch (CE on margin logits)."""
  trainMode = optimizer is not None
  model.train(trainMode)
  totalLoss = 0.0
  correct = 0
  correctTop5 = 0
  totalCount = 0
  progress = tqdm(loader, desc='train', leave=False)
  for images, labels in progress:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if trainMode:
      optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(trainMode):
      logits, _ = model(images, labels)
      loss = F.cross_entropy(logits, labels)
      if trainMode:
        loss.backward()
        if gradClip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradClip)
        optimizer.step()
    batchSize = labels.size(0)
    totalLoss += loss.item() * batchSize
    correct += (logits.argmax(dim=1) == labels).sum().item()
    topk = min(5, logits.size(1))
    correctTop5 += (logits.topk(topk, dim=1).indices == labels.unsqueeze(1)).any(
      dim=1,
    ).sum().item()
    totalCount += batchSize
    progress.set_postfix(loss=f'{loss.item():.3f}')
  return {
    'loss': totalLoss / max(totalCount, 1),
    'classAcc': correct / max(totalCount, 1),
    'classAccTop5': correctTop5 / max(totalCount, 1),
  }


@torch.no_grad()
def runPairValEpoch(
  model: ArcFaceModel,
  loader: DataLoader,
  device: torch.device,
) -> dict[str, float]:
  """Cosine match metrics on held-out identity pairs."""
  model.eval()
  cosineCorrect = 0
  totalCount = 0
  sameCount = 0
  diffCount = 0
  sameCosineSum = 0.0
  diffCosineSum = 0.0
  for imageA, imageB, labels in tqdm(loader, desc='val', leave=False):
    imageA = imageA.to(device, non_blocking=True)
    imageB = imageB.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    embedA = F.normalize(model.encoder(imageA), p=2, dim=1)
    embedB = F.normalize(model.encoder(imageB), p=2, dim=1)
    cosine = (embedA * embedB).sum(dim=1)
    preds = (cosine >= 0.0).float()
    batchSize = labels.size(0)
    cosineCorrect += (preds == labels).sum().item()
    totalCount += batchSize
    sameMask = labels == 1
    diffMask = labels == 0
    sameCount += int(sameMask.sum().item())
    diffCount += int(diffMask.sum().item())
    sameCosineSum += float(cosine[sameMask].sum().item())
    diffCosineSum += float(cosine[diffMask].sum().item())
  sameCosine = sameCosineSum / max(sameCount, 1)
  diffCosine = diffCosineSum / max(diffCount, 1)
  return {
    'cosineMatchAcc': cosineCorrect / max(totalCount, 1),
    'sameCosine': sameCosine,
    'diffCosine': diffCosine,
    'cosineGap': sameCosine - diffCosine,
  }


def saveCheckpoint(
  model: ArcFaceModel,
  optimizer: torch.optim.Optimizer,
  epoch: int,
  metrics: dict[str, float],
  config: ArcFaceTrainConfig,
  path: Path,
) -> None:
  """Persist full ArcFace training state."""
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      'epoch': epoch,
      'metrics': metrics,
      'modelState': model.state_dict(),
      'optimizerState': optimizer.state_dict(),
      'config': asdict(config),
      'pipeline': 'arcface',
    },
    path,
  )


def runArcFaceTraining(
  datasetDir: Path,
  outputDir: Path,
  config: ArcFaceTrainConfig,
  deviceChoice: str = 'auto',
  resume: Path | None = None,
) -> Path:
  """Train ArcFace on IResNet and return path to encoder_best.pt."""
  if config.backbone not in BACKBONE_NAMES:
    raise ValueError(
      f'Unknown backbone {config.backbone!r}. Choose from {list(BACKBONE_NAMES)}',
    )
  if config.augSize < config.imageSize:
    raise ValueError('augSize must be >= imageSize')

  setSeed(config.seed)
  device = resolveDevice(deviceChoice)
  checkpointsDir = outputDir / 'checkpoints'
  checkpointsDir.mkdir(parents=True, exist_ok=True)

  exportMeta = {
    'pipeline': 'arcface',
    'backbone': config.backbone,
    'inputNormalize': CENTERED_NORMALIZE,
  }
  with (outputDir / 'config.json').open('w', encoding='utf-8') as handle:
    json.dump({**asdict(config), **exportMeta}, handle, indent=2)

  index = FaceIndex(datasetDir)
  trainAnchors, valAnchors, trainIds, valIds = splitByIdentity(
    index,
    config.valImages,
    config.seed,
  )
  labelById = {int(idIndex): label for label, idIndex in enumerate(trainIds.tolist())}
  numClasses = len(labelById)

  trainTransform = buildArcFaceTrainTransform(config.imageSize, config.augSize)
  evalTransform = buildArcFaceEvalTransform(config.imageSize)
  valLoader = DataLoader(
    PairDataset(
      index,
      valAnchors,
      valIds,
      config.imageSize,
      pairSeed=config.seed + 1,
      augment=False,
      augSize=config.augSize,
      transform=evalTransform,
    ),
    batch_size=config.batchSize,
    shuffle=False,
    num_workers=config.numWorkers,
    pin_memory=device.type == 'cuda',
  )

  model = ArcFaceModel(
    numClasses=numClasses,
    backbone=config.backbone,
    embedDim=config.embedDim,
    scale=config.arcScale,
    margin=config.arcMargin,
    dropout=config.dropout,
  ).to(device)
  optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config.learningRate,
    momentum=config.momentum,
    weight_decay=config.weightDecay,
  )

  startEpoch = 1
  bestGap = float('-inf')
  history: list[dict[str, float]] = []

  if resume is not None:
    checkpoint = torch.load(resume, map_location=device, weights_only=False)
    savedConfig = checkpoint.get('config', {})
    savedClasses = int(checkpoint['modelState']['margin.weight'].shape[0])
    if savedClasses != numClasses:
      raise ValueError(
        f'Resume checkpoint has {savedClasses} classes; dataset has {numClasses}',
      )
    model = ArcFaceModel(
      numClasses=savedClasses,
      backbone=str(savedConfig.get('backbone', config.backbone)),
      embedDim=int(savedConfig.get('embedDim', config.embedDim)),
      scale=float(savedConfig.get('arcScale', config.arcScale)),
      margin=float(savedConfig.get('arcMargin', config.arcMargin)),
      dropout=float(savedConfig.get('dropout', config.dropout)),
    ).to(device)
    model.load_state_dict(checkpoint['modelState'])
    optimizer.load_state_dict(checkpoint['optimizerState'])
    startEpoch = int(checkpoint['epoch']) + 1
    bestGap = float(checkpoint.get('metrics', {}).get('valCosineGap', bestGap))
    print(f'Resumed from {resume} at epoch {startEpoch - 1}')

  print(f'Device: {device}')
  print(
    f'Pipeline: arcface  backbone={config.backbone}  '
    f'classes={numClasses}  embed={config.embedDim}  '
    f'scale={config.arcScale}  margin={config.arcMargin}',
  )
  print(f'Identities: train={len(trainIds)} val={len(valIds)}')
  print(f'Train images: {len(trainAnchors)}  val images: {len(valAnchors)}')

  for epoch in range(startEpoch, config.epochs + 1):
    lr = setEpochLearningRate(
      optimizer,
      config.learningRate,
      epoch,
      config.warmupEpochs,
      config.epochs,
    )
    trainLoader = DataLoader(
      IdentityImageDataset(index, trainIds, labelById, trainTransform),
      batch_size=config.batchSize,
      shuffle=True,
      num_workers=config.numWorkers,
      pin_memory=device.type == 'cuda',
      drop_last=True,
    )
    trainMetrics = runClassificationEpoch(
      model,
      trainLoader,
      device,
      optimizer,
      gradClip=config.gradClip,
    )
    valMetrics = runPairValEpoch(model, valLoader, device)
    gap = float(valMetrics['cosineGap'])
    metrics = {
      'epoch': float(epoch),
      'trainLoss': trainMetrics['loss'],
      'trainClassAcc': trainMetrics['classAcc'],
      'trainClassAccTop5': trainMetrics['classAccTop5'],
      'valCosineMatchAcc': valMetrics['cosineMatchAcc'],
      'valSameCosine': valMetrics['sameCosine'],
      'valDiffCosine': valMetrics['diffCosine'],
      'valCosineGap': gap,
      'lr': lr,
    }
    history.append(metrics)

    print(
      f'epoch {epoch:03d}/{config.epochs}  '
      f'loss={trainMetrics["loss"]:.4f}  '
      f'train_acc={trainMetrics["classAcc"] * 100:.2f}%  '
      f'train_top5={trainMetrics["classAccTop5"] * 100:.2f}%  '
      f'val_cosine={valMetrics["cosineMatchAcc"] * 100:.2f}%  '
      f'cos(same)={valMetrics["sameCosine"]:.3f}  '
      f'cos(diff)={valMetrics["diffCosine"]:.3f}  '
      f'gap={gap:.3f}  '
      f'lr={lr:.2e}',
    )

    saveCheckpoint(
      model,
      optimizer,
      epoch,
      metrics,
      config,
      checkpointsDir / 'latest.pt',
    )
    exportEncoder(
      model.encoder,
      checkpointsDir / 'encoder_latest.pt',
      config.embedDim,
      config.imageSize,
      extraMeta=exportMeta,
    )

    if gap > bestGap + 1e-6:
      bestGap = gap
      saveCheckpoint(
        model,
        optimizer,
        epoch,
        metrics,
        config,
        checkpointsDir / 'best.pt',
      )
      exportEncoder(
        model.encoder,
        checkpointsDir / 'encoder_best.pt',
        config.embedDim,
        config.imageSize,
        extraMeta=exportMeta,
      )
      print(f'  new best gap={bestGap:.3f} → encoder_best.pt')

    if epoch % config.saveEvery == 0:
      saveCheckpoint(
        model,
        optimizer,
        epoch,
        metrics,
        config,
        checkpointsDir / f'epoch_{epoch:03d}.pt',
      )

  with (outputDir / 'history.json').open('w', encoding='utf-8') as handle:
    json.dump(history, handle, indent=2)

  bestPath = checkpointsDir / 'encoder_best.pt'
  print(f'Done. Best encoder: {bestPath.resolve()}')
  return bestPath
