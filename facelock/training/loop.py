"""Train MatchModel and export the encoder."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from facelock.config import TrainConfig
from facelock.data import FaceIndex, PairDataset, splitByIdentity
from facelock.device import resolveDevice, setSeed
from facelock.models import MatchModel
from facelock.models.checkpoint import exportEncoder as writeEncoderCheckpoint


def saveCheckpoint(
  model: MatchModel,
  optimizer: torch.optim.Optimizer,
  epoch: int,
  metrics: dict[str, float],
  config: TrainConfig,
  path: Path,
) -> None:
  """Persist full training state."""
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      'epoch': epoch,
      'metrics': metrics,
      'modelState': model.state_dict(),
      'optimizerState': optimizer.state_dict(),
      'config': asdict(config),
    },
    path,
  )


def runEpoch(
  model: MatchModel,
  loader: DataLoader,
  device: torch.device,
  optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
  """One train or eval epoch."""
  trainMode = optimizer is not None
  model.train(trainMode)

  totalLoss = 0.0
  headCorrect = 0
  cosineCorrect = 0
  totalCount = 0
  sameCount = 0
  diffCount = 0
  sameCosineSum = 0.0
  diffCosineSum = 0.0

  progress = tqdm(loader, desc='train' if trainMode else 'val', leave=False)
  for imageA, imageB, labels in progress:
    imageA = imageA.to(device, non_blocking=True)
    imageB = imageB.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if trainMode:
      optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(trainMode):
      logits, embedA, embedB = model(imageA, imageB)
      matchLoss = F.binary_cross_entropy_with_logits(logits, labels)
      embedA = F.normalize(embedA, p=2, dim=1)
      embedB = F.normalize(embedB, p=2, dim=1)
      cosine = (embedA * embedB).sum(dim=1)
      cosineTarget = labels * 2.0 - 1.0
      cosineLoss = F.mse_loss(cosine, cosineTarget)
      loss = matchLoss + cosineLoss
      if trainMode:
        loss.backward()
        optimizer.step()

    batchSize = labels.size(0)
    headPreds = (torch.sigmoid(logits) >= 0.5).float()
    cosinePreds = (cosine.detach() >= 0.0).float()
    totalLoss += loss.item() * batchSize
    headCorrect += (headPreds == labels).sum().item()
    cosineCorrect += (cosinePreds == labels).sum().item()
    totalCount += batchSize

    sameMask = labels == 1
    diffMask = labels == 0
    sameCount += int(sameMask.sum().item())
    diffCount += int(diffMask.sum().item())
    sameCosineSum += float(cosine.detach()[sameMask].sum().item())
    diffCosineSum += float(cosine.detach()[diffMask].sum().item())
    progress.set_postfix(loss=f'{loss.item():.3f}')

  return {
    'loss': totalLoss / max(totalCount, 1),
    'headMatchAcc': headCorrect / max(totalCount, 1),
    'cosineMatchAcc': cosineCorrect / max(totalCount, 1),
    'sameCosine': sameCosineSum / max(sameCount, 1),
    'diffCosine': diffCosineSum / max(diffCount, 1),
  }


def runTraining(
  datasetDir: Path,
  outputDir: Path,
  config: TrainConfig,
  deviceChoice: str = 'auto',
  resume: Path | None = None,
) -> Path:
  """Train the match model and return the path to encoder_best.pt."""
  if config.augSize < config.imageSize:
    raise ValueError('augSize must be >= imageSize')

  setSeed(config.seed)
  device = resolveDevice(deviceChoice)
  checkpointsDir = outputDir / 'checkpoints'
  checkpointsDir.mkdir(parents=True, exist_ok=True)
  with (outputDir / 'config.json').open('w', encoding='utf-8') as handle:
    json.dump(asdict(config), handle, indent=2)

  index = FaceIndex(datasetDir)
  trainAnchors, valAnchors, trainIds, valIds = splitByIdentity(
    index,
    config.valImages,
    config.seed,
  )

  valLoader = DataLoader(
    PairDataset(
      index,
      valAnchors,
      valIds,
      config.imageSize,
      pairSeed=config.seed + 1,
      augment=False,
      augSize=config.augSize,
    ),
    batch_size=config.batchSize,
    shuffle=False,
    num_workers=config.numWorkers,
    pin_memory=device.type == 'cuda',
  )

  model = MatchModel(embedDim=config.embedDim).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learningRate,
    weight_decay=config.weightDecay,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs,
  )

  startEpoch = 1
  bestValAcc = -1.0
  history: list[dict[str, float]] = []

  if resume is not None:
    checkpoint = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['modelState'])
    optimizer.load_state_dict(checkpoint['optimizerState'])
    startEpoch = int(checkpoint['epoch']) + 1
    bestValAcc = float(
      checkpoint.get('metrics', {}).get('valCosineMatchAcc', bestValAcc),
    )
    print(f'Resumed from {resume} at epoch {startEpoch - 1}')

  print(f'Device: {device}')
  print(f'Identities: train={len(trainIds)} val={len(valIds)}')
  print(f'Train images: {len(trainAnchors)}  val images: {len(valAnchors)}')
  print(
    f'Embed dim: {config.embedDim}  '
    f'image: {config.imageSize}  aug: {config.augSize}',
  )

  for epoch in range(startEpoch, config.epochs + 1):
    trainLoader = DataLoader(
      PairDataset(
        index,
        trainAnchors,
        trainIds,
        config.imageSize,
        pairSeed=config.seed + epoch * 17,
        augment=True,
        augSize=config.augSize,
      ),
      batch_size=config.batchSize,
      shuffle=True,
      num_workers=config.numWorkers,
      pin_memory=device.type == 'cuda',
      drop_last=True,
    )

    trainMetrics = runEpoch(model, trainLoader, device, optimizer)
    valMetrics = runEpoch(model, valLoader, device, optimizer=None)
    scheduler.step()

    metrics = {
      'epoch': float(epoch),
      'trainLoss': trainMetrics['loss'],
      'trainCosineMatchAcc': trainMetrics['cosineMatchAcc'],
      'valLoss': valMetrics['loss'],
      'valCosineMatchAcc': valMetrics['cosineMatchAcc'],
      'valHeadMatchAcc': valMetrics['headMatchAcc'],
      'valSameCosine': valMetrics['sameCosine'],
      'valDiffCosine': valMetrics['diffCosine'],
    }
    history.append(metrics)

    print(
      f'epoch {epoch:03d}/{config.epochs}  '
      f'train_loss={trainMetrics["loss"]:.4f}  '
      f'train_cosine={trainMetrics["cosineMatchAcc"] * 100:.2f}%  '
      f'val_loss={valMetrics["loss"]:.4f}  '
      f'val_cosine={valMetrics["cosineMatchAcc"] * 100:.2f}%  '
      f'cos(same)={valMetrics["sameCosine"]:.3f} '
      f'cos(diff)={valMetrics["diffCosine"]:.3f}  '
      f'lr={optimizer.param_groups[0]["lr"]:.2e}',
    )

    saveCheckpoint(
      model,
      optimizer,
      epoch,
      metrics,
      config,
      checkpointsDir / 'latest.pt',
    )
    writeEncoderCheckpoint(
      model.encoder,
      checkpointsDir / 'encoder_latest.pt',
      config.embedDim,
      config.imageSize,
      extraMeta={'pipeline': 'pair'},
    )

    if valMetrics['cosineMatchAcc'] >= bestValAcc:
      bestValAcc = valMetrics['cosineMatchAcc']
      saveCheckpoint(
        model,
        optimizer,
        epoch,
        metrics,
        config,
        checkpointsDir / 'best.pt',
      )
      bestEncoder = checkpointsDir / 'encoder_best.pt'
      writeEncoderCheckpoint(
        model.encoder,
        bestEncoder,
        config.embedDim,
        config.imageSize,
        extraMeta={'pipeline': 'pair'},
      )
      print(f'  new best val_cosine={bestValAcc * 100:.2f}% → {bestEncoder}')

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
