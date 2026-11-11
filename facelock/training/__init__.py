"""Training loops for pair-match and ArcFace encoders."""

from facelock.training.arcface import runArcFaceTraining
from facelock.training.loop import runTraining

__all__ = ['runArcFaceTraining', 'runTraining']
