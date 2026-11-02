"""Detection, embedding, tracking, and camera loop."""

from facelock.inference.camera import runCamera
from facelock.inference.embed import embedImagePath, loadEncoder

__all__ = ['embedImagePath', 'loadEncoder', 'runCamera']
