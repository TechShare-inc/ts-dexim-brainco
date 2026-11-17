"""Public SDK/native-free BrainCo supervised offline integration seam."""

from .config import OfflineConfig
from .controller import (
    ActionRecord,
    ControlReport,
    ObservationRecord,
    OfflineController,
    StepResult,
)
from .hand import load_replay, synthetic_hand
from .interface import OfflineMockInterface
from .model import ModelDescriptor, describe_model

__all__ = [
    "OfflineConfig",
    "OfflineController",
    "OfflineMockInterface",
    "ModelDescriptor",
    "describe_model",
    "synthetic_hand",
    "load_replay",
    "ActionRecord",
    "ObservationRecord",
    "ControlReport",
    "StepResult",
]
