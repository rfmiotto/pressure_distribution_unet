from enum import Enum, auto
from typing import Protocol


class Stage(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class NetworkTracker(Protocol):
    def get_stage(self, stage: Stage):
        """Get the current stage of the network"""

    def set_stage(self, stage: Stage):
        """Set the current stage of the network"""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric"""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging an epoch-level metric"""
