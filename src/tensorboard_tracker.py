import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.tracking import Stage


class TensorboardTracker:
    def __init__(self, log_dir: str, filename: str = "", create: bool = True):
        self._validate_log_dir(log_dir, create=create)
        self.stage = Stage.TRAIN
        default_name = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        filename = filename if filename else default_name
        self._writer = SummaryWriter(log_dir + "/" + filename)

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        path = Path(log_dir).resolve()
        if path.exists():
            return
        if not path.exists() and create:
            path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def get_stage(self):
        return self.stage

    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self._writer.flush()

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)
