import numpy as np
import torch

from src.model import UNet
from src.checkpoint import load_checkpoint, save_checkpoint
from src.early_stopping import EarlyStopping
from src.loaders import get_dataloaders
from src.augmentation import data_transforms
from src.running import Runner, run_epoch
from src.tensorboard_tracker import TensorboardTracker
from src.timeit import timeit


# hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
LOAD_MODEL = False
PATH_TO_DATASET = "dataset.csv"
LOG_ROOTDIR = "./tensorboard_runs/"
INPUT_COLUMN_NAMES = ["images_vel_x", "images_vel_y"]
# INPUT_COLUMN_NAMES = ["images_vel_x"]
# INPUT_COLUMN_NAMES = ["images_z_vort"]
OUTPUT_COLUMN_NAME = "images_pressure"
USE_DATA_AUGMENTATION = False

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@timeit
def main():

    model = UNet(in_channels=3 * len(INPUT_COLUMN_NAMES))

    if USE_DATA_AUGMENTATION:
        transform_train = data_transforms
        transform_valid = data_transforms
    else:
        transform_train = data_transforms
        transform_valid = data_transforms

    train_loader, valid_loader, _ = get_dataloaders(
        PATH_TO_DATASET,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        input_column_names=INPUT_COLUMN_NAMES,
        output_column_name=OUTPUT_COLUMN_NAME,
        transform_train=transform_train,
        transform_valid=transform_valid,
        transform_test=data_transforms,
    )

    optimizer = torch.optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=40)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=13, factor=0.6, verbose=True
    )

    train_runner = Runner(train_loader, model, optimizer=optimizer)
    valid_runner = Runner(valid_loader, model)

    tracker = TensorboardTracker(log_dir=LOG_ROOTDIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_acc = np.inf

    if LOAD_MODEL:
        (
            epoch_from_previous_run,
            _,
            best_acc,
        ) = load_checkpoint(model=model, optimizer=optimizer, device=device)

        train_runner.epoch = epoch_from_previous_run
        valid_runner.epoch = epoch_from_previous_run

    for epoch in range(NUM_EPOCHS):
        epoch_loss, epoch_acc = run_epoch(
            train_runner=train_runner,
            valid_runner=valid_runner,
            tracker=tracker,
        )

        scheduler.step(epoch_acc)
        early_stopping(epoch_acc)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        should_save_model = best_acc > epoch_acc
        if should_save_model:
            best_acc = epoch_acc
            save_checkpoint(
                valid_runner.model, optimizer, valid_runner.epoch, epoch_loss, best_acc
            )
            print(f"Best acc: {epoch_acc} \t Best loss: {epoch_loss}")

        print(f"Epoch acc: {epoch_acc} \t Epoch loss: {epoch_loss}\n")


if __name__ == "__main__":
    main()
