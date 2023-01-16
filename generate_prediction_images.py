import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.model import UNet
from src.checkpoint import load_checkpoint
from src.dataset import Image2ImageDataset

IMG_SIZE = 256
NUM_WORKERS = 4
PATH_TO_DATASET = "./dataset.csv"
INPUT_COLUMN_NAMES = ["images_vel_x", "images_vel_y"]
OUTPUT_COLUMN_NAME = "images_pressure"
PATH_TO_SAVED_MODEL = "./my_checkpoint.pth.tar"
PATH_TO_SAVE_PREDICTED_IMAGES = "./outputs"
SELECTED_INDICES = []  # Leave this array empty to select all frames

def get_dataloader():
    data_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )

    dataset = Image2ImageDataset(
        PATH_TO_DATASET, INPUT_COLUMN_NAMES, OUTPUT_COLUMN_NAME, data_transform
    )

    if SELECTED_INDICES:
        dataset = torch.utils.data.Subset(dataset, SELECTED_INDICES)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=6)
    model.to(device=device)

    load_checkpoint(model=model, device=device)

    dataloader = get_dataloader()

    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, total=num_batches)

    output_path_exists = os.path.exists(PATH_TO_SAVE_PREDICTED_IMAGES)
    if not output_path_exists:
        os.makedirs(PATH_TO_SAVE_PREDICTED_IMAGES)

    model.eval()
    with torch.no_grad():
        for batch_index, (input_img, _) in enumerate(progress_bar):

            input_img = torch.hstack(input_img)
            input_img = input_img.to(device=device)

            prediction = model(input_img)

            save_image(
                prediction,
                os.path.join(PATH_TO_SAVE_PREDICTED_IMAGES, f"{batch_index:04d}.png"),
            )


if __name__ == "__main__":
    main()
