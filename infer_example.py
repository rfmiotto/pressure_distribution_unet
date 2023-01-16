import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.model import UNet
from src.checkpoint import load_checkpoint


def visualize(transpose_channels=False, figsize=(30, 30), **images):
    """
    Helper function for data visualization
    PyTorch CHW tensor will be converted to HWC if `transpose_channels=True`
    """
    n_images = len(images)

    plt.figure(figsize=figsize)
    for idx, (key, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.title(key.replace("_", " ").title(), fontsize=12)
        if transpose_channels:
            plt.imshow(np.transpose(image, (1, 2, 0)))
        else:
            plt.imshow(image)
        plt.axis("off")
    plt.show()


def get_input_and_target_images():
    file_ = pd.read_csv("./dataset.csv")

    input_column_names = ["images_vel_x", "images_vel_y"]
    output_column_name = "images_pressure"

    input_img_paths = [file_[column][0] for column in input_column_names]

    input_images = []
    for img_path in input_img_paths:
        input_img = Image.open(img_path).convert("RGB")
        input_images.append(input_img)

    # input_img = Image.open(file_["images_z_vort"][0]).convert("RGB")
    output_img = Image.open(file_[output_column_name][0]).convert("RGB")

    return input_images, output_img


def parse_input(input_images: np.ndarray, device: torch.device):
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )

    for i, image in enumerate(input_images):
        input_images[i] = data_transform(image)
        input_images[i].unsqueeze_(0)
    input_images = torch.hstack(input_images)
    input_images = input_images.to(device)

    return input_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=6)
    model.to(device=device)

    load_checkpoint(model=model, device=device)

    input_img, output_img = get_input_and_target_images()
    input_img = parse_input(input_img, device)

    model.eval()
    with torch.no_grad():
        prediction = model(input_img)

    prediction_viz = prediction.cpu().detach().numpy()
    prediction_viz = np.transpose(prediction_viz, (0, 2, 3, 1))[0, :, :, :]
    # prediction_viz_new = np.clip(prediction_viz * 255.0, 0, 255)
    # prediction_viz_new = np.round(prediction_viz_new).astype(np.uint8)

    input_img_viz = input_img.cpu().detach().numpy()
    input_img_viz = np.transpose(input_img_viz, (0, 2, 3, 1))[0, :, :, :]

    visualize(
        input=input_img_viz[:, :, 3:6], true=output_img, prediction=prediction_viz
    )


if __name__ == "__main__":
    main()
