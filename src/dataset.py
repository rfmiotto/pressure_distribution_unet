from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

# import torch
# import numpy as np
# from torchvision.io import read_image


class Image2ImageDataset(Dataset):
    """
    The CSV file must have columns with the name of the properties of interest
    (both input and output). Ex:

    property_tom           property_dick           property_harry
    /input/path/img1.jpg   /output/path/img1.jpg   /output/path/img1.jpg
    /input/path/img2.jpg   /output/path/img2.jpg   /output/path/img2.jpg
    .                      .                       .
    .                      .                       .

    Args:
        csv_file (string): Path to the csv file with image paths
        transform (callable, optional): Optional transform to be applied
            on a sample
    """

    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        output_column_name: str,
        transform=None,
    ):
        self.files = pd.read_csv(csv_file)
        self.transform = transform
        self.input_column_names = input_column_names
        self.output_column_name = output_column_name

    def preprocess(self, output):
        output = transforms.Resize((256, 256))(output)
        output = transforms.functional.pil_to_tensor(output) / 255.0
        return output

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_paths = [
            self.files[column][index] for column in self.input_column_names
        ]

        input_images = []
        for img_path in input_img_paths:
            input_img = Image.open(img_path).convert("RGB")
            input_images.append(input_img)

        output_img_path = self.files[self.output_column_name][index]
        output_img = Image.open(output_img_path).convert("RGB")

        if self.transform:
            for i, image in enumerate(input_images):
                input_images[i] = self.transform(image)
            # output_img = self.transform(output_img)
            output_img = self.preprocess(output_img)

        return input_images, output_img


# def _test():
#     import matplotlib.pyplot as plt
#     from torchvision import transforms
#     import numpy as np

#     WIDTH = 256
#     HEIGHT = 256

#     data_transforms = transforms.Compose(
#         [
#             transforms.Resize((HEIGHT, WIDTH)),
#             transforms.ToTensor(),
#         ]
#     )

#     dataset = Image2ImageDataset(
#         "./dataset.csv",
#         input_column_names=["images_z_vort"],
#         output_column_name="images_pressure",
#     )

#     input_img, output_img = dataset[0]

#     input_img_transformed = data_transforms(input_img)
#     input_img_transformed = np.transpose(input_img_transformed, (1, 2, 0))

#     plt.figure()
#     plt.imshow(input_img)

#     plt.figure()
#     plt.imshow(input_img_transformed)

#     plt.show()


# if __name__ == "__main__":
#     _test()
