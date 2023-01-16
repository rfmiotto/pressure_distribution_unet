import torch
from torchvision import transforms

WIDTH = 256
HEIGHT = 256


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


data_transforms = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        # transforms.RandomApply([AddGaussianNoise(0.0, 0.001)], p=0.5),
    ]
)
