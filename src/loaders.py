from typing import List, Any
import torch
from torch.utils.data.dataloader import DataLoader
from src.dataset import Image2ImageDataset


def get_dataloaders(
    path_to_dataset,
    num_workers: int,
    batch_size: int,
    input_column_names: List[str],
    output_column_name: str,
    transform_train: Any,
    transform_valid: Any,
    transform_test: Any,
) -> List[DataLoader]:
    dataset = Image2ImageDataset(
        path_to_dataset,
        input_column_names=input_column_names,
        output_column_name=output_column_name,
    )

    train_set = dataset
    valid_set = dataset
    test_set = dataset

    train_set.transform = transform_train
    valid_set.transform = transform_valid
    test_set.transform = transform_test

    train_proportion = 0.8
    valid_proportion = 0.1
    # test proportion is the remaining to 1

    train_size = int(train_proportion * len(train_set))
    valid_size = int(valid_proportion * len(train_set))

    indices = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(42)
    )

    indices_train = indices[:train_size].tolist()
    indices_valid = indices[train_size : (train_size + valid_size)].tolist()
    indices_test = indices[(train_size + valid_size) :].tolist()

    train_set = torch.utils.data.Subset(train_set, indices_train)
    valid_set = torch.utils.data.Subset(valid_set, indices_valid)
    test_set = torch.utils.data.Subset(test_set, indices_test)

    print("# of training data", len(train_set))
    print("# of validation data", len(valid_set))
    print("# of test data", len(test_set))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, valid_loader, test_loader
