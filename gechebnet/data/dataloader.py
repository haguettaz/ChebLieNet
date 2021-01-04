import math
from typing import Optional, Tuple, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, STL10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    ToTensor,
)

from ..utils import shuffle_tensor

T_co = TypeVar("T_co", covariant=True)

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
STL10_MEAN, STL10_STD = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)


def get_train_val_data_loaders(
    dataset_name: str,
    batch_size: Optional[int] = 32,
    val_ratio: Optional[float] = 0.2,
    data_path: Optional[str] = "data",
) -> Tuple[DataLoader, DataLoader]:
    """
    [summary]

    Args:
        dataset_name (str): [description]
        batch_size (Optional[int], optional): [description]. Defaults to 32.
        val_ratio (Optional[float], optional): [description]. Defaults to 0.2.
        data_path (Optional[str], optional): [description]. Defaults to "data".

    Raises:
        ValueError: [description]

    Returns:
        Tuple[DataLoader, DataLoader]: [description]
    """

    if dataset_name not in {"MNIST", "STL10"}:
        raise ValueError(f"{dataset_name} is not a valid value for dataset_name: must be in 'MNIST', 'STL10'")

    if dataset_name == "MNIST":
        dataset = MNIST(
            data_path, train=True, download=True, transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)])
        )

    elif dataset_name == "STL10":
        dataset = STL10(
            data_path, split="train", download=True, transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD)])
        )

    N = len(dataset)
    split = math.floor(val_ratio * N)
    indices = shuffle_tensor(torch.arange(N))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_equivariance_data_loader(
    dataset_name: str, batch_size: Optional[int] = 32, data_path: Optional[str] = "data"
) -> DataLoader:
    """
    [summary]

    Args:
        dataset_name (str): [description]
        batch_size (Optional[int], optional): [description]. Defaults to 32.
        data_path (Optional[str], optional): [description]. Defaults to "data".

    Raises:
        ValueError: [description]

    Returns:
        DataLoader: [description]
    """

    if dataset_name not in {"MNIST", "STL10"}:
        raise ValueError(f"{dataset_name} is not a valid value for dataset_name: must be in 'MNIST', 'STL10'")

    if dataset_name == "MNIST":

        classic_dataset = MNIST(
            data_path, train=False, download=True, transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)])
        )

        rotated_dataset = MNIST(
            data_path,
            train=False,
            download=True,
            transform=Compose([RandomRotation(degrees=180), ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)]),
        )

        flipped_dataset = MNIST(
            data_path,
            train=False,
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    Normalize(MNIST_MEAN, MNIST_STD),
                ]
            ),
        )

    elif dataset_name == "STL10":

        classic_dataset = STL10(
            data_path, split="test", download=True, transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD)])
        )

        rotated_dataset = STL10(
            data_path,
            split="test",
            download=True,
            transform=Compose([RandomRotation(degrees=180), ToTensor(), Normalize(STL10_MEAN, STL10_STD)]),
        )

        flipped_dataset = STL10(
            data_path,
            split="test",
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    Normalize(STL10_MEAN, STL10_STD),
                ]
            ),
        )

    classic_loader = DataLoader(classic_dataset, batch_size=batch_size)
    rotated_loader = DataLoader(rotated_dataset, batch_size=batch_size)
    flipped_loader = DataLoader(flipped_dataset, batch_size=batch_size)

    return classic_loader, rotated_loader, flipped_loader
