import math
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10, MNIST, STL10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    ToTensor,
)

from ..utils import shuffle_tensor

# mean and std per channels on the training sets
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
STL10_MEAN, STL10_STD = (0.4472, 0.4396, 0.4050), (0.2606, 0.2567, 0.2700)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_train_val_dataloaders(
    dataset: str,
    batch_size: Optional[int] = 32,
    val_ratio: Optional[float] = 0.2,
    data_path: Optional[str] = "data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Gets train and validation dataloaders.

    Args:
        dataset (str): name of the dataset.
        batch_size (int, optional): size of a batch. Defaults to 32.
        val_ratio (float, optional): ratio of validation samples. Defaults to 0.2.
        data_path (str, optional): path to data folder to download dataset into. Defaults to "data".

    Raises:
        ValueError: dataset has to be 'mnist' or 'stl10',

    Returns:
        Tuple[DataLoader, DataLoader]: train and validation dataloaders.
    """

    if dataset not in {"mnist", "stl10", "cifar10"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be 'mnist', 'stl10' or 'cifar10'.")

    if dataset == "mnist":
        dataset = MNIST(
            data_path,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)]),
        )

    elif dataset == "cifar10":
        dataset = CIFAR10(
            data_path,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]),
        )

    elif dataset == "stl10":
        dataset = STL10(
            data_path,
            split="test",
            download=True,
            transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD)]),
        )

    N = len(dataset)
    split = math.floor(val_ratio * N)
    indices = shuffle_tensor(torch.arange(N))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def get_test_equivariance_dataloaders(
    dataset: str, batch_size: Optional[int] = 32, data_path: Optional[str] = "data"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gets test dataloaders to test equivariance under rotation and flip property of the network.

    Args:
        dataset (str): name of the dataset.
        batch_size (Optional[int], optional): size of a batch. Defaults to 32.
        data_path (Optional[str], optional): path to data folder to download dataset into. Defaults to "data".


    Raises:
        ValueError: dataset has to be 'mnist' or 'stl10',

    Returns:
        Tuple[DataLoader, DataLoader]: classic, rotated and flipped dataloaders.
    """

    if dataset not in {"mnist", "stl10", "cifar10"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be 'mnist', 'stl10' or 'cifar10'.")

    if dataset == "mnist":
        classic_dataset = MNIST(
            data_path,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)]),
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

    if dataset == "cifar10":
        classic_dataset = CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]),
        )
        rotated_dataset = CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=Compose([RandomRotation(degrees=180), ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]),
        )
        flipped_dataset = CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            ),
        )
    elif dataset == "stl10":
        classic_dataset = STL10(
            data_path,
            split="train",
            download=True,
            transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD)]),
        )
        rotated_dataset = STL10(
            data_path,
            split="train",
            download=True,
            transform=Compose(
                [
                    RandomRotation(degrees=180),
                    ToTensor(),
                    Normalize(STL10_MEAN, STL10_STD),
                ]
            ),
        )
        flipped_dataset = STL10(
            data_path,
            split="train",
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
