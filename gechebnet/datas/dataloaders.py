# coding=utf-8

import math
import os
from typing import Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10, MNIST, STL10
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

from ..utils.utils import shuffle_tensor
from .datasets import ARTCDataset
from .transforms import BoolToInt, Compose, Normalize, Random90Rotation, ToGraphSignal, ToTensor

# computed mean and std on the training sets.
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
STL10_MEAN, STL10_STD = (0.4472, 0.4396, 0.4050), (0.2606, 0.2567, 0.2700)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
ARTC_MEAN = (
    2.61613026e01,
    9.84836817e-01,
    1.16275556e-01,
    -4.59006906e-01,
    1.92794472e-01,
    1.07488334e-02,
    9.83474609e04,
    1.01012266e05,
    2.16136414e02,
    2.58974274e02,
    3.76523026e-08,
    2.88836273e02,
    2.88041748e02,
    3.42482941e02,
    1.20318799e04,
    6.34397430e01,
)
ARTC_STD = (
    1.7042067e01,
    8.1648855e00,
    5.6875458e00,
    6.4972377e00,
    5.4460597e00,
    6.3829664e-03,
    7.7596729e03,
    3.8046602e03,
    9.7163849e00,
    1.4277466e01,
    1.8769481e-07,
    1.9797100e01,
    1.9022367e01,
    6.2425934e02,
    6.7624426e02,
    4.2128639e00,
)


def get_train_val_loaders(
    dataset: str,
    batch_size: Optional[int] = 32,
    val_ratio: Optional[float] = 0.2,
    num_layers: Optional[int] = 6,
    path_to_data: Optional[str] = "data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation dataloaders.

    Args:
        dataset (str): dataset's name.
        batch_size (int, optional): batch's size. Defaults to 32.
        val_ratio (float, optional): validation samples' ratio. Defaults to 0.2.
        num_layers (int, optional): number of symmetric's layers. Defaults to 6.
        path_to_data (str, optional): path to data folder. Defaults to "data".

    Raises:
        ValueError: dataset has to be 'mnist', 'cifar10', 'stl10' or 'artc',

    Returns:
        (DataLoader): training dataloader.
        (DataLoader): validation dataloader.
    """

    if dataset not in {"mnist", "stl10", "cifar10", "artc"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be 'mnist', 'stl10', 'cifar10' or 'artc'.")

    if dataset == "mnist":
        dataset = MNIST(
            path_to_data,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "cifar10":
        dataset = CIFAR10(
            path_to_data,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "stl10":
        dataset = STL10(
            path_to_data,
            split="test",  # we use test set as training set since it has more samples
            download=True,
            transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "artc":
        dataset = ARTCDataset(
            os.path.join(path_to_data, "data_train"),
            transform_image=Compose([ToTensor(), Normalize(ARTC_MEAN, ARTC_STD), ToGraphSignal(num_layers)]),
            transform_target=Compose([ToTensor(), BoolToInt()]),
            download=True,
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


def get_equiv_test_loaders(
    dataset: str, batch_size: Optional[int] = 32, num_layers: Optional[int] = 6, path_to_data: Optional[str] = "data"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gets test dataloaders to test equivariance under rotation and flip property of the network.

    Args:
        dataset (str): name of the dataset.
        batch_size (Optional[int], optional): size of a batch. Defaults to 32.
        path_to_data (Optional[str], optional): path to data folder to download dataset into. Defaults to "data".

    Raises:
        ValueError: dataset has to be 'mnist', 'cifar10' or 'stl10',

    Returns:
        (DataLoader): test dataloader.
        (DataLoader): test dataloader with random rotations.
        (DataLoader): test dataloader with random flips.
    """
    if dataset not in {"mnist", "stl10", "cifar10"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be 'mnist', 'stl10' or 'cifar10'.")

    if dataset == "mnist":
        classic_dataset = MNIST(
            path_to_data,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD), ToGraphSignal(num_layers)]),
        )
        rotated_dataset = MNIST(
            path_to_data,
            train=False,
            download=True,
            transform=Compose(
                [RandomRotation(180), ToTensor(), Normalize(MNIST_MEAN, MNIST_STD), ToGraphSignal(num_layers)]
            ),
        )
        flipped_dataset = MNIST(
            path_to_data,
            train=False,
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    ToTensor(),
                    RandomVerticalFlip(p=0.5),
                    Normalize(MNIST_MEAN, MNIST_STD),
                    ToGraphSignal(num_layers),
                ]
            ),
        )

    elif dataset == "cifar10":
        classic_dataset = CIFAR10(
            path_to_data,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD), ToGraphSignal(num_layers)]),
        )
        rotated_dataset = CIFAR10(
            path_to_data,
            train=False,
            download=True,
            transform=Compose(
                [Random90Rotation(), ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD), ToGraphSignal(num_layers)]
            ),
        )
        flipped_dataset = CIFAR10(
            path_to_data,
            train=False,
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    Normalize(CIFAR10_MEAN, CIFAR10_STD),
                    ToGraphSignal(num_layers),
                ]
            ),
        )
    elif dataset == "stl10":
        classic_dataset = STL10(
            path_to_data,
            split="train",
            download=True,
            transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD), ToGraphSignal(num_layers)]),
        )
        rotated_dataset = STL10(
            path_to_data,
            split="train",
            download=True,
            transform=Compose(
                [Random90Rotation(), ToTensor(), Normalize(STL10_MEAN, STL10_STD), ToGraphSignal(num_layers)]
            ),
        )
        flipped_dataset = STL10(
            path_to_data,
            split="train",
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    Normalize(STL10_MEAN, STL10_STD),
                    ToGraphSignal(num_layers),
                ]
            ),
        )

    classic_loader = DataLoader(classic_dataset, batch_size=batch_size)
    rotated_loader = DataLoader(rotated_dataset, batch_size=batch_size)
    flipped_loader = DataLoader(flipped_dataset, batch_size=batch_size)

    return classic_loader, rotated_loader, flipped_loader


def get_test_loader(
    dataset: str, batch_size: Optional[int] = 32, num_layers: Optional[int] = 6, path_to_data: Optional[str] = "data"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gets test dataloaders to test equivariance under rotation and flip property of the network.

    Args:
        dataset (str): name of the dataset.
        batch_size (Optional[int], optional): size of a batch. Defaults to 32.
        path_to_data (Optional[str], optional): path to data folder to download dataset into. Defaults to "data".

    Raises:
        ValueError: dataset has to be 'mnist', 'cifar10', 'stl10' or 'artc'

    Returns:
        (DataLoader): test dataloader.
    """

    if dataset not in {"mnist", "stl10", "cifar10", "artc"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be 'mnist', 'stl10', 'cifar10' or 'artc'.")

    if dataset == "mnist":
        dataset = MNIST(
            path_to_data,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "cifar10":
        dataset = CIFAR10(
            path_to_data,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "stl10":
        dataset = STL10(
            path_to_data,
            split="train",
            download=True,
            transform=Compose([ToTensor(), Normalize(STL10_MEAN, STL10_STD), ToGraphSignal(num_layers)]),
        )

    elif dataset == "artc":
        dataset = ARTCDataset(
            os.path.join(path_to_data, "data_test"),
            transform_image=Compose([ToTensor(), Normalize(ARTC_MEAN, ARTC_STD), ToGraphSignal(num_layers)]),
            transform_target=Compose([ToTensor(), BoolToInt()]),
            download=True,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader
