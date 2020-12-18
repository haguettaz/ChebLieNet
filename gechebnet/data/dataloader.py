import math
from typing import Optional, Tuple, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, STL10
from torchvision.transforms import Compose, ToTensor

from ..utils import shuffle_tensor

T_co = TypeVar("T_co", covariant=True)


def get_train_val_data_loaders(
    dataset: Dataset[T_co], batch_size: Optional[int] = 32, val_ratio: Optional[float] = 0.2, data_path: Optional[str] = "data"
) -> Tuple[DataLoader, DataLoader]:
    """
    [summary]

    Args:
        dataset (Dataset[T_co]): [description]
        batch_size (Optional[int], optional): [description]. Defaults to 32.
        val_ratio (Optional[float], optional): [description]. Defaults to 0.2.
        data_path (Optional[str], optional): [description]. Defaults to "data".

    Raises:
        ValueError: [description]

    Returns:
        Tuple[DataLoader, DataLoader]: [description]
    """

    if dataset not in {"MNIST", "ROT-MNIST", "STL10"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be in 'MNIST', 'ROT-MNIST', 'STL10'")

    if dataset == "MNIST":
        dataset = MNIST(data_path, train=True, download=True, transform=Compose([ToTensor()]))

    # elif dataset == "ROT-MNIST":
    #     dataset = ROTMNIST(data_path, train=True, download=True, transform=Compose([ToTensor()]))

    elif dataset == "STL10":
        dataset = STL10(data_path, split="train", download=True, transform=Compose([ToTensor()]))

    N = len(dataset)
    split = math.floor(val_ratio * N)
    indices = shuffle_tensor(torch.arange(N))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_data_loader(dataset: Dataset[T_co], batch_size: Optional[int] = 32, data_path: Optional[str] = "data") -> DataLoader:
    """
    [summary]

    Args:
        dataset (Dataset[T_co]): [description]
        batch_size (Optional[int], optional): [description]. Defaults to 32.
        data_path (Optional[str], optional): [description]. Defaults to "data".

    Raises:
        ValueError: [description]

    Returns:
        DataLoader: [description]
    """

    if dataset not in {"MNIST", "ROT-MNIST", "STL10"}:
        raise ValueError(f"{dataset} is not a valid value for dataset: must be in 'MNIST', 'ROT-MNIST', 'STL10'")

    if dataset == "MNIST":
        dataset = MNIST(data_path, train=False, download=True, transform=Compose([ToTensor()]))

    # elif dataset == "ROT-MNIST":
    #     dataset = ROTMNIST(data_path, train=True, download=True, transform=Compose([ToTensor()]))

    elif dataset == "STL10":
        dataset = STL10(data_path, split="test", download=True, transform=Compose([ToTensor()]))

    test_loader = DataLoader(dataset, batch_size=batch_size)

    return test_loader
