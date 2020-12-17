import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, STL10
from torchvision.transforms import Compose, ToTensor

from ..utils import shuffle_tensor


def get_train_val_data_loaders(dataset, batch_size=32, val_ratio=0.2, data_path="data"):

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


def get_test_data_loader(dataset, batch_size=32, data_path="data"):

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
