import os

import numpy as np
import torch
from torch_geometric.data import DataLoader

from .dataloader import preprocess_mnist, preprocess_rotated_mnist


def get_datalist_mnist(graph_data, processed_path, train=True):
    """
    [summary]

    Args:
        graph_data ([type]): [description]
        processed_path ([type]): [description]
        train (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if train:
        images, targets = torch.load(os.path.join(processed_path, "training.pt"))
    else:
        images, targets = torch.load(os.path.join(processed_path, "test.pt"))

    images, targets = preprocess_mnist(images, targets)

    return graph_data.embed_on_graph(images, targets)


def get_datalist_rotated_mnist(graph_data, processed_path, train=True):
    """
    [summary]

    Args:
        graph_data ([type]): [description]
        processed_path ([type]): [description]
        train (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if train:
        dataset = np.load(os.path.join(processed_path, "train_all.npz"))
    else:
        dataset = np.load(os.path.join(processed_path, "test.npz"))

    images, targets = preprocess_rotated_mnist(dataset["data"], dataset["labels"])

    return graph_data.projection(images, targets)


def get_dataloaders(data_list, batch_size=16, shuffle=True):
    """
    [summary]

    Args:
        data_list ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 16.
        shuffle (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
