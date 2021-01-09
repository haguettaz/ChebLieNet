import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import download_and_extract_archive


def get_data_list_rotated_mnist(graph_data, processed_path, train=True):
    """
    Get the list of Data object with rotated MNIST images and targets embedded on the given graph.

    Args:
        graph_data (GraphData): the graph data object.
        processed_path ([type]): the path to the folder containing the processed dataset.
        train (bool, optional): the indicator wether to use training dataset or not. Defaults to True.

    Returns:
        (list): the list of Data object.
    """
    if train:
        dataset = np.load(os.path.join(processed_path, "train_all.npz"))
    else:
        dataset = np.load(os.path.join(processed_path, "test.npz"))

    images, targets = preprocess_rotated_mnist(dataset["data"], dataset["labels"])

    return graph_data.embed_on_graph(images, targets)


def download_mnist(data_path):
    """
    Download MNIST dataset and store it at the given path.

    Args:
        data_path (str): the path to the data folder to download data on.
    """

    def check_exists(processed_path):
        """
        Check if processed data already exists at the given path. If yes, does not download data again.

        Args:
            processed_path (str): the path to the folder to put processed data on.

        Returns:
            (bool): the indicator if the processed data already exists at the given path.
        """
        return os.path.exists(os.path.join(processed_path, "training.pt")) and os.path.exists(
            os.path.join(processed_path, "test.pt")
        )

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    raw_path = os.path.join(data_path, "MNIST", "raw")
    processed_path = os.path.join(data_path, "MNIST", "processed")

    if check_exists(processed_path):
        return processed_path

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    # download files
    for url, md5 in resources:
        filename = url.rpartition("/")[2]
        download_and_extract_archive(url, download_root=raw_path, filename=filename, md5=md5)

    # process and save as torch files
    print("Processing...")

    training_set = (
        read_image_file(os.path.join(raw_path, "train-images-idx3-ubyte")),
        read_label_file(os.path.join(raw_path, "train-labels-idx1-ubyte")),
    )
    test_set = (
        read_image_file(os.path.join(raw_path, "t10k-images-idx3-ubyte")),
        read_label_file(os.path.join(raw_path, "t10k-labels-idx1-ubyte")),
    )
    with open(os.path.join(processed_path, "training.pt"), "wb") as f:
        torch.save(training_set, f)
    with open(os.path.join(processed_path, "test.pt"), "wb") as f:
        torch.save(test_set, f)

    print("Done!")

    return processed_path


def download_rotated_mnist(data_path):
    """
    Download rotated MNIST dataset and store it at the given path.

    Args:
        data_path (str): the path to the data folder to download data on.
    """

    def check_exists(processed_path):
        """
        Check if processed data already exists at the given path. If yes, does not download data again.

        Args:
            processed_path (str): the path to the folder to put processed data on.

        Returns:
            (bool): the indicator if the processed data already exists at the given path.
        """
        return os.path.exists(os.path.join(processed_path, "train_all.npz")) and os.path.exists(
            os.path.join(processed_path, "test.npz")
        )

    url = "https://staff.fnwi.uva.nl/e.j.bekkers/MLData/ROT_MNIST.zip"

    raw_path = os.path.join(data_path, "RotatedMNIST", "raw")
    processed_path = os.path.join(data_path, "RotatedMNIST", "processed")

    if check_exists(processed_path):
        return processed_path

    download_and_extract_archive(url, raw_path, processed_path)
    print("Done!")

    return processed_path
