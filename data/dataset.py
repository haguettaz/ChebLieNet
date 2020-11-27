import os

import matplotlib.pyplot as plt
import torch
from torchvision.datasets.utils import download_and_extract_archive


def download_mnist(data_path):
    def check_exists(processed_path):
        return os.path.exists(os.path.join(processed_path, "training.pt")) and os.path.exists(os.path.join(processed_path, "test.pt"))

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
    test_set = (read_image_file(os.path.join(raw_path, "t10k-images-idx3-ubyte")), read_label_file(os.path.join(raw_path, "t10k-labels-idx1-ubyte")))
    with open(os.path.join(processed_path, "training.pt"), "wb") as f:
        torch.save(training_set, f)
    with open(os.path.join(processed_path, "test.pt"), "wb") as f:
        torch.save(test_set, f)

    print("Done!")

    return processed_path


def download_rotated_mnist(data_path):
    def check_exists(processed_path):
        return os.path.exists(os.path.join(processed_path, "train_all.npz")) and os.path.exists(os.path.join(processed_path, "test.npz"))

    url = "https://staff.fnwi.uva.nl/e.j.bekkers/MLData/ROT_MNIST.zip"

    raw_path = os.path.join(data_path, "RotatedMNIST", "raw")
    processed_path = os.path.join(data_path, "RotatedMNIST", "processed")

    if check_exists(processed_path):
        return processed_path

    download_and_extract_archive(url, raw_path, processed_path)
    print("Done!")

    return processed_path


def preprocess_mnist(images, targets):
    """
    Preprocess a tensor of images and a tensor of targets from MNIST dataset. The images
    are normalized.

    Args:
        images (torch.tensor): the images with shape (N, H, W)
        targets (torch.tensor): the targets with shape (N)

    Returns:
        images (torch.tensor): the images with shape (N, C, H, W)
        targets (torch.tensor): the targets with shape (N, D)
    """
    images = images.float()
    targets = targets.long()

    images = torch.divide(images - images.mean(), images.std())
    images = images.flip(1).unsqueeze(1)  # flip image on the y axis and add the channel dimension

    # Return preprocessed dataset
    return images, targets


def preprocess_rotated_mnist(images, targets):
    """
    Preprocess a tensor of images and a tensor of targets from rotated MNIST dataset. The images
    are normalized.

    Args:
        images (torch.tensor): the images with shape (N, C, H, W)
        targets (torch.tensor): the targets with shape (N)

    Returns:
        images (torch.tensor): the images with shape (N, C, H, W)
        targets (torch.tensor): the targets with shape (N, D)
    """
    images = torch.from_numpy(images.astype(np.float32))
    images = torch.divide(images - images.mean(), images.std())

    targets = torch.from_numpy(targets.astype(np.int64)).unsqueeze(1)

    # Return preprocessed dataset
    return images, targets


def visualize_samples(datalist, grid_size=(3, 3)):
    """
    [summary]

    Args:
        datalist ([type]): [description]
        grid_size (tuple, optional): [description]. Defaults to (3,3).
    """
    num_rows, num_cols = grid_size

    fig = plt.figure(figsize=(num_rows * 8.0, num_cols * 8.0))

    for r in range(num_rows):
        for c in range(num_cols):
            sample_idx = random_choice(torch.arange(len(datalist))).item()
            sample = datalist[sample_idx]
            ax = fig.add_subplot(
                num_rows,
                num_cols,
                r * num_cols + c + 1,
                projection="3d",
                xlim=(graph_data.x_axis.min(), graph_data.x_axis.max()),
                ylim=(graph_data.y_axis.min(), graph_data.y_axis.max()),
                zlim=(graph_data.z_axis.min(), graph_data.z_axis.max()),
            )

            im = ax.scatter(sample.pos[:, 0], sample.pos[:, 1], sample.pos[:, 2], c=sample.x, s=50, alpha=0.5)

            plt.colorbar(im, fraction=0.04, pad=0.1)

            ax.set_title(f"sample #{sample_idx} labeled {sample.y.item()}")

    plt.show()
