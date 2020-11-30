import torch

import os
import math
import shutil
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import download_and_extract_archive

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import ChebConv, GCNConv, global_mean_pool, global_max_pool, max_pool, voxel_grid
from torch_geometric.transforms import LaplacianLambdaMax

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy

import PIL
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# class GraphData(object):
#     def __init__(self, grid_size, self_loop=True, sigma=1, weight_threshold=1, lambdas=(1., 1., 1.)):
#         """
#             grid_size (tuple): the size of the grid in format (nx1, nx2, nx3)
#             self_loop (bool): the indicator if the graph contains self loop or not
#             connection (tuple): the scheme to construct edges, it contains the name of the scheme ("tg" or "nn") with float parameter
#             origin_center (bool): the indicator if the grid must by centered at origin
#             weight_coeff (float): the coefficient in the weights exponential
#             ker_orient (tuple): the orientation angles of the first eigen vector of the anisotropic tensor
#             ker_lambda (tuple): the eigenvalues of the anisotropic tensor, correspond to anisotropy intensity
#         """
#         self.nx1, self.nx2, self.nx3 = grid_size
#
#         self.num_nodes = self.nx1 * self.nx2 * self.nx3
#
#         self.self_loop = self_loop
#         self.weight_threshold = weight_threshold
#         self.sigma = sigma
#
#         self.init_ref()
#
#         self.init_nodes()
#
#         self.l1, self.l2, self.l3 = lambdas
#
#         self.init_edges()
#
#     def data(self):
#         return Data(edge_index=self.edge_index, edge_attr=self.edge_weight)
#
#     def add_signal(self, signal):
#         if len(signal) == self.num_nodes:
#             self.signal = signal.flatten()
#             return
#
#         if signal.shape[0] != self.nx1 or signal.shape[1] != self.nx2:
#             raise ValueError(f"the size of the signal does not coincide with the size of the graph")
#
#         self.signal = signal.expand(self.nx3, -1, -1).permute(2, 1, 0).flatten()
#
#     def init_ref(self):
#
#         z_min, z_max = 0, 2 * np.pi
#         z_res = 2 * np.pi / self.nx3
#
#         self.centroid_idx = self.nx1 // 2 * self.nx2 * self.nx3 + self.nx2 // 2 * self.nx3 + self.nx3 // 2
#
#         # we define the 3 axis
#         self.x1_axis = torch.arange(0., self.nx1, 1.)
#         self.x2_axis = torch.arange(0., self.nx2, 1.)
#         self.x3_axis = torch.arange(0., 2 * np.pi, z_res)
#
#     def init_nodes(self):
#         self.node_index = torch.arange(self.num_nodes)
#         # we define the grid points and reshape them to get 1-d arrays
#         xv, yv, zv = torch.meshgrid(self.x1_axis, self.x2_axis, self.x3_axis)
#         self.node_pos = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
#
#     def init_edges(self):
#
#         distances = self.compute_distances()
#         weights = self.compute_weights(distances)
#         edge_indices = torch.reshape(torch.stack(torch.meshgrid(self.node_index, self.node_index), -1), [-1, 2])
#         threshold_mask = (weights >= self.weight_threshold)
#
#         self.edge_index = torch.transpose(edge_indices[threshold_mask], 1, 0)
#         self.edge_metric = distances[threshold_mask]
#         self.edge_weight = weights[threshold_mask]
#
#     def metric_tensor(self, theta):
#         e1 = torch.tensor([np.cos(theta), np.sin(theta), 0], dtype=torch.float32)
#         e2 = torch.tensor([-np.sin(theta), np.cos(theta), 0], dtype=torch.float32)
#         e3 = torch.tensor([0, 0, 1], dtype=torch.float32)
#
#         D = e1[:, None] * e1[None, :] * self.l1
#         D += e2[:, None] * e2[None, :] * self.l2
#         D += e3[:, None] * e3[None, :] * self.l3
#
#         return D
#
#     def compute_distances(self):
#
#         distances = torch.zeros([len(self.node_pos), len(self.node_pos)], dtype=torch.float32)
#         difference_vectors = (self.node_pos[:, None, :] - self.node_pos[None, :, :])
#         for z in self.x3_axis:
#             z_selection = (self.node_pos[:, 2] == z)
#             dists = torch.matmul(difference_vectors[z_selection, :, None, :],
#                                  torch.matmul(self.metric_tensor(z), difference_vectors[z_selection, :, :, None]))
#             distances[z_selection, :] = dists[:, :, 0, 0]
#
#         return distances.flatten()
#
#     def compute_weights(self, distances):
#         return torch.exp(-distances ** 2 / (2 * self.sigma ** 2))
#
# def projection(self, images, targets):
#     # images dimension : (n_images, height, width, n_orientations, n_channels)
#
#     if len(images.shape) != 5:
#         raise ValueError("images must be in format (n_images, n_orientations, height, width, n_channels)")
#
#     n_images, height, width, _, n_channels = images.size()
#
#     if self.nx1 != width or self.nx2 != height:
#         raise ValueError(f"grid size and image size should coincide but are ({self.nx1, self.nx2}"
#                          f"and ({width, height})")
#     if n_channels > 1:
#         raise ValueError(f"images with channels > 1 are supported for the moment")
#
#     x = images.permute(0, 2, 1, 3, 4).expand(-1, -1, -1, self.nx3, -1).reshape(n_images, -1, n_channels)
#
#     if n_images == 1:
#         return Data(x=x[0], y=targets, pos=self.node_pos, edge_index=self.edge_index, edge_attr=self.edge_weight)
#
#     return [Data(x=x[idx],
#                  pos=self.node_pos,
#                  y=targets[idx],
#                  edge_index=self.edge_index,
#                  edge_attr=self.edge_weight) for idx in range(n_images)]


class GraphData(object):
    def __init__(self, grid_size, self_loop=True, sigma=1, weight_threshold=1, lambdas=(1.0, 1.0, 1.0)):
        """
            grid_size (tuple): the size of the grid in format (nx1, nx2, nx3)
            self_loop (bool): the indicator if the graph contains self loop or not
            connection (tuple): the scheme to construct edges, it contains the name of the scheme ("tg" or "nn") with float parameter
            origin_center (bool): the indicator if the grid must by centered at origin
            weight_coeff (float): the coefficient in the weights exponential
            ker_orient (tuple): the orientation angles of the first eigen vector of the anisotropic tensor
            ker_lambda (tuple): the eigenvalues of the anisotropic tensor, correspond to anisotropy intensity
        """
        self.nx1, self.nx2, self.nx3 = grid_size
        self.num_nodes = self.nx1 * self.nx2 * self.nx3
        self.self_loop = self_loop
        self.weight_threshold = weight_threshold
        self.sigma = sigma
        self.init_ref()
        self.init_nodes()
        self.l1, self.l2, self.l3 = lambdas
        self.init_edges()

    @property
    def data(self):
        return Data(edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes)

    def add_signal(self, signal):
        if len(signal) == self.num_nodes:
            self.signal = signal.flatten()
            return
        if signal.shape[0] != self.nx1 or signal.shape[1] != self.nx2:
            raise ValueError(f"the size of the signal does not coincide with the size of the graph")
        self.signal = signal.expand(self.nx3, -1, -1).permute(2, 1, 0).flatten()

    def init_ref(self):
        # we define the 3 axis
        self.x1_axis = torch.arange(0.0, self.nx1, 1.0)
        self.x2_axis = torch.arange(0.0, self.nx2, 1.0)
        self.x3_axis = torch.arange(0.0, math.pi, math.pi / self.nx3)

    def init_nodes(self):
        self.node_index = torch.arange(self.num_nodes)
        # we define the grid points and reshape them to get 1-d arrays
        xv, yv, zv = torch.meshgrid(self.x1_axis, self.x2_axis, self.x3_axis)
        self.node_pos = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)

    def init_edges(self):
        distances = self.compute_distances()
        weights = self.compute_weights(distances)
        edge_indices = torch.reshape(torch.stack(torch.meshgrid(self.node_index, self.node_index), -1), [-1, 2])
        threshold_mask = weights >= self.weight_threshold
        self.edge_index = torch.transpose(edge_indices[threshold_mask], 1, 0)
        self.edge_metric = distances[threshold_mask]
        self.edge_weight = weights[threshold_mask]

    def metric_tensor(self, theta):
        e1 = torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.float32)
        e2 = torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.float32)
        e3 = torch.tensor([0, 0, 1], dtype=torch.float32)
        D = e1.unsqueeze(1) * e1.unsqueeze(0) * self.l1
        D += e2.unsqueeze(1) * e2.unsqueeze(0) * self.l2
        D += e3.unsqueeze(1) * e3.unsqueeze(0) * self.l3
        return D

    def compute_distances(self):
        distances = torch.zeros([len(self.node_pos), len(self.node_pos)], dtype=torch.float32)
        difference_vectors = torch.cat(
            (
                self.node_pos[:, :2].unsqueeze(1) - self.node_pos[:, :2].unsqueeze(0),
                (((self.node_pos[:, 2].unsqueeze(1) - self.node_pos[:, 2].unsqueeze(0) - math.pi / 2) % math.pi) - math.pi / 2).unsqueeze(2),
            ),
            dim=2,
        )
        for z in self.x3_axis:
            z_selection = self.node_pos[:, 2] == z
            dists = torch.matmul(
                difference_vectors[z_selection].unsqueeze(2), torch.matmul(self.metric_tensor(z), difference_vectors[z_selection].unsqueeze(3))
            )
            distances[z_selection, :] = dists[:, :, 0, 0]
        return distances.flatten()

    def compute_weights(self, distances):
        return torch.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def embed_on_graph(self, images, targets):
        # images dimension : (n_images, height, width, n_orientations, n_channels)
        if len(images.shape) != 5:
            raise ValueError("images must be in format (n_images, n_orientations, height, width, n_channels)")
        n_images, height, width, _, n_channels = images.size()
        if self.nx1 != width or self.nx2 != height:
            raise ValueError(f"grid size and image size should coincide but are ({self.nx1, self.nx2}" f"and ({width, height})")
        if n_channels > 1:
            raise ValueError(f"images with channels > 1 are supported for the moment")
        x = images.permute(0, 2, 1, 3, 4).expand(-1, -1, -1, self.nx3, -1).reshape(n_images, -1, n_channels)
        if n_images == 1:
            return Data(x=x[0], y=targets, pos=self.node_pos, edge_index=self.edge_index, edge_attr=self.edge_weight)
        return [Data(x=x[idx], pos=self.node_pos, y=targets[idx], edge_index=self.edge_index, edge_attr=self.edge_weight) for idx in range(n_images)]

    def random_mask(self, num_true, num_false, true_force=None):
        mask = torch.zeros(num_true + num_false)
        mask[:num_true] = 1.0
        if true_force:
            mask[true_force] = 1.0
        idx = torch.randperm(mask.nelement())
        mask = mask.view(-1)[idx].view(mask.size())
        return mask.bool()

    def random_purge(self, frac):
        # remove random nodes (node at origin cannot be removed)
        node_mask = self.random_mask(self.num_nodes - int(frac * self.num_nodes), int(frac * self.num_nodes), self.o_idx)
        node_index = self.node_index[node_mask]
        # remove edges between these nodes
        source_mask = torch.stack([(self.edge_index[0] == n) for n in node_index]).sum(0).bool()
        target_mask = torch.stack([(self.edge_index[1] == n) for n in node_index]).sum(0).bool()
        edge_mask = source_mask & target_mask
        self.edge_index = self.edge_index[:, edge_mask]
        self.edge_metric = self.edge_metric[edge_mask]
        self.edge_weight = self.edge_weight[edge_mask]

    def random_purge_edges(self, frac):
        # remove random nodes (node at origin cannot be removed)
        n_edges = self.edge_index.shape[1]
        n_keep = int(n_edges * (1 - frac))
        keep_indices = np.random.permutation(n_edges)[:n_keep]
        self.edge_index = self.edge_index[:, keep_indices]
        self.edge_weight = self.edge_weight[keep_indices]

    def projection(self, images, targets):
        # images dimension : (n_images, height, width, n_orientations, n_channels)

        if len(images.shape) != 5:
            raise ValueError("images must be in format (n_images, n_orientations, height, width, n_channels)")

        n_images, height, width, _, n_channels = images.size()

        if self.nx1 != width or self.nx2 != height:
            raise ValueError(f"grid size and image size should coincide but are ({self.nx1, self.nx2}" f"and ({width, height})")
        if n_channels > 1:
            raise ValueError(f"images with channels > 1 are supported for the moment")

        x = images.permute(0, 2, 1, 3, 4).expand(-1, -1, -1, self.nx3, -1).reshape(n_images, -1, n_channels)

        if n_images == 1:
            return Data(x=x[0], y=targets, pos=self.node_pos, edge_index=self.edge_index, edge_attr=self.edge_weight)

        return [Data(x=x[idx], pos=self.node_pos, y=targets[idx], edge_index=self.edge_index, edge_attr=self.edge_weight) for idx in range(n_images)]


def plot_weight_field(graph_data, ax, node_idx, only_neighbors=False):
    mask = graph_data.edge_index[0] == node_idx

    targets = graph_data.edge_index[:, mask][1]
    weight = graph_data.edge_weight[mask]

    if only_neighbors:
        mask = weight > 0

        im = ax.scatter(
            graph_data.node_pos[targets[mask], 0],
            graph_data.node_pos[targets[mask], 1],
            graph_data.node_pos[targets[mask], 2],
            c=weight,
            s=100,
            alpha=0.5,
        )

    else:
        im = ax.scatter(graph_data.node_pos[targets, 0], graph_data.node_pos[targets, 1], graph_data.node_pos[targets, 2], c=weight, s=100, alpha=0.5)

    plt.colorbar(im, fraction=0.04, pad=0.1)
    im = ax.scatter(
        graph_data.node_pos[node_idx, 0],
        graph_data.node_pos[node_idx, 1],
        graph_data.node_pos[node_idx, 2],
        s=100,
        c="white",
        edgecolors="black",
        linewidth=3,
        alpha=1.0,
    )

    ax.set_title(f"weight field from node {node_idx}")


eps = 0.25
xi = 0.01
nx1, nx2, nx3 = (28, 28, 12)
graph_data = GraphData(grid_size=(nx1, nx2, nx3), self_loop=True, weight_threshold=0.25, sigma=0.25, lambdas=((xi / eps), xi, 1))
graph_data.random_purge_edges(0.75)


fig = plt.figure(figsize=(10, 9))

ax = fig.add_subplot(
    1, 1, 1, projection="3d", xlabel="x", ylabel="y", zlabel="z", xlim=(0.0, graph_data.nx1 - 1), ylim=(0.0, graph_data.nx2 - 1), zlim=(0.0, np.pi)
)
plot_weight_field(graph_data, ax, 5002)

plt.show()


"""# Image on graph"""


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


def preprocess_mnist(images, targets):
    images = images.float()
    targets = targets.long()

    images = torch.divide(images - images.mean(), images.std())
    images = images.flip(1)  # flip image on the y axis
    images = images.unsqueeze(3).unsqueeze(4)

    # Return preprocessed dataset
    return images, targets


def plot_graph_signal(graph_data, signal, ax, mask_zero=False, color_bar=True):
    if mask_zero:
        mask = signal > 1e-3
        im = ax.scatter(graph_data.node_pos[mask, 0], graph_data.node_pos[mask, 1], graph_data.node_pos[mask, 2], c=signal[mask], alpha=0.5)

    else:
        im = ax.scatter(graph_data.node_pos[:, 0], graph_data.node_pos[:, 1], graph_data.node_pos[:, 2], c=signal, alpha=0.5)

    if color_bar:
        plt.colorbar(im, fraction=0.04, pad=0.1)


def plot_random_sample(datalist, ax):
    random_idx = torch.randint(len(datalist), (1, 1))
    plot_graph_signal(graph_data, datalist[random_idx].x, ax)
    ax.set_title(fr"sample #{random_idx.item()} labeled {datalist[random_idx].y.item()}")


def get_datalist(graph_data, processed_path, train=True):
    if train:
        images, targets = torch.load(os.path.join(processed_path, "training.pt"))
    else:
        images, targets = torch.load(os.path.join(processed_path, "test.pt"))

    images, targets = preprocess_mnist(images, targets)

    return graph_data.projection(images, targets)


data_path = "data"
processed_path = download_mnist(data_path)


"""# Neural networks : Chebyschev"""


def create_supervised_trainer(model, optimizer, device=None):
    device = device or torch.device("cpu")

    model = model.to(device)

    def prepare_batch(batch, device):
        batch = batch.to(device)
        return batch

    def output_transform(batch, loss, device):
        return loss.item()

    def update(engine, batch):  # pylint: disable=W0613
        model.train()
        optimizer.zero_grad()

        batch = prepare_batch(batch, device)
        y_hat = model(batch)
        loss = F.nll_loss(y_hat, batch.y)
        loss.backward()
        optimizer.step()

        return output_transform(batch, loss, device)

    return Engine(update)


def create_supervised_evaluator(model, metrics, device=None):
    device = device or torch.device("cpu")

    model = model.to(device)

    def prepare_batch(batch, device):
        batch = batch.to(device)
        return batch

    def output_transform(batch, y_hat, device):
        return y_hat, batch.y

    def inference(engine, batch):  # pylint: disable=W0613
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device)
            y_hat = model(batch)

            return output_transform(batch, y_hat, device=device)

    engine = Engine(inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def log_training_loss(trainer, loss_log):
    writer.add_scalar("training/loss", loss_log.mean(), trainer.state.epoch)


def append_training_loss(trainer, loss_log, num_train_batch):
    loss_log[(trainer.state.iteration - 1) % num_train_batch] = trainer.state.output


def log_test_results(trainer):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    for k in metrics:
        print(f"testing {k} @{trainer.state.epoch}: {metrics[k]}")
        writer.add_scalar(f"testing/{k}", metrics[k], trainer.state.epoch)


def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    for k in metrics:
        writer.add_scalar(f"training/{k}", metrics[k], trainer.state.epoch)


def get_dataloaders(processed_path, graph_data, train_batch_size=16, test_batch_size=16, shuffle=True):
    train_list = get_datalist(graph_data, processed_path, train=True)
    test_list = get_datalist(graph_data, processed_path, train=False)

    train_loader = DataLoader(train_list, batch_size=train_batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_list, batch_size=test_batch_size, shuffle=shuffle)

    return train_loader, test_loader


def rescale_pos(pos):
    # scale pos between 0 and 1
    min_, _ = pos.min(dim=0)
    max_, _ = pos.max(dim=0)
    for i in range(3):
        if max_[i] - min_[i] == 0:
            pos[:, i] = pos[:, i] - min_[i]
        else:
            pos[:, i] = torch.divide(pos[:, i] - min_[i], max_[i] - min_[i])
    # scale pos on the natural integer axis
    dX, _ = torch.max(pos[1:] - pos[:-1], dim=0)
    for i in range(3):
        if dX[i] != 0:
            pos[:, i] = torch.divide(pos[:, i], dX[i])
    return pos.ceil()


def spatial_subsampling(pos, batch, divider=2.0):
    pos = rescale_pos(pos)

    # number of nodes along each dimension
    nx1, _ = pos.max(dim=0)
    nx1 = nx1 + 1

    divider = torch.tensor([divider, divider, 1.0], device=pos.device)

    cluster = torch.floor_divide(pos, divider)
    cluster = batch * torch.prod(nx1) + cluster[:, 2] * torch.prod(nx1[:2]) + cluster[:, 1] * nx1[0] + cluster[:, 0]

    return cluster.double()


def orientation_subsampling(pos, batch):
    # rescale pos on natural integers
    pos = rescale_pos(pos)

    # number of nodes along each dimension
    nx1, _ = pos.max(dim=0)
    nx1 = nx1 + 1

    cluster = batch * torch.prod(nx1) + pos[:, 1] * nx1[0] + pos[:, 0]

    return cluster.double()


class ChebNet(torch.nn.Module):
    def __init__(self, K):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(1, 8, K)  # 1*16*K weights + 16 bias
        self.conv2 = ChebConv(8, 16, K)  # 16*32*K weights + 32 bias

        self.conv3 = ChebConv(16, 16, K)  # 32*32*K weights + 32 bias
        self.conv4 = ChebConv(16, 16, K)  # 32*10*K weights + 10 bias

        self.conv5 = ChebConv(16, 16, K)  # 32*32*K weights + 32 bias
        self.conv6 = ChebConv(16, 10, K)  # 32*10*K weights + 10 bias

        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.bn5 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        # First layer: 2 Chebyschev convolution - Spatial max pooling /2
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn1(data.x)
        data.x = data.x.relu()

        data.x = self.conv2(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn2(data.x)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        # Second layer: 2 Chebyschev convolution - Spatial max pooling /2
        data.x = self.conv3(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn3(data.x)
        data.x = data.x.relu()

        data.x = self.conv4(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn4(data.x)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        # Second layer: 2 Chebyschev convolution - Spatial max pooling /2
        data.x = self.conv5(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn5(data.x)
        data.x = data.x.relu()

        data.x = self.conv6(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        # Third layer: Orientation max pooling
        cluster = orientation_subsampling(data.pos, data.batch)
        data = max_pool(cluster, data)

        # Final layer: Global mean pooling
        data.x = global_max_pool(data.x, data.batch)

        return F.log_softmax(data.x, dim=1)

    @property
    def capacity(self):
        return sum(p.numel() for p in self.parameters())


"""## Only spatial"""

eps = 1
xi = 0.01
nx1, nx2, nx3 = (28, 28, 1)
graph_data = GraphData(grid_size=(nx1, nx2, nx3), self_loop=True, weight_threshold=0.25, sigma=0.25, lambdas=((xi / eps), xi, 1))
graph_data.random_purge_edges(0.5)


train_loader, test_loader = get_dataloaders(processed_path, graph_data, train_batch_size=64, test_batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChebNet(10)
optimizer = torch.optim.Adam(model.parameters())

print("model capacity 2D", model.capacity)

metrics = {
    "accuracy": Accuracy(),
}

num_train_batch = len(train_loader)
loss_log = torch.zeros(num_train_batch)  # used to take the mean loss on the current epoch

trainer = create_supervised_trainer(model, optimizer, device=device)
evaluator = create_supervised_evaluator(model, metrics, device=device)

ProgressBar(persist=False, desc="Training").attach(trainer)
ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

tensorboard_log = os.path.join("tensorboard", "mnist", "2d_grid_chebnet_edge_purge_0.5")

if os.path.exists(tensorboard_log):
    shutil.rmtree(tensorboard_log)

writer = SummaryWriter(log_dir=tensorboard_log)

_ = trainer.add_event_handler(Events.ITERATION_COMPLETED, append_training_loss, loss_log, num_train_batch)
_ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_loss, loss_log)
_ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_test_results)
# _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)

max_epochs = 20
trainer.run(train_loader, max_epochs=max_epochs)
writer.close()

model_path = os.path.join("drive", "MyDrive", "models", "2d_chebnet.pt")
torch.save(model.state_dict(), model_path)


"""## Spatial and orientation"""

eps = 0.25
xi = 0.01
nx1, nx2, nx3 = (28, 28, 12)
graph_data = GraphData(grid_size=(nx1, nx2, nx3), self_loop=True, weight_threshold=0.25, sigma=0.25, lambdas=((xi / eps), xi, 1))
graph_data.random_purge_edges(0.5)

train_loader, test_loader = get_dataloaders(processed_path, graph_data, train_batch_size=16, test_batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChebNet(10)
optimizer = torch.optim.Adam(model.parameters())

print("model capacity", model.capacity)

metrics = {
    "accuracy": Accuracy(),
}

num_train_batch = len(train_loader)
loss_log = torch.zeros(num_train_batch)  # used to take the mean loss on the current epoch

trainer = create_supervised_trainer(model, optimizer, device=device)
evaluator = create_supervised_evaluator(model, metrics, device=device)

ProgressBar(persist=False, desc="Training").attach(trainer)
ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

tensorboard_log = os.path.join("tensorboard", "mnist", "se2_grid_chebnet_edge_purge_0.5")

if os.path.exists(tensorboard_log):
    shutil.rmtree(tensorboard_log)

writer = SummaryWriter(log_dir=tensorboard_log)

_ = trainer.add_event_handler(Events.ITERATION_COMPLETED, append_training_loss, loss_log, num_train_batch)
_ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_loss, loss_log)
_ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_test_results)
# _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)

max_epochs = 20
trainer.run(train_loader, max_epochs=max_epochs)
writer.close()
