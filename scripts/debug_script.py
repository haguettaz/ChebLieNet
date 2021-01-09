import math
import random

import numpy as np
import torch
from gechebnet.graph.graph import HyperCubeGraph
from gechebnet.model.chebnet import ChebNet

NX1 = 28
NX2 = 28

DATASET_NAME = "MNIST"  # STL10
VAL_RATIO = 0.2
NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

EPOCHS = 20
OPTIMIZER = "adam"

DEVICE = torch.device("cuda")

NUM_ITER = 100


def get_model(nx3, knn, eps, xi, weight_sigma, weight_kernel, K, pooling):

    # Different graphs are for successive pooling layers
    graph_1 = HyperCubeGraph(
        grid_size=(NX1, NX2),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=(weight_kernel, weight_sigma),
    )
    if graph_1.num_nodes > graph_1.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    graph_2 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 2),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=(weight_kernel, weight_sigma),
    )
    if graph_2.num_nodes > graph_2.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    graph_3 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=(weight_kernel, weight_sigma),
    )
    if graph_3.num_nodes > graph_3.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    model = ChebNet(
        (graph_1, graph_2, graph_3),
        K,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        laplacian_device=DEVICE,
        pooling=pooling,
    )

    return model.to(DEVICE)


if __name__ == "__main__":

    for _ in range(NUM_ITER):
        nx3 = random.choice([3, 4, 6, 9, 12])
        knn = random.choice([2, 4, 8, 16])
        eps = math.exp(random.uniform(math.log(1e-1), math.log(1.0)))
        xi = math.exp(random.uniform(math.log(1e-2), math.log(1.0)))
        weight_sigma = math.exp(random.uniform(math.log(0.1), math.log(10.0)))
        weight_kernel = random.choice(["gaussian", "cauchy", "laplacian"])
        K = random.choice([2, 4, 8, 16, 32, 64])
        pooling = random.choice(["max", "avg"])

        get_model(nx3, knn, eps, xi, weight_sigma, weight_kernel, K, pooling)
