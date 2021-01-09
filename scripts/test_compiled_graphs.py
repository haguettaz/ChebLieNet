import math
import random

import numpy as np
import torch
from gechebnet.graph.graph import HyperCubeGraph

NX1 = 28
NX2 = 28

POOLING_SIZE = 2

DEVICE = torch.device("cuda")

import time

NUM_ITER = 1


def build_graphs(knn):
    nx3 = random.randint(3, 12)
    eps = math.exp(random.uniform(math.log(0.1), math.log(1.0)))
    xi = math.exp(random.uniform(math.log(1e-2), math.log(1.0)))

    print((xi, xi * eps, 1.0))

    times = []
    print(f"KNN = {int(knn * POOLING_SIZE ** 4)} and V = {NX1*NX2*nx3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_1 = HyperCubeGraph(
            grid_size=(NX1, NX2),
            nx3=nx3,
            knn=int(knn * POOLING_SIZE ** 4),
            weight_comp_device=DEVICE,
            sigmas=(xi, xi * eps, 1.0),
        )
        end = time.time()
        if graph_1.num_nodes > graph_1.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")

    times = []
    print(f"KNN = {int(knn * POOLING_SIZE ** 2)} and V = {(NX1//POOLING_SIZE)*(NX2//POOLING_SIZE)*nx3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_2 = HyperCubeGraph(
            grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
            nx3=nx3,
            knn=int(knn * POOLING_SIZE ** 2),
            weight_comp_device=DEVICE,
            sigmas=(xi / eps, xi, 1.0),
        )
        end = time.time()
        if graph_2.num_nodes > graph_2.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")

    times = []
    print(f"KNN = {int(knn)} and V = {(NX1//POOLING_SIZE//POOLING_SIZE)*(NX2//POOLING_SIZE//POOLING_SIZE)*nx3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_3 = HyperCubeGraph(
            grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
            nx3=nx3,
            knn=int(knn),
            weight_comp_device=DEVICE,
            sigmas=(xi / eps, xi, 1.0),
        )
        end = time.time()
        if graph_3.num_nodes > graph_3.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32]:
        build_graphs(knn)
