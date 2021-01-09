import numpy as np
import torch
from gechebnet.graph.graph import HyperCubeGraph

NX1 = 28
NX2 = 28
NX3 = 8

POOLING_SIZE = 2

DEVICE = torch.device("cuda")

import time

NUM_ITER = 100


def build_graphs(knn):

    times = []
    print(f"KNN = {int(knn * POOLING_SIZE ** 4)} and V = {NX1*NX2*NX3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_1 = HyperCubeGraph(
            grid_size=(NX1, NX2), nx3=NX3, knn=int(knn * POOLING_SIZE ** 4), weight_comp_device=DEVICE
        )
        end = time.time()
        if graph_1.num_nodes > graph_1.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")

    times = []
    print(f"KNN = {int(knn * POOLING_SIZE ** 2)} and V = {(NX1//POOLING_SIZE)*(NX2//POOLING_SIZE)*NX3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_2 = HyperCubeGraph(
            grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
            nx3=NX3,
            knn=int(knn * POOLING_SIZE ** 2),
            weight_comp_device=DEVICE,
        )
        end = time.time()
        if graph_2.num_nodes > graph_2.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")

    times = []
    print(f"KNN = {int(knn)} and V = {(NX1//POOLING_SIZE//POOLING_SIZE)*(NX2//POOLING_SIZE//POOLING_SIZE)*NX3}")
    for _ in range(NUM_ITER):
        start = time.time()
        graph_3 = HyperCubeGraph(
            grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
            nx3=NX3,
            knn=int(knn),
            weight_comp_device=DEVICE,
        )
        end = time.time()
        if graph_3.num_nodes > graph_3.num_edges:
            print("Value Error: an error occured during the construction of the graph")
        times.append(end - start)
    print(f"time: mean {np.mean(times)} std {np.std(times)}")


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32]:
        build_graphs(knn)
