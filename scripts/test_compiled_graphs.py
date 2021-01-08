import pykeops
import torch
from gechebnet.graph.graph import HyperCubeGraph

NX1 = 28
NX2 = 28
NX3 = 8

POOLING_SIZE = 2

DEVICE = torch.device("cuda")

from timeit import timeit


def compile_graphs(knn):

    print(f"KNN = {int(knn * POOLING_SIZE ** 4)}")
    print(f"V = {NX1*NX2*NX3}")
    print(
        f"time = {timeit('HyperCubeGraph(grid_size=(NX1, NX2), nx3=NX3, knn=int(knn * POOLING_SIZE ** 4), weight_comp_device=DEVICE)', number=100)}"
    )

    print(f"KNN = {int(knn * POOLING_SIZE ** 2)}")
    print(f"V = {(NX1//POOLING_SIZE)*(NX2//POOLING_SIZE)*NX3}")
    print(
        f"time = {timeit('HyperCubeGraph(grid_size=(NX1//POOLING_SIZE, NX2//POOLING_SIZE), nx3=NX3, knn=int(knn * POOLING_SIZE ** 2), weight_comp_device=DEVICE)', number=100)}"
    )

    print(f"KNN = {int(knn)}")
    print(f"V = {(NX1//POOLING_SIZE//POOLING_SIZE)*(NX2//POOLING_SIZE//POOLING_SIZE)*NX3}")
    print(
        f"time = {timeit('HyperCubeGraph(grid_size=(NX1//POOLING_SIZE//POOLING_SIZE, NX2//POOLING_SIZE//POOLING_SIZE), nx3=NX3, knn=int(knn), weight_comp_device=DEVICE)', number=100)}"
    )


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32]:
        compile_graphs(knn)
