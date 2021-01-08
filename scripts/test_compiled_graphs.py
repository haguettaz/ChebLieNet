import pykeops
import torch
from gechebnet.graph.graph import HyperCubeGraph

NX1 = 28
NX2 = 28
NX3 = 8

POOLING_SIZE = 2

DEVICE = torch.device("cuda")


def compile_graphs(knn):

    for eps in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for xi in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            graph_1 = HyperCubeGraph(
                grid_size=(NX1, NX2),
                nx3=NX3,
                knn=int(knn * POOLING_SIZE ** 4),
                sigmas=(xi / eps, xi, 1.0),
                weight_comp_device=DEVICE,
            )

            if graph_1.num_edges < graph_1.num_nodes:
                print(
                    f"compilation of graph_1 with size {(NX1, NX2, NX3)} failed : {graph_1.num_edges} edges for {graph_1.num_nodes} nodes"
                )

            graph_2 = HyperCubeGraph(
                grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
                nx3=NX3,
                knn=int(knn * POOLING_SIZE ** 2),
                sigmas=(xi / eps, xi, 1.0),
                weight_comp_device=DEVICE,
            )

            if graph_2.num_edges < graph_2.num_nodes:
                print(
                    f"compilation of graph_2 with size {(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE, NX3)} failed : {graph_2.num_edges} edges for {graph_2.num_nodes} nodes"
                )

            graph_3 = HyperCubeGraph(
                grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
                nx3=NX3,
                knn=int(knn),
                sigmas=(xi / eps, xi, 1.0),
                weight_comp_device=DEVICE,
            )

            if graph_3.num_edges < graph_3.num_nodes:
                print(
                    f"compilation of graph_3 with size {(NX1 // POOLING_SIZE// POOLING_SIZE, NX2 // POOLING_SIZE// POOLING_SIZE, NX3)} failed : {graph_3.num_edges} edges for {graph_3.num_nodes} nodes"
                )


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32]:
        compile_graphs(knn)
