import torch
from gechebnet.graph.graph import HyperCubeGraph

MIN_KNN = 2
MULT_KNN = 2

NX1 = 20
NX2 = 20
NX3 = 20

POOLING_SIZE = 2

DEVICE = torch.device("cuda")


def compile_graphs(exp_knn):

    graph_1 = HyperCubeGraph(
        grid_size=(NX1, NX2),
        nx3=NX3,
        knn=int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 4),
        weight_comp_device=DEVICE,
    )

    if graph_1.num_edges < graph_1.num_nodes * int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 4) / 2:
        print(
            f"compilation of graph_1 with size {(NX1, NX2, NX3)} failed : {graph_1.num_edges} edges for {graph_1.num_nodes} nodes"
        )

    graph_2 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
        nx3=NX3,
        knn=int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 2),
        weight_comp_device=DEVICE,
    )

    if graph_2.num_edges < graph_2.num_nodes * int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 2) / 2:
        print(
            f"compilation of graph_2 with size {(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE, NX3)} failed : {graph_2.num_edges} edges for {graph_2.num_nodes} nodes"
        )

    graph_3 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
        nx3=NX3,
        knn=int(MIN_KNN * MULT_KNN ** exp_knn),
        weight_comp_device=DEVICE,
    )

    if graph_3.num_edges < graph_3.num_nodes * int(MIN_KNN * MULT_KNN ** exp_knn) / 2:
        print(
            f"compilation of graph_3 with size {(NX1 // POOLING_SIZE// POOLING_SIZE, NX2 // POOLING_SIZE// POOLING_SIZE, NX3)} failed : {graph_3.num_edges} edges for {graph_3.num_nodes} nodes"
        )


if __name__ == "__main__":
    pykeops.clean_pykeops()
    for exp_knn in [0, 1, 2, 3, 4]:
        compile_graphs(exp_knn)
