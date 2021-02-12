import math

import torch


def polyhedron_init(polyhedron: str):
    if polyhedron == "tetrahedron":
        vertices = torch.tensor(
            [[1, 0, -1 / math.sqrt(2)], [-1, 0, -1 / math.sqrt(2)], [0, 1, 1 / math.sqrt(2)], [0, -1, 1 / math.sqrt(2)]]
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])

    if polyhedron == "octahedron":
        vertices = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        faces = torch.tensor([[0, 1, 5], [1, 5, 2], [2, 5, 3], [0, 5, 3], [0, 3, 4], [0, 1, 4], [1, 2, 4], [2, 3, 4]])

    if polyhedron == "icosahedron":
        phi = (1 + math.sqrt(5)) / 2

        vertices = torch.tensor(
            [
                [phi, 1.0, 0.0],
                [-phi, 1.0, 0.0],
                [phi, -1.0, 0.0],
                [-phi, -1.0, 0.0],
                [1.0, 0.0, phi],
                [1.0, 0.0, -phi],
                [-1.0, 0.0, phi],
                [-1.0, 0.0, -phi],
                [0.0, phi, 1.0],
                [0.0, -phi, 1.0],
                [0.0, phi, -1.0],
                [0.0, -phi, -1.0],
            ]
        )
        faces = torch.tensor(
            [
                [0, 2, 4],
                [0, 2, 5],
                [0, 4, 8],
                [0, 8, 10],
                [0, 5, 10],
                [1, 3, 6],
                [1, 3, 7],
                [1, 6, 8],
                [1, 7, 10],
                [1, 8, 10],
                [2, 4, 9],
                [2, 5, 11],
                [2, 9, 11],
                [3, 7, 11],
                [3, 9, 11],
                [3, 6, 9],
                [4, 6, 9],
                [4, 6, 8],
                [5, 7, 10],
                [5, 7, 11],
            ]
        )
    return vertices, faces


def polyhedron_division(vertices, faces, level):
    vertices /= torch.norm(vertices, dim=1, keepdim=True)
    for _ in range(level):
        sub_faces = torch.stack((faces[:, [0, 1, 2]], faces[:, [2, 0, 1]], faces[:, [1, 2, 0]]), dim=1)
        vertices_1 = vertices[sub_faces[:, :, [0, 1]]].mean(dim=2)
        vertices_1 /= torch.norm(vertices_1, dim=2, keepdim=True)
        vertices_2 = vertices[sub_faces[:, :, [1, 2]]].mean(dim=2)
        vertices_2 /= torch.norm(vertices_2, dim=2, keepdim=True)

        N = vertices.shape[0]
        NV1 = vertices_1.shape[0] * vertices_1.shape[1]
        NV2 = vertices_2.shape[0] * vertices_1.shape[1]

        sub_faces[:, :, 0] = torch.arange(N, N + NV1).reshape(-1, 3)
        sub_faces[:, :, 2] = torch.arange(N + NV1, N + NV1 + NV2).reshape(-1, 3)

        faces = torch.cat((sub_faces, torch.arange(N, N + NV1).reshape(-1, 1, 3)), dim=1).reshape(-1, 3)
        vertices = torch.cat((vertices, vertices_1.reshape(-1, 3), vertices_2.reshape(-1, 3)), dim=0)

    vertices = vertices.unique(dim=0)

    return vertices[:, 0], vertices[:, 1], vertices[:, 2]
