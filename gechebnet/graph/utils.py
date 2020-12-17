import torch


def metric_tensor(X, sigmas, device):
    """
    Return the anisotropic metric tensor, the main directions of the kernel are:
        1. aligned with theta and orthogonal to the orientation axis.
        2. orthogonal to theta and to the orientation axis.
        3. aligned with the orientation axis.

    Args:
        theta (float): the orientation of the first main direction of the kernel (in radians).
        sigmas (tuple): the intensities of the three main anisotropic directions.

    Returns:
        (torch.tensor): the metric tensor with shape (3, 3).
    """
    s1, s2, s3 = sigmas

    e1 = torch.stack((torch.cos(X[:, 2]), torch.sin(X[:, 2]), torch.zeros(X[:, 2].shape, device=device)), dim=1)

    e2 = torch.stack((-torch.sin(X[:, 2]), torch.cos(X[:, 2]), torch.zeros(X[:, 2].shape, device=device)), dim=1)

    e3 = torch.stack(
        (torch.zeros(X[:, 2].shape, device=device), torch.zeros(X[:, 2].shape, device=device), torch.ones(X[:, 2].shape, device=device)), dim=1
    )

    D = e1.unsqueeze(2) * e1.unsqueeze(1) * s1 + e2.unsqueeze(2) * e2.unsqueeze(1) * s2 + e3.unsqueeze(2) * e3.unsqueeze(1) * s3
    return D.reshape(-1, 9)
