import torch
from torch import FloatTensor
from tqdm import tqdm


def repulsive_loss(x, alpha=1.0, beta=1.0):
    N = x.shape[0]

    dist = (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(dim=2) + 10000 * torch.eye(N, N).to(x.device)

    norm = x.norm(dim=1)

    return alpha * (1 / dist).mean() + beta * torch.abs(1.0 - norm).mean()


def repulsive_sampling(num_samples, loss_fn, radius=1.0, device=None, max_iter=10000):
    lr = 1e-3

    X = FloatTensor(num_samples, 3).uniform_(-1, 1).to(device)
    X.requires_grad_()

    for _ in tqdm(range(max_iter), desc="Repulsive sampling"):
        loss = loss_fn(X)
        loss.backward()

        with torch.no_grad():
            X -= lr * X.grad
            X.grad.zero_()

    X = X.detach().cpu()
    X /= X.norm(dim=1, keepdim=True) / radius

    return X[:, 0], X[:, 1], X[:, 2]
