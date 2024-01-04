import torch
import torch.nn as nn
import numpy as np
from time import time


def psi(x, delta=1e-3):
    # Fair Potential, approximates the L1 norm with a smooth function
    return delta * (torch.abs(x) - delta * torch.log(1 + torch.abs(x) / delta))


def psi_dot(x, delta=1e-3):
    return torch.sign(x) * (1 - delta / (delta + torch.abs(x)))


def w(x, delta=1e-3):
    return psi_dot(x, delta) / x


def build_majorant_metric(x, H, lambd, delta=1e-3):
    batch_size = x.shape[0]
    L = torch.eye(x.shape[1], device=x.device)
    ws = w(x, delta)
    Ax = (
        H.transpose(1, 2) @ H + lambd * L.T @ torch.diag_embed(ws, dim1=-2, dim2=-1) @ L
    )
    return Ax


def F(x, H, z, lambd, delta=1e-3):
    return 0.5 * torch.norm(H @ x - z) ** 2 + lambd * torch.sum(psi(x, delta))


def F_grad(x, H, z, lambd, delta=1e-3):
    batch_size = x.shape[0]
    return (
        H.transpose(1, 2)
        @ (H @ x.reshape(batch_size, -1, 1) - z.reshape(batch_size, -1, 1))
    ).reshape(batch_size, -1) + lambd * psi_dot(x, delta)


class HalfQuadraticLayer(nn.Module):
    def __init__(self, n, device="cpu"):
        super().__init__()
        self.n = n
        self.device = device
        self.relu = nn.ReLU()
        self.lambd = nn.Parameter(
            torch.FloatTensor([1]).to(self.device), requires_grad=True
        )
        self.time_to_invert = 0

    def forward(self, x, z, H):
        batch_size = x.shape[0]

        lambd = self.relu(self.lambd)

        Ax = build_majorant_metric(x, H, lambd, delta=1e-1)
        start_time = time()
        Ax_inv = torch.inverse(Ax)
        self.time_to_invert = time() - start_time
        grad_F = F_grad(x, H, z, lambd)

        x = x - (Ax_inv @ grad_F.reshape(batch_size, -1, 1)).reshape(batch_size, -1)
        return x


class HalfQuadratic(nn.Module):
    def __init__(self, n, m, n_layers, device="cpu"):
        super().__init__()
        self.model_name = "ista"
        self.device = device
        self.n = n
        self.m = m
        self.layers = nn.ModuleList(
            [HalfQuadraticLayer(n, device=device) for _ in range(n_layers)]
        )

    def forward(self, z, H):
        x = torch.randn(z.shape[0], self.n, requires_grad=False, device=self.device)
        for layer in self.layers:
            x = layer(x, z, H)
        return x
