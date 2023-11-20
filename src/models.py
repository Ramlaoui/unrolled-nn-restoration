import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def prox_primal(x, tau):
    return torch.mul(torch.sign(x), torch.max(torch.abs(x) - tau, torch.zeros_like(x)))


def prox_dual(y, gamma, z, rho):
    if torch.norm(1 / gamma * y - z) <= rho:
        return 0
    else:
        return y - gamma * (z + rho * (y - gamma * z) / (torch.norm(y - gamma * z)))


class PrimalDualLayer(nn.Module):
    def __init__(self, n, H):
        super().__init__()
        self.n = n
        self.relu = nn.ReLU()
        self.linear_primal = nn.Linear(n, n, bias=False)
        self.linear_dual = nn.Linear(n, n, bias=False)
        self.tau = self.relu(nn.Parameter(torch.randn(1)))
        self.gamma = self.relu(nn.Parameter(torch.randn(1)))
        self.rho = self.relu(nn.Parameter(torch.randn(1)))
        self.H = H

    def forward(self, x, yk, ykm1, z):
        y_tilde = 2 * yk - ykm1
        bpk = -self.tau * torch.mm(self.H.weight.T, y_tilde)
        x = self.linear_primal(x.reshape(-1)).reshape(-1, 1) + bpk
        x = prox_primal(x, self.tau)

        bdk = self.gamma * self.H(x.reshape(-1)).reshape(-1, 1)
        y = self.linear_dual(yk.reshape(-1)).reshape(-1, 1) + bdk
        y = prox_dual(y, self.gamma, z, self.rho)

        return x, y


class PrimalDual(nn.Module):
    def __init__(self, n, m, n_layers):
        super().__init__()
        self.n = n
        self.H = nn.Linear(m, n, bias=False)
        self.layers = nn.ModuleList(
            [PrimalDualLayer(n, self.H) for _ in range(n_layers)]
        )

    def forward(self, z):
        x = torch.zeros_like(z)
        y = torch.zeros_like(z)
        ykm1 = torch.zeros_like(z)
        for layer in self.layers:
            x, y = layer(x, y, ykm1, z)
            ykm1 = y
        return x
