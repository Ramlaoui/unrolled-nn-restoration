import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.utils import prox_primal, prox_dual


class PrimalDualLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.relu = nn.ReLU()
        # self.linear_primal = nn.Linear(n, n, bias=False)
        # self.linear_dual = nn.Linear(n, n, bias=False)
        self.tau = nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)
        self.rho = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

    def forward(self, x, yk, ykm1, z, H):
        batch_size = x.shape[0]
        tau = self.relu(self.tau)
        gamma = self.relu(self.gamma)
        rho = self.relu(self.rho)
        y_tilde = 2 * yk - ykm1

        # bpk = -tau*H^T*y_tilde
        bpk = -tau * (H.transpose(1, 2) @ y_tilde.reshape(batch_size, -1, 1)).reshape(
            batch_size, -1
        )
        # x = self.linear_primal(x) + bpk
        x = prox_primal(x + bpk, tau)

        # bdk = gamma*H*x
        bdk = gamma * (H @ x.reshape(batch_size, -1, 1)).reshape(batch_size, -1)
        # y = self.linear_dual(yk) + bdk
        y = prox_dual(yk + bdk, gamma, z, rho)

        return x, y


class PrimalDual(nn.Module):
    def __init__(self, n, m, n_layers):
        super().__init__()
        self.n = n
        self.m = m
        self.layers = nn.ModuleList([PrimalDualLayer(n) for _ in range(n_layers)])

    def forward(self, z, H):
        x = torch.randn(z.shape[0], self.n, requires_grad=False)
        y = torch.randn(z.shape[0], self.m, requires_grad=False)
        ykm1 = torch.randn(z.shape[0], self.m, requires_grad=False)
        for layer in self.layers:
            x, y = layer(x, y, ykm1, z, H)
            ykm1 = y
        return x
