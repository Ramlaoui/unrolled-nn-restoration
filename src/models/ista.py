import torch
import torch.nn as nn
import numpy as np
from src.utils import prox_primal


class ISTALayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.chi = nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)

    def forward(self, x, z, H):
        batch_size = x.shape[0]

        chi = self.relu(self.chi)
        gamma = self.relu(self.gamma)

        temp = H @ x.reshape(batch_size, -1, 1) - z.reshape(batch_size, -1, 1)
        bk = -gamma * (H.transpose(1, 2) @ (temp)).reshape(batch_size, -1)
        x = prox_primal(x + bk, gamma * chi)

        return x


class ISTA(nn.Module):
    def __init__(self, n, m, n_layers):
        super().__init__()
        self.n = n
        self.m = m
        self.layers = nn.ModuleList([ISTALayer(n) for _ in range(n_layers)])

    def forward(self, z, H):
        x = torch.randn(z.shape[0], self.n, requires_grad=False)
        for layer in self.layers:
            x = layer(x, z, H)
        return x
