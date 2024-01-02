import torch
import torch.nn as nn
import numpy as np
from src.utils import prox_primal


class ISTALayer(nn.Module):
    def __init__(self, n, device="cpu"):
        super().__init__()
        self.n = n
        self.device = device
        self.relu = nn.ReLU()
        self.chi = nn.Parameter(
            torch.FloatTensor([4]), requires_grad=True, device=self.device
        )
        self.gamma = nn.Parameter(
            torch.FloatTensor([0.2]), requires_grad=True, device=self.device
        )

    def forward(self, x, z, H):
        batch_size = x.shape[0]

        chi = self.relu(self.chi)
        gamma = self.relu(self.gamma)

        temp = H @ x.reshape(batch_size, -1, 1) - z.reshape(batch_size, -1, 1)
        bk = -gamma * (H.transpose(1, 2) @ (temp)).reshape(batch_size, -1)
        x = prox_primal(x + bk, gamma * chi, device=self.device)

        return x


class ISTA(nn.Module):
    def __init__(self, n, m, n_layers, device="cpu"):
        super().__init__()
        self.model_name = "ista"
        self.device = device
        self.n = n
        self.m = m
        self.layers = nn.ModuleList(
            [ISTALayer(n, device=device) for _ in range(n_layers)]
        )

    def forward(self, z, H):
        x = torch.randn(z.shape[0], self.n, requires_grad=False, device=self.device)
        for layer in self.layers:
            x = layer(x, z, H)
        return x
