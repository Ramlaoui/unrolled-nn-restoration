import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.utils import prox_primal, prox_dual


class PrimalDualLayer(nn.Module):
    def __init__(self, n, m, device="cpu"):
        super().__init__()
        self.n = n
        self.m = m
        self.relu = nn.ReLU()
        # self.linear_primal = nn.Linear(n, n, bias=False)
        # self.linear_dual = nn.Linear(n, n, bias=False)
        self.device = device
        self.tau = nn.Parameter(
            torch.FloatTensor([10]).to(self.device), requires_grad=True
        )
        self.gamma = nn.Parameter(
            torch.FloatTensor([0.01]).to(self.device), requires_grad=True
        )
        self.rho = nn.Parameter(
            torch.FloatTensor([1]).to(self.device),
            requires_grad=True,
        )

    def forward(self, x, yk, ykm1, z, H):
        batch_size = x.shape[0]
        tau = self.relu(self.tau)
        gamma = self.relu(self.gamma)
        rho = self.relu(self.rho)
        y_tilde = 2 * yk - ykm1

        # bpk = -tau*H^T*y_tilde
        if type(H) != torch.Tensor:
            # H is a convolutional layer
            bpk = -tau * (
                torch.nn.functional.conv_transpose1d(
                    y_tilde.reshape(batch_size, 1, -1),
                    H.weight,
                    padding=H.padding,
                )
                .reshape(batch_size, -1, 1)
                .transpose(1, 2)
            ).reshape(batch_size, -1)
        else:
            bpk = -tau * (
                H.transpose(1, 2) @ y_tilde.reshape(batch_size, -1, 1)
            ).reshape(batch_size, -1)
        # x = self.linear_primal(x) + bpk
        x = prox_primal(x + bpk, tau, device=self.device)

        # bdk = gamma*H*x
        if type(H) != torch.Tensor:
            # H is a convolutional layer
            bdk = gamma * (
                torch.nn.functional.conv1d(
                    x.reshape(batch_size, 1, -1),
                    H.weight,
                    padding=H.padding,
                )
                .reshape(batch_size, -1, 1)
                .transpose(1, 2)
            ).reshape(batch_size, -1)
        else:
            bdk = gamma * (H @ x.reshape(batch_size, -1, 1)).reshape(batch_size, -1)
        # y = self.linear_dual(yk) + bdk
        y = prox_dual(yk + bdk, gamma, z, rho)

        return x, y


class PrimalDual(nn.Module):
    def __init__(self, n, m, n_layers, learn_kernel=False, device="cpu"):
        super().__init__()
        self.model_name = "primal_dual"
        self.device = device
        self.n = n
        self.m = m
        self.learn_kernel = learn_kernel
        # # Learn kernel h and then use it to create H
        # #initialize to gaussian kernel
        # full padding to make the output bigger than the input
        # m > n
        if learn_kernel:
            kernel_size = (m - n) + 1
            padding = ((m - 1) + kernel_size - n) // 2
            self.h = torch.nn.Conv1d(
                1, 1, kernel_size, padding_mode="zeros", bias=False, padding=padding
            )
        self.layers = nn.ModuleList(
            [PrimalDualLayer(n, m, device=device) for _ in range(n_layers)]
        )

    def forward(self, z, H=None):
        if H is None:
            H = self.h
        x = torch.randn(z.shape[0], self.n, requires_grad=False, device=self.device)
        y = torch.randn(z.shape[0], self.m, requires_grad=False, device=self.device)
        ykm1 = torch.randn(z.shape[0], self.m, requires_grad=False)
        for layer in self.layers:
            x, y = layer(x, y, ykm1, z, H)
            ykm1 = y
        return x
