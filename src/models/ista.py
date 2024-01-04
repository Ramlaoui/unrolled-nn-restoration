import torch
import torch.nn as nn
import numpy as np
from src.utils import prox_primal

chi = 4
gamma = 0.2
class ISTALayer(nn.Module):
    # define global Class variables
 
    
    def __init__(self, n, init_factor = 1,  device="cpu"):
        """
        Args:
            n (int): length of the signal
            init_factor (float, optional): factor to initialize chi and gamma with. Defaults to 1.
            device (str, optional): device to run the model on. Defaults to "cpu".
        """
        super().__init__()
        self.n = n
        self.device = device
        self.relu = nn.ReLU()
        self.chi = nn.Parameter(
            torch.FloatTensor([chi*init_factor]).to(self.device),
            requires_grad=True,
        )
        self.gamma = nn.Parameter(
            torch.FloatTensor([gamma*init_factor]).to(self.device), requires_grad=True
        )

    def forward(self, x, z, H):
        """
        Args:
            x (torch.Tensor): input signal
            z (torch.Tensor): measurement
            H (torch.Tensor): measurement matrix
        """
        batch_size = x.shape[0]

        chi = self.relu(self.chi)
        gamma = self.relu(self.gamma)

        temp = H @ x.reshape(batch_size, -1, 1) - z.reshape(batch_size, -1, 1)
        bk = -gamma * (H.transpose(1, 2) @ (temp)).reshape(batch_size, -1)
        x = prox_primal(x + bk, gamma * chi, device=self.device)

        return x


class ISTA(nn.Module):

    def __init__(self, n, m, n_layers, init_factor = 1, device="cpu"):
        super().__init__()
        self.model_name = "ista"
        self.device = device
        self.init_factor = init_factor
        self.n = n
        self.m = m
        self.layers = nn.ModuleList(
            [ISTALayer(n, device=device, init_factor=init_factor) for _ in range(n_layers)]
        )

    def forward(self, z, H):
        x = torch.randn(z.shape[0], self.n, requires_grad=False, device=self.device)
        for layer in self.layers:
            x = layer(x, z, H)
        return x
