import torch
import torch.nn as nn
import numpy as np
from src.utils import prox_primal

CHI = 4
GAMMA = 0.2


class ISTALayer(nn.Module):
    # define global Class variables

    def __init__(self, n, init_factor=1, device="cpu"):
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
            torch.FloatTensor([CHI * init_factor]).to(self.device),
            requires_grad=True,
        )
        self.gamma = nn.Parameter(
            torch.FloatTensor([GAMMA * init_factor]).to(self.device), requires_grad=True
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

        if type(H) != torch.Tensor:
            # H is a convolutional layer
            temp_conv = torch.nn.functional.conv1d(
                x.reshape(batch_size, 1, -1),
                H.weight,
                padding=H.padding,
            ).reshape(batch_size, -1, 1)
        else:
            temp_conv = H @ x.reshape(batch_size, -1, 1)

        temp = temp_conv - z.reshape(batch_size, -1, 1)
        if type(H) != torch.Tensor:
            # H is a convolutional layer
            bk = -gamma * (
                torch.nn.functional.conv_transpose1d(
                    temp.reshape(batch_size, 1, -1), H.weight, padding=H.padding
                )
                .reshape(batch_size, -1, 1)
                .transpose(1, 2)
            ).reshape(batch_size, -1)
        else:
            bk = -gamma * (H.transpose(1, 2) @ (temp)).reshape(batch_size, -1)
        x = prox_primal(x + bk, gamma * chi, device=self.device)

        return x


class ISTA(nn.Module):
    def __init__(
        self,
        n,
        m,
        n_layers,
        learn_kernel=False,
        init_factor=1,
        init_kernel="gaussian",
        device="cpu",
    ):
        super().__init__()
        self.model_name = "ista"
        self.device = device
        self.init_factor = init_factor
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
            if init_kernel == "gaussian":
                # Initialize kernel to gaussian
                self.h.weight.data = (
                    torch.from_numpy(
                        (1 / 2 * np.pi)
                        * np.exp(
                            -np.arange(
                                -(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1
                            )
                            ** 2
                            / 2
                        )
                    )
                    .reshape(1, 1, -1)
                    .float()
                    .to(self.device)
                )
        self.layers = nn.ModuleList(
            [
                ISTALayer(n, device=device, init_factor=init_factor)
                for _ in range(n_layers)
            ]
        )

    def forward(self, z, H=None):
        if H is None:
            H = self.h
        # x = torch.randn(z.shape[0], self.n, requires_grad=False, device=self.device)
        x = torch.zeros(z.shape[0], self.n, requires_grad=False, device=self.device)
        for layer in self.layers:
            x = layer(x, z, H)
        return x
