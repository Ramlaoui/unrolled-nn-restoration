import numpy as np
import torch
from tqdm import tqdm
from torch import nn


class PrimalDualTrainer:
    def __init__(self, model, criterion=None, optimizer=None, device=None):
        self.model = model
        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        self.optimizer = optimizer
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def train(self, z, x, n_epochs, batch_size=1):
        self.model.train()
        for epoch in tqdm(range(n_epochs)):
            for i in range(0, z.shape[0], batch_size):
                z_batch = torch.tensor(z[i : max(i + batch_size, z.shape[0])]).to(
                    self.device
                )
                x_batch = torch.tensor(x[i : max(i + batch_size, x.shape[0])]).to(
                    self.device
                )
                for j in range(len(x_batch)):
                    self.optimizer.zero_grad()
                    x_pred = self.model(z_batch[j])
                    loss = self.criterion(x_pred, x_batch[j])
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
