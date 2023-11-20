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
            optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer = optimizer
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def train(self, z, x, n_epochs, batch_size=1):
        self.model.train()
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            for i in range(0, z.shape[0], batch_size):
                self.optimizer.zero_grad()
                z_batch = torch.tensor(
                    z[i : min(i + batch_size, z.shape[0])].reshape(-1, z.shape[1])
                ).to(self.device)
                x_batch = torch.tensor(
                    x[i : min(i + batch_size, x.shape[0])].reshape(-1, x.shape[1])
                ).to(self.device)
                x_pred = self.model(z_batch)
                batch_loss = self.criterion(x_pred, x_batch)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss/(z.shape[0]/batch_size)}")
