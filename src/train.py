import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import wandb
import matplotlib.pyplot as plt


class PrimalDualTrainer:
    def __init__(
        self, model, config, criterion=None, optimizer=None, device=None, debug=False
    ):
        self.model = model
        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
            )
        self.optimizer = optimizer
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.debug = debug
        if not (debug):
            self.logger = wandb
            self.logger.init(project="deep-unrolling", config=config)
            self.plot_every = 2

    def train(self, z, x, n_epochs, batch_size=1, validation_data=None):
        if validation_data is not None:
            z_val, x_val = validation_data["z"], validation_data["x"]
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
                if not (self.debug):
                    self.logger.log({"loss": batch_loss.item()})
                epoch_loss += batch_loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss/(z.shape[0]/batch_size)}")
            if validation_data is not None:
                print("Calculating validation loss...")
                with torch.no_grad():
                    z_val = torch.tensor(z_val.reshape(-1, z.shape[1])).to(self.device)
                    x_val = torch.tensor(x_val.reshape(-1, x.shape[1])).to(self.device)
                    x_pred_val = self.model(z_val)
                    val_loss = self.criterion(x_pred_val, x_val)
                    if not (self.debug):
                        if epoch % self.plot_every == 0:
                            # Pick random element form batch
                            p = np.random.randint(x_pred_val.shape[0])
                            x_pred_plot = x_pred_val.detach().numpy()[p]
                            x_val_plot = x_val.detach().numpy()[p]
                            plt.plot(x_pred_plot, label="Restored signal")
                            plt.plot(x_val_plot, label="Original signal")
                            plt.legend()
                            self.logger.log(
                                {"val_loss": val_loss.item(), "prediction_example": plt}
                            )
                    print(f"Validation loss: {val_loss.item()}")
