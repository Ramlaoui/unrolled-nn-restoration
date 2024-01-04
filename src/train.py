import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import wandb
import matplotlib.pyplot as plt
from pathlib import Path


class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_type="training"):
        super().__init__()
        self.config = config
        self.path = Path(config["data_path"]) / data_type
        self.z_path = self.path / "Degraded"
        self.x_path = self.path / "Groundtruth"
        self.H_path = self.path / "H"

        self.z_files = sorted(list(self.z_path.glob("*.npy")))
        self.x_files = sorted(list(self.x_path.glob("*.npy")))
        self.H_files = sorted(list(self.H_path.glob("*.npy")))

    def __len__(self):
        return len(list(self.z_path.glob("*.npy")))

    def __getitem__(self, idx):
        z = np.load(self.z_files[idx])
        x = np.load(self.x_files[idx])
        H = np.load(self.H_files[idx])
        z = torch.from_numpy(z).float()
        x = torch.from_numpy(x).float()
        H = torch.from_numpy(H).float()
        return z, x, H


class SingleTrainer:
    def __init__(
        self,
        model,
        config,
        criterion=None,
        optimizer=None,
        device=None,
        debug=False,
    ):
        self.model = model
        self.model_name = model.__class__.__name__
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

    def train(self, train_loader, n_epochs, validation_loader=None):
        if self.model.device != self.device:
            self.device = self.model.device
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            for z_batch, x_batch, H_batch in train_loader:
                z_batch = z_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                H_batch = H_batch.to(self.device)
                self.optimizer.zero_grad()
                x_pred = self.model(z_batch, H_batch)
                batch_loss = self.criterion(x_pred, x_batch)
                batch_loss.backward()
                self.optimizer.step()
                if not (self.debug):
                    self.logger.log({"loss": batch_loss.item()})
                    # Log all the weights
                    self.logger.log(
                        {
                            f"{name}": param.detach().cpu().numpy()
                            for name, param in self.model.named_parameters()
                        }
                    )
                epoch_loss += batch_loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss/train_loader.__len__()}")
            if validation_loader is not None:
                print("Calculating validation loss...")
                z_val, x_val, H_val = next(iter(validation_loader))
                z_val = z_val.to(self.device)
                x_val = x_val.to(self.device)
                H_val = H_val.to(self.device)
                with torch.no_grad():
                    x_pred_val = self.model(z_val, H_val)
                    val_loss = self.criterion(x_pred_val, x_val)
                    if not (self.debug):
                        if epoch % self.plot_every == 0:
                            # Pick random element form batch
                            p = np.random.randint(x_pred_val.shape[0])
                            x_pred_plot = x_pred_val.detach().cpu().numpy()[p]
                            x_val_plot = x_val.detach().cpu().numpy()[p]
                            z_batch_plot = z_val.detach().cpu().numpy()[p]
                            plt.plot(x_pred_plot, label="Restored signal")
                            plt.plot(x_val_plot, label="Original signal")
                            plt.plot(z_batch_plot, label="Degraded signal")
                            plt.legend()
                            self.logger.log(
                                {"val_loss": val_loss.item(), "prediction_example": plt}
                            )
                    print(f"Validation loss: {val_loss.item()}")
