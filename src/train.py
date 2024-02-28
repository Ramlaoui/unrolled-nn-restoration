import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import snr, mae, mse, convmtx_torch


class SparseDataset(torch.utils.data.Dataset):
    def __init__(
        self, config, learn_kernel=True, load_kernel=False, data_type="training"
    ):
        super().__init__()
        self.config = config
        self.path = Path(config["data_path"]) / data_type
        self.z_path = self.path / "Degraded"
        self.x_path = self.path / "Groundtruth"

        self.learn_kernel = learn_kernel
        self.load_kernel = load_kernel
        if self.load_kernel:
            self.H_path = Path(config["data_path"]) / "H.npy"
            self.H = np.load(self.H_path)
        elif not (self.learn_kernel):
            self.H_path = self.path / "H"

        self.z_files = sorted(list(self.z_path.glob("*.npy")))
        self.x_files = sorted(list(self.x_path.glob("*.npy")))
        if not (self.learn_kernel) and not (self.load_kernel):
            self.H_files = sorted(list(self.H_path.glob("*.npy")))

    def __len__(self):
        return len(list(self.z_path.glob("*.npy")))

    def __getitem__(self, idx):
        z = np.load(self.z_files[idx])
        x = np.load(self.x_files[idx])
        if not (self.learn_kernel) and not (self.load_kernel):
            H = np.load(self.H_files[idx])
        elif self.load_kernel:
            H = self.H
        z = torch.from_numpy(z).float()
        x = torch.from_numpy(x).float()
        if not (self.learn_kernel) or self.load_kernel:
            H = torch.from_numpy(H).float()
            return z, (x, H)
        return z, x


class SingleTrainer:
    def __init__(
        self,
        model,
        config,
        learn_kernel=False,
        load_kernel=False,
        criterion=None,
        optimizer=None,
        model_path="models/",
        run_name=None,
        device=None,
        debug=False,
    ):
        self.model = model
        self.model_name = model.__class__.__name__
        self.model_path = Path(model_path)
        self.learn_kernel = learn_kernel
        self.load_kernel = load_kernel
        self.run_name = run_name
        if not (self.model_path.exists()):
            self.model_path.mkdir()
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
            self.logger.init(project="deep-unrolling", config=config, name=run_name)
            self.plot_every = 2

    def train(
        self,
        train_loader,
        n_epochs,
        validation_loader=None,
        save_best_model=False,
    ):
        if self.model.device != self.device:
            self.device = self.model.device
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            for z_batch, x_H in train_loader:
                z_batch = z_batch.to(self.device)
                if not (self.learn_kernel):
                    x_batch, H_batch = x_H
                    x_batch = x_batch.to(self.device)
                    H_batch = H_batch.to(self.device)
                elif self.load_kernel:
                    x_batch, H_true = x_H
                    x_batch = x_batch.to(self.device)
                    H_batch = None
                else:
                    x_batch = x_H.to(self.device)
                    H_batch = None
                self.optimizer.zero_grad()
                x_pred = self.model(z_batch, H_batch)
                batch_loss = self.criterion(x_pred, x_batch)
                barch_snr = snr(x_batch, x_pred)
                batch_mae = mae(x_batch, x_pred)
                batch_loss.backward()
                self.optimizer.step()
                if self.learn_kernel and self.load_kernel:
                    mse_h = mse(
                        convmtx_torch(self.model.h.weight.detach().reshape(-1), x_batch.shape[1]),
                        H_true[0],
                    )
                if not (self.debug):
                    self.logger.log({"loss": batch_loss.item()})
                    self.logger.log({"snr": barch_snr.item()})
                    self.logger.log({"mae": batch_mae.item()})

                    if self.learn_kernel and self.load_kernel:
                        self.logger.log(
                            {
                                "mse_h": mse_h,
                            }
                        )

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
                best_val_loss = np.inf
                z_val, (x_H_val) = next(iter(validation_loader))
                z_val = z_val.to(self.device)
                if not (self.learn_kernel):
                    x_val, H_val = x_H_val
                    x_val = x_val.to(self.device)
                    H_val = H_val.to(self.device)
                elif self.load_kernel:
                    x_val, H_true = x_H_val
                    x_val = x_val.to(self.device)
                    H_val = None
                else:
                    x_val = x_H_val.to(self.device)
                    H_val = None
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
                            plt.plot(
                                z_batch_plot,
                                label="Degraded signal"
                                + str(validation_loader.dataset.z_files[p]),
                            )
                            plt.legend()
                            self.logger.log(
                                {"val_loss": val_loss.item(), "prediction_example": plt}
                            )
                    print(f"Validation loss: {val_loss.item()}")
                if save_best_model and val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    if not (self.debug):
                        torch.save(
                            self.model.state_dict(),
                            self.model_path
                            / f"{self.model_name}_best_{self.run_name}.pt",
                        )
        print("Training finished, keeping best model...")
        if not (self.debug):
            self.model.load_state_dict(
                torch.load(
                    self.model_path / f"{self.model_name}_best_{self.run_name}.pt",
                    map_location=self.device,
                )
            )

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                self.model_path / f"{self.model_name}_best_{self.run_name}.pt",
                map_location=self.device,
            )
        )

    def test(self, test_loader, name=""):
        if self.model.device != self.device:
            self.device = self.model.device
        test_loss = 0
        for z_batch, x_H in test_loader:
            z_batch = z_batch.to(self.device)
            if not (self.learn_kernel):
                x_batch, H_batch = x_H
                x_batch = x_batch.to(self.device)
                H_batch = H_batch.to(self.device)
            else:
                x_batch = x_H.to(self.device)
                H_batch = None
            with torch.no_grad():
                x_pred = self.model(z_batch, H_batch)
                batch_loss = self.criterion(x_pred, x_batch)
                test_loss += batch_loss.item()
                test_snr = snr(x_batch, x_pred)
                test_mae = mae(x_batch, x_pred)
        print(name)
        print(f"Test loss: {test_loss/test_loader.__len__()}")
        print(f"Test SNR: {test_snr.item()}")
        print(f"Test MAE: {test_mae.item()}")
        if not (self.debug):
            self.logger.log({f"test_loss{name}": test_loss / test_loader.__len__()})
            self.logger.log({f"test_snr{name}": test_snr.item()})
            self.logger.log({f"test_mae{name}": test_mae.item()})
