import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from tqdm import tqdm
import yaml

from models import PrimalDual
from train import PrimalDualTrainer, SparseDataset


def generate_sparse_data(N, n_signal, max_peaks, window_length=5):
    assert window_length * 2 + 1 < n_signal
    n_peaks = np.random.randint(5, max_peaks, N)
    peaks_positions = np.random.randint(
        window_length + 1, n_signal - window_length - 1, (N, max(n_peaks))
    )
    x = np.zeros((N, n_signal))
    peak_signal = np.random.uniform(0.5, 20, (N, max(n_peaks)))
    for i in range(N):
        for j in range(n_peaks[i]):
            x[
                i,
                peaks_positions[i, j]
                - window_length : peaks_positions[i, j]
                + window_length
                + 1,
            ] += np.abs(
                np.sort(np.random.normal(0, 0.1, 2 * window_length + 1))
                * peak_signal[i, j]
            )
    noise = np.random.normal(0, 0.1, (N, n_signal))
    z = x + noise
    x = x.reshape(N, -1, 1).astype(np.float32)
    z = z.reshape(N, -1, 1).astype(np.float32)
    return z, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, default="data/data1.npz")
    parser.add_argument("--output", type=str, default="data/data1.npz")
    parser.add_argument("--validation_data", type=str, default="data/data_val.npz")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    # For these arguments only use them if they are inputted:
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    if args.generate:
        # Generate a dataset
        n_signal = 1000
        N = 1000
        max_peaks = 20
        z, x = generate_sparse_data(N, n_signal, max_peaks)
        np.savez(args.output, x=x, z=z)

        # Generate a validation dataset
        N_val = 100
        z_val, x_val = generate_sparse_data(N_val, n_signal, max_peaks)
        np.savez(args.validation_data, x=x_val, z=z_val)
        # plt.plot(x[0], label="Noisy signal")
        # plt.plot(z[0], label="Original signal")
        # plt.legend()
        # plt.show()

    else:
        data = np.load(args.input)
        x = data["x"].astype(np.float32)
        z = data["z"].astype(np.float32)
        n_signal = x.shape[1]
        N = x.shape[0]

    # Fill the config object with the args
    for arg in vars(args):
        if arg in config and (getattr(args, arg) is not None):
            config[arg] = getattr(args, arg)

    train_dataset = SparseDataset(config, data_type="training")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    config_test = config.copy()
    config_test["data_path"] = config["data_path"].replace("training", "validation")
    config_test["batch_size"] = 1
    validation_dataset = SparseDataset(config_test, data_type="validation")
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=True
    )

    n_signal = train_dataset[0][1].shape[0]
    m_signal = train_dataset[0][0].shape[0]
    model = PrimalDual(n_signal, m_signal, 2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    trainer = PrimalDualTrainer(
        model, config, criterion, optimizer, debug=args.is_debug
    )

    if args.validation_data is not None:
        validation_data = np.load(args.validation_data)

    trainer.train(
        train_loader,
        n_epochs=config["n_epochs"],
        validation_loader=validation_loader,
    )
