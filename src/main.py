import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from tqdm import tqdm

from models import PrimalDual
from train import PrimalDualTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/data1.npz")
    parser.add_argument("--output", type=str, default="data/data1.npz")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.generate:
        # Generate a dataset
        n_signal = 1000
        N = 200
        n_peaks = np.random.randint(5, 100, N)
        peaks_positions = np.random.randint(0, n_signal, (N, max(n_peaks)))
        x = np.zeros((N, n_signal))
        x[np.arange(N).reshape(-1, 1), peaks_positions] = 1
        noise = np.random.normal(0, 0.1, (N, n_signal))
        z = x + noise
        x = x.reshape(N, -1, 1).astype(np.float32)
        z = z.reshape(N, -1, 1).astype(np.float32)

        np.savez(args.output, x=x, z=z)

        plt.plot(x[0])
        plt.plot(x[1])
        plt.show()
    else:
        data = np.load(args.input)
        x = data["x"].astype(np.float32)
        z = data["z"].astype(np.float32)
        n_signal = x.shape[1]
        N = x.shape[0]

    model = PrimalDual(n_signal, n_signal, 2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = PrimalDualTrainer(model, criterion, optimizer)

    trainer.train(z, x, args.n_epochs, args.batch_size)
