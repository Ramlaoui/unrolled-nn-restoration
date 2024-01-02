import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from tqdm import tqdm
import yaml
from pathlib import Path

from src.models.primal_dual import PrimalDual
from src.models.ista import ISTA
from src.models.half_quadratic import HalfQuadratic
from src.train import SingleTrainer, SparseDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    # For these arguments only use them if they are inputted:
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()
    args.config = Path("configs/").joinpath(args.config + ".yaml")

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    default_config = yaml.load(
        open("configs/default.yaml", "r"), Loader=yaml.FullLoader
    )
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]

    print(f"Using config: {args.config}")
    if args.model_type is None:
        args.model_type = config["model_type"]

    assert args.model_type in ["primal_dual", "ista", "half_quadratic"]

    # Fill the config object with the args
    for arg in vars(args):
        if (arg in config and (getattr(args, arg) is not None)) or (arg not in config):
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
    device = torch.device(config["device"])
    if args.model_type == "ista":
        model = ISTA(n_signal, m_signal, config["n_layers"], device=device)
    elif args.model_type == "primal_dual":
        model = PrimalDual(n_signal, m_signal, config["n_layers"], device=device)
    elif args.model_type == "half_quadratic":
        model = HalfQuadratic(n_signal, m_signal, config["n_layers"], device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    trainer = SingleTrainer(
        model, config, criterion, optimizer, debug=config["is_debug"], device=device
    )

    trainer.train(
        train_loader,
        n_epochs=config["n_epochs"],
        validation_loader=validation_loader,
    )
