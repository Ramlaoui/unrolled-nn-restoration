import numpy as np
import torch
import matplotlib.pyplot as plt
from src.train import SparseDataset, SingleTrainer
from src.models.primal_dual import PrimalDual
from src.models.ista import ISTA
from pathlib import Path
import yaml
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--model_path", type=str, default="models/")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--init_factor", type=float, default=1)

    # For these arguments only use them if they are inputted:
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--learn_kernel", type=bool)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    args.config = Path("configs/").joinpath(args.config + ".yaml")

    config_all = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config = config_all["train"]
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

    train_dataset = SparseDataset(
        config, learn_kernel=config["learn_kernel"], data_type="training"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    config_val = config.copy()

    validation_dataset = SparseDataset(
        config_val, learn_kernel=config["learn_kernel"], data_type="validation"
    )

    config_val["data_path"] = config["data_path"].replace("training", "validation")
    config_val["batch_size"] = len(validation_dataset)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False
    )

    if config["learn_kernel"]:
        n_signal = train_dataset[0][1].shape[0]
    else:
        n_signal = train_dataset[0][1][0].shape[0]
    m_signal = train_dataset[0][0].shape[0]
    device = torch.device(config["device"])

    init_kernel = config["init_kernel"] if "init_kernel" in config else "gaussian"

    if args.model_type == "ista":
        model = ISTA(
            n_signal,
            m_signal,
            config["n_layers"],
            learn_kernel=config["learn_kernel"],
            device=device,
            init_factor=config["init_factor"],
            init_kernel=init_kernel,
        )
    elif args.model_type == "primal_dual":
        model = PrimalDual(
            n_signal,
            m_signal,
            config["n_layers"],
            learn_kernel=config["learn_kernel"],
            init_factor=config["init_factor"],
            device=device,
            init_kernel=init_kernel,
        )

    model.to(device)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    trainer = SingleTrainer(
        model,
        config,
        learn_kernel=config["learn_kernel"],
        criterion=criterion,
        optimizer=optimizer,
        debug=config["is_debug"],
        model_path=args.model_path,
        run_name=str(args.config).split("/")[-1].split(".")[0],
        device=device,
    )

    if not (args.test):
        trainer.train(
            train_loader,
            n_epochs=config["n_epochs"],
            validation_loader=validation_loader,
            save_best_model=True,
        )
    else:
        trainer.load_model()

    config_test = config.copy()
    config_test["batch_size"] = len(validation_dataset)

    for key in config_all:
        if "test" in key:
            config_test["data_path"] = config_all[key]["data_path"]
            test_dataset = SparseDataset(
                config_test, learn_kernel=config["learn_kernel"], data_type="test"
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=config_test["batch_size"], shuffle=True
            )

            trainer.test(test_loader, name=f"_{key}")
