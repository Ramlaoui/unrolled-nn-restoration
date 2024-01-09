# unrolled-nn-restoration

This repository contains the content of the mini-project for the Machine Learning for Time Series course of the MVA master (2023-2024).

The paper implemented here is [Unrolled deep networks for sparse signal restoration](https://dumas.ccsd.cnrs.fr/INRIA-SACLAY/hal-03988686v2).

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Data

The datasets used in this project are the following:
- [MassBank](https://massbank.eu/MassBank/) (for the mass spectrometry data), with the same script used by (Gharbi et al., 2023) to degrade the dataset using specified kernels that can vary.
Generate the dataset using the following command:

```bash
python data/generate_dataset.py --Dataset massbank
```

- Synthetic data generated using the notebook `gen_sparse_signal_draft.ipynb`.

Note that the final generated datasets used for training the models are extremely large and not directly included in this repository.

We include a small dataset for testing purposes and in order to run the main notebook `main.ipynb` which shows how to train and test the models rapidly. However, the experiments in the report were run on the full datasets and were run using the scripts `src/main.py` and `src/experiments.py` along with a few notebooks for visualization purposes.

## Usage

The main notebook `1_presentation_notebook.ipynb` shows how to train and test the models rapidly and outline the main idea behind the different objects and functions in the source code.

In order to train a model, choose a config file in `configs/` (for example `primal_dual.yaml`) and run the following command:

```bash
python -m src.main --config configs/primal_dual.yaml --data_path data/massbank --n_epochs 10
```

It is also possible to specify multiple other parameters to script that will be used in the model and override the parameters in the config file or to create your own config file and directly use it. A more detailed description of the parameters can be found in `src/main.py`.

## Experiments

The experiments in the report were run using `src/experiments.py` and the configurations files are in `configs/experiments/`. The models are then saved in `models/` and the results were uploaded to [Weights & Biases](https://wandb.ai/).

An experiment consists of training a model on a dataset with specified parameters in the config file (or given as argument to the script) and then testing the model on datasets specified in the config file of the experiment. The results are then uploaded to [Weights & Biases](https://wandb.ai/) and can be visualized in the corresponding project.

In order to run the experiments, run the following command:

```bash
python -m src.experiments --config configs/experiments/learn_h_default_fixed_ista.yaml --n_epochs 30
```

If a model has already been trained and saved, it is possible to test it on a dataset by running the following command:

```bash
python -m src.experiments --config configs/experiments/learn_h_default_fixed_ista.yaml --test
```