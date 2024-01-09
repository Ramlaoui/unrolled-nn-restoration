import numpy as np
from scipy.linalg import convolution_matrix
import argparse

identity_kernel = np.array([0, 1, 0])


def generate_kernel_h(length=50, type="random"):
    """
    Generate a convolution kernel h.

    Parameters:
    length (int): Length of the kernel.
    type (str): Type of the kernel. Can be 'gaussian', 'deriv', 'random', 'sparse_random', 'identity'.

    Returns:
    np.array: The generated convolution kernel.
    """
    length += 1
    if type == "gaussian":
        h = (
            1
            / (np.sqrt(2 * np.pi))
            * np.exp(-((np.linspace(-20, 20, length)) ** 2) / 2)
        )
        # h = h / np.sum(h)
    elif type == "gaussian_heteroscedastic":
        scale = np.random.uniform(0.5 + 1e-10, 1.5, length)
        h = (
            1
            / (scale * np.sqrt(2 * np.pi))
            * np.exp(-((np.linspace(-20, 20, length)) ** 2) / (2 * scale**2))
        )
        # h = h / np.sum(h)
    elif type == "deriv":
        h = np.zeros(length)
        h[length // 2] = 2
        h[length // 2 - 1] = -1
        h[length // 2 + 1] = -1
    elif type == "sparse_random":
        h = np.zeros(length // 2 * 2 + 1)
        h[length // 2] = 1
        randomness = (np.random.rand(length // 2 * 2 + 1) < 0.15) * 0.15
        h = h + randomness
        # h = h / np.sum(h)
    elif type == "hard_sparse_random":
        h = np.zeros(length // 2 * 2 + 1)
        h[length // 2] = 1
        randomness = (np.random.rand(length // 2 * 2 + 1) < 0.3) * 0.3
        h = h + randomness
        # h = h / np.sum(h)
    elif type == "identity":
        h = np.zeros(length)
        h[length // 2] = 1
    else:
        raise ValueError("Invalid kernel type")

    return h


def convmtx(h, n_in):
    """
    Generates a convolution matrix H from a kernel h applied to a signal x of length n_in.
    Parameters:
    h (np.array): The kernel.
    n_in (int): Length of the signal.
    Returns:
    np.array: The convolution matrix H of shape (n_in + len(h) - 1, n_in).
    usage:
    H = convmtx(h, len(x))
    H @ x == np.convolve(h, x)
    """
    N = len(h)
    N1 = N + 2 * n_in - 2
    hpad = np.concatenate([np.zeros(n_in - 1), h[:], np.zeros(n_in - 1)])

    H = np.zeros((len(h) + n_in - 1, n_in))
    for i in range(n_in):
        H[:, i] = hpad[n_in - i - 1 : N1 - i]
    return H


def generate_synthetic_sparse_signal(
    length=2000, sparsity=0.1, noise_level=0.05, h=identity_kernel
):
    """
    Generate a synthetic sparse signal.

    Parameters:
    length (int): Length of the signal.
    sparsity (float): Fraction of non-zero elements in the signal.
    noise_level (float): Standard deviation of Gaussian noise added to the signal.
    h (np.array): The convolution kernel used to generate the signal.

    Returns:
    np.array: The generated sparse signal with noise.
    """
    # Create a sparse signal
    signal = np.zeros(length)
    num_nonzeros = int(length * sparsity)
    non_zero_indices = np.random.choice(length, num_nonzeros, replace=False)
    signal[non_zero_indices] = np.abs(30 + np.random.randn(num_nonzeros) * 30)

    # Apply H to the signal
    H = convolution_matrix(h, length)  # mode = "full"
    signal_conv = np.convolve(signal, h, mode="full")  # mode = "same"
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(signal_conv))
    noisy_signal = signal_conv + noise

    return noisy_signal, signal, H


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="synthetic",
    help="Name of the dataset.",
)


parser.add_argument(
    "--length", type=int, default=2000, help="Length of the signal to be generated."
)
parser.add_argument(
    "--sparsities",
    nargs="+",
    default=[0.1],
    help="Fraction of non-zero elements in the signal.",
)

parser.add_argument(
    "--noise_levels",
    nargs="+",
    default=[0.05],
    help="Standard deviation of Gaussian noise added to the signal.",
)

parser.add_argument(
    "--kernel_types",
    nargs="+",
    default=["identity"],
    help="Type of the kernel. Can be 'gaussian', 'gaussian_heteroscedastic', 'deriv', 'random', 'sparse_random', 'identity'.",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    help="Number of samples to be generated per parameters combination.",
)

sparsity = parser.parse_args().sparsities
noise_level = parser.parse_args().noise_levels
kernel_types = parser.parse_args().kernel_types
length = parser.parse_args().length
num_samples = parser.parse_args().num_samples
dataset_name = parser.parse_args().dataset_name

sparsity = [float(s) for s in sparsity]
noise_level = [float(n) for n in noise_level]

print("Kernel types:")
print(kernel_types)
print("Noise levels:")
print(noise_level)
print("Sparsities:")
print(sparsity)

# Create dataset folder
import shutil
import os

datasetRoot = os.path.join("./data", dataset_name)

try:
    shutil.rmtree(datasetRoot)
except:
    pass
os.makedirs(datasetRoot)

# Create training, validation and test folders
# Training
training_path = os.path.join(datasetRoot, "training")
os.makedirs(training_path)
training_path_Groundtruth = os.path.join(training_path, "Groundtruth")
os.makedirs(training_path_Groundtruth)
training_path_Degraded2000 = os.path.join(training_path, "Degraded")
os.makedirs(training_path_Degraded2000)
training_path_H = os.path.join(training_path, "H")
os.makedirs(training_path_H)
# Validation
validation_path = os.path.join(datasetRoot, "validation")
os.makedirs(validation_path)
validation_path_Groundtruth = os.path.join(validation_path, "Groundtruth")
os.makedirs(validation_path_Groundtruth)
validation_path_Degraded2000 = os.path.join(validation_path, "Degraded")
os.makedirs(validation_path_Degraded2000)
validation_path_H = os.path.join(validation_path, "H")
os.makedirs(validation_path_H)
# Test
test_path = os.path.join(datasetRoot, "test")
os.makedirs(test_path)
test_path_Groundtruth = os.path.join(test_path, "Groundtruth")
os.makedirs(test_path_Groundtruth)
test_path_Degraded2000 = os.path.join(test_path, "Degraded")
os.makedirs(test_path_Degraded2000)
test_path_H = os.path.join(test_path, "H")
os.makedirs(test_path_H, exist_ok=True)

# parametes combinations
parameters = [(s, n, k) for s in sparsity for n in noise_level for k in kernel_types]
print(len(parameters))

# generate original_signal, noisy_signal, H for each parameter combination
count = 0
for s, n, k in parameters:
    print(s, n, k)
    print(count, "th out of", len(parameters))
    for i in range(num_samples):
        count += 1
        if i < num_samples * 0.6:
            path = training_path
        elif i < num_samples * 0.8:
            path = validation_path
        else:
            path = test_path
        h = generate_kernel_h(length=50, type=k)
        degraded_signal, signal, H = generate_synthetic_sparse_signal(
            length=length, sparsity=s, noise_level=n, h=h
        )
        np.save(
            os.path.join(
                path, "Groundtruth", "x_Gr_tr_{}_{}_{}_{}.npy".format(s, n, k, count)
            ),
            signal,
        )
        np.save(
            os.path.join(
                path, "Degraded", "x_De_tr_{}_{}_{}_{}.npy".format(s, n, k, count)
            ),
            degraded_signal,
        )
        np.save(
            os.path.join(path, "H", "H_tr_{}_{}_{}_{}.npy".format(s, n, k, count)), H
        )
