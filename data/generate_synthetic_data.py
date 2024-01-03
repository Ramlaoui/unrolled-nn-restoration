import numpy as np
from scipy.linalg import convolution_matrix


identity_kernel = np.array([0, 1, 0])

def generate_kernel_h(length=50, type = 'random'):
    """
    Generate a convolution kernel h.

    Parameters:
    length (int): Length of the kernel.
    type (str): Type of the kernel. Can be 'gaussian', 'deriv', 'random', 'sparse_random', 'identity'.
    
    Returns:
    np.array: The generated convolution kernel.
    """
    if type == 'gaussian':
        h = np.exp(-np.linspace(-5, 5, length) ** 2)
        #h = h / np.sum(h)
    elif type == "gaussian_heteroscedastic":
        scale = np.random.uniform(0.5 + 1e-10, 1.5, length)
        h = np.exp(-((np.linspace(-5, 5, length)) ** 2) / (2 * scale**2))
        #h = h / np.sum(h)
    elif type == "deriv":
        h = np.array([-1, 2, -1])/2
    elif type =="sparse_random":
        h = np.zeros(length//2*2+1)
        h[length//2] = 1
        randomness =  (np.random.rand(length//2*2+1) < 0.15)*0.15
        h = h+randomness
        #h = h / np.sum(h)
    elif type == "hard_sparse_random":
        h = np.zeros(length//2*2+1)
        h[length//2] = 1
        randomness =  (np.random.rand(length//2*2+1) < 0.3)*0.3
        h = h+randomness
        #h = h / np.sum(h)
    elif type == "identity":
        h = identity_kernel
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


def generate_synthetic_sparse_signal(length=2000, sparsity=0.1, noise_level=0.05, h = identity_kernel):
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
    signal[non_zero_indices] = np.random.randn(num_nonzeros)
    
    # Apply H to the signal
    H = convolution_matrix(h, length) # mode = "full"
    signal_conv = np.convolve(signal, h, mode="full") # mode = "same"
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(signal_conv))
    noisy_signal = signal_conv + noise

    return noisy_signal, signal, H

def generate_synthetic_sparse_signal_2(length=2000, sparsity=0.1, noise_level=0.05, h = identity_kernel):
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
    signal[non_zero_indices] = np.abs(30+ np.random.randn(num_nonzeros)*30)
    
    # Apply H to the signal
    H = convolution_matrix(h, length) # mode = "full"
    signal_conv = np.convolve(signal, h, mode="full") # mode = "same"
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(signal_conv))
    noisy_signal = signal_conv + noise

    return noisy_signal, signal, H
