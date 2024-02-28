import torch


def prox_primal(x, tau, device="cpu"):
    return torch.mul(
        torch.sign(x),
        torch.maximum(
            torch.abs(x) - tau, torch.zeros_like(x, requires_grad=False, device=device)
        ),
    )


def prox_dual(y, gamma, z, rho):
    batch_wise_norm = torch.norm(1 / gamma * y - z, dim=1)
    in_ball = batch_wise_norm <= rho
    out_ball = batch_wise_norm > rho
    y[in_ball] = 0
    y[out_ball] = y[out_ball] - gamma * (
        z[out_ball]
        + rho
        * (y[out_ball] - gamma * z[out_ball])
        / (torch.norm(y[out_ball] - gamma * z[out_ball]))
    )
    return y


def snr(x, x_pred):
    return 10 * torch.log10(torch.sum(x**2) / torch.sum((x - x_pred) ** 2))


def mae(x, x_pred):
    return torch.mean(torch.abs(x - x_pred))

def mse(x, x_pred):
    return torch.mean((x - x_pred) ** 2)

def convmtx_torch(h, n_in, device="cpu"):
    """
    Generates a convolution matrix H
    such that the product of H and an i_n element vector
    x is the convolution of x and h.

    Usage: H = convm(h,n_in)
    Given a column vector h of length N, an (N+n_in-1 x n_in)_in convolution matrix is
    generated

    This method has the same functionning as that of np.convolve(x,h)
    """
    N = len(h)
    N1 = N + 2 * n_in - 2
    hpad = torch.concatenate([torch.zeros(n_in - 1, device=device), h[:], torch.zeros(n_in - 1, device=device)])

    H = torch.zeros((len(h) + n_in - 1, n_in), device=device)
    for i in range(n_in):
        H[:, i] = hpad[n_in - i - 1 : N1 - i]
    return H