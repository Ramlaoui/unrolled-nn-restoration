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
