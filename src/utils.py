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
        z + rho * (y[out_ball] - gamma * z) / (torch.norm(y[out_ball] - gamma * z))
    )
    return y
