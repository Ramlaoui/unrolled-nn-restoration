import torch
import torch.nn as nn

Soft = nn.Softplus()
# ReLU is the activation used during training, if local minima problems are encountered, try using Softplus.
r = nn.ReLU()


class R_Arch(torch.nn.Module):
    """"
    architectures to learn regularization parameter at each layer
    """ ""

    def __init__(self, n, m, Arch):
        super(R_Arch, self).__init__()
        self.n, self.m = n, m

        self.architecture = Arch

        if self.architecture == "lambda_Arch1":
            # U-HQ-FixN
            self.lambda_cvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == "lambda_Arch2":
            self.fc_cvx = nn.Linear(self.m, 1, bias=False)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.fc_ncvx = nn.Linear(self.m, 1, bias=False)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)

            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == "lambda_Arch1_cvx":
            self.lambda_cvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == "lambda_Arch1_ncvx":
            self.lambda_ncvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == "lambda_Arch2_cvx":
            self.fc_cvx = nn.Linear(self.m, 1, bias=False)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)
            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == "lambda_Arch2_ncvx":
            self.fc_ncvx = nn.Linear(self.m, 1, bias=False)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)
            self.gamma = nn.Parameter(torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == "lambda_Arch1_cvx_overparam":
            self.lambda_cvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

        if self.architecture == "lambda_Arch1_ncvx_overparam":
            self.lambda_ncvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

        if self.architecture == "lambda_Arch1_overparam":
            # U-HQ-FixN-OverP
            self.lambda_cvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_cvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

            self.lambda_ncvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.lambda_ncvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

        if self.architecture == "lambda_Arch2_overparam":
            # U-HQ
            self.fc_cvx = nn.Linear(self.m, 1, bias=True)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.fc_ncvx = nn.Linear(self.m, 1, bias=True)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

        if self.architecture == "lambda_Arch2_cvx_overparam":
            self.fc_cvx = nn.Linear(self.m, 1, bias=True)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

        if self.architecture == "lambda_Arch2_ncvx_overparam":
            self.fc_ncvx = nn.Linear(self.m, 1, bias=True)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.01, b=0.02)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True
            )

    def forward(self, H, x, xdeg):

        if self.architecture == "lambda_Arch1":
            lambda_cvx = r(self.lambda_cvx)
            lambda_ncvx = r(self.lambda_ncvx)
            gamma = r(self.gamma)
            return (gamma, lambda_cvx, lambda_ncvx)

        if self.architecture == "lambda_Arch2":
            res1 = ((torch.mm(H, x.T) - xdeg.T).T) ** 2
            lambda_cvx = r(self.fc_cvx(res1))
            lambda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma)
            return (gamma, lambda_cvx, lambda_ncvx)

        if self.architecture == "lambda_Arch1_cvx":
            lambda_cvx = r(self.lambda_cvx)
            gamma = r(self.gamma)
            return gamma, lambda_cvx, torch.zeros_like(lambda_cvx)

        if self.architecture == "lambda_Arch1_ncvx":
            lambda_ncvx = r(self.lambda_ncvx)
            gamma = r(self.gamma)
            return gamma, torch.zeros_like(lambda_ncvx), lambda_ncvx

        if self.architecture == "lambda_Arch1_cvx_overparam":
            lambda_cvx = r(
                self.lambda_cvx_1
                * self.lambda_cvx_2
                * self.lambda_cvx_3
                * self.lambda_cvx_4
                * self.lambda_cvx_5
                * self.lambda_cvx_6
                * self.lambda_cvx_7
                * self.lambda_cvx_8
                * self.lambda_cvx_9
                * self.lambda_cvx_10
            )
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )
            return gamma, lambda_cvx, torch.zeros_like(lambda_cvx)

        if self.architecture == "lambda_Arch1_ncvx_overparam":
            lambda_ncvx = r(
                self.lambda_ncvx_1
                * self.lambda_ncvx_2
                * self.lambda_ncvx_3
                * self.lambda_ncvx_4
                * self.lambda_ncvx_5
                * self.lambda_ncvx_6
                * self.lambda_ncvx_7
                * self.lambda_ncvx_8
                * self.lambda_ncvx_9
                * self.lambda_ncvx_10
            )
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )
            return gamma, torch.zeros_like(lambda_ncvx), lambda_ncvx

        if self.architecture == "lambda_Arch1_overparam":
            lambda_cvx = r(
                self.lambda_cvx_1
                * self.lambda_cvx_2
                * self.lambda_cvx_3
                * self.lambda_cvx_4
                * self.lambda_cvx_5
                * self.lambda_cvx_6
                * self.lambda_cvx_7
                * self.lambda_cvx_8
                * self.lambda_cvx_9
                * self.lambda_cvx_10
            )
            lambda_ncvx = r(
                self.lambda_ncvx_1
                * self.lambda_ncvx_2
                * self.lambda_ncvx_3
                * self.lambda_ncvx_4
                * self.lambda_ncvx_5
                * self.lambda_ncvx_6
                * self.lambda_ncvx_7
                * self.lambda_ncvx_8
                * self.lambda_ncvx_9
                * self.lambda_ncvx_10
            )
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )
            return gamma, lambda_cvx, lambda_ncvx

        if self.architecture == "lambda_Arch2_cvx":
            res1 = ((torch.mm(H, x.T) - xdeg.T).T) ** 2

            lambda_cvx = r(self.fc_cvx(res1))
            gamma = r(self.gamma)
            return (gamma, lambda_cvx, torch.zeros_like(lambda_cvx))

        if self.architecture == "lambda_Arch2_ncvx":

            res1 = ((torch.mm(H, x.T) - xdeg.T).T) ** 2
            lambda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma)
            return (gamma, torch.zeros_like(lambda_ncvx), lambda_ncvx)

        if self.architecture == "lambda_Arch2_overparam":
            res1 = (
                (torch.bmm(H, x.unsqueeze(dim=2)) - xdeg.unsqueeze(dim=2)).squeeze(
                    dim=2
                )
            ) ** 2
            lambda_cvx = r(self.fc_cvx(res1))
            lambda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )

            return (gamma, lambda_cvx, lambda_ncvx)

        if self.architecture == "lambda_Arch2_cvx_overparam":
            res1 = (
                (torch.bmm(H, x.unsqueeze(dim=2)) - xdeg.unsqueeze(dim=2)).squeeze(
                    dim=2
                )
            ) ** 2
            lambda_cvx = r(self.fc_cvx(res1))
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )
            return (gamma, lambda_cvx, torch.zeros_like(lambda_cvx))

        if self.architecture == "lambda_Arch2_ncvx_overparam":
            res1 = (
                (torch.bmm(H, x.unsqueeze(dim=2)) - xdeg.unsqueeze(dim=2)).squeeze(
                    dim=2
                )
            ) ** 2
            lambda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(
                self.gamma_1
                * self.gamma_2
                * self.gamma_3
                * self.gamma_4
                * self.gamma_5
                * self.gamma_6
                * self.gamma_7
                * self.gamma_8
                * self.gamma_9
                * self.gamma_10
            )
            return (gamma, torch.zeros_like(lambda_ncvx), lambda_ncvx)


"""Fair penalization"""


def phi_sfair(input, delta_s):
    y = delta_s * input / (abs(input) + delta_s)
    return y


def omega_sfair(u, delta_s):
    return delta_s / (abs(u) + delta_s)


def psi_sfair(input, delta):
    return delta * (torch.abs(input) - delta * torch.log(torch.abs(input) / delta + 1))


"""End Fair penalization"""

"""Tikhonov penalization"""


def phi_sTikhonov(input, delta_s):
    return 2 * input


def psi_sTikhonov(input, delta_s):
    return input**2


def omega_sTikhonov(input, delta_s):
    return 2 * torch.ones_like(input)


"""End Tikhonov penalization"""


"""Green Penlization"""


def psi_sgreen(input, delta):
    return torch.log(torch.cosh(input))


def phi_sgreen(input, delta):
    return torch.tanh(input)


def omega_sgreen(input, delta):
    return torch.tanh(input) / input


"""End Green penalization"""

"""Cauchy penalization"""


def phi_scauchy(input, delta_s):
    return (input * delta_s**2) / (delta_s**2 + input**2)


def psi_scauchy(input, delta_s):
    return 0.5 * delta_s**2 * torch.log(1 + (input**2) / delta_s**2)


def omega_scauchy(input, delta_s):
    return (delta_s**2) / (delta_s**2 + input**2)


"""End Cauchy penalization"""


"""Welsh penalization"""


def psi_swelsh(input, delta):
    return (delta**2) * (1 - torch.exp((-(input**2)) / (2 * delta**2)))


def phi_swelsh(input, delta):
    return (input) * torch.exp((-(input**2)) / (2 * delta**2))


def omega_swelsh(input, delta_s):
    return torch.exp((-(input**2)) / (2 * delta_s**2))


"""End Welsh penalization"""

"""Begin Geman MCClure"""


def phi_sGMc(input, delta):
    return (4 * (delta**4)) * input / (2 * (delta**2) + input**2) ** 2


def omega_sGMc(t, delta_s):
    return (4 * (delta_s**4)) / (2 * (delta_s**2) + t**2) ** 2


def psi_sGMc(input, delta):
    return (input**2) * (delta**2) / (2 * delta**2 + input**2)


"""end Geman MCClure"""
