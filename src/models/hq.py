# Implementation from https://github.com/GHARBIMouna/Unrolled-Half-Quadratic/blob/main/Unrolled_methods/U_HQ/model.py

import torch.nn as nn
import numpy as np
import torch
from src.models.hq_utils import R_Arch
import src.models.hq_utils as hq
from src.utils import convmtx_torch

r = nn.ReLU()
Soft = nn.Softplus()


def MM(
    x,
    H,
    L,
    delta_s_cvx,
    delta_s_ncvx,
    lamda_cvx,
    lamda_ncvx,
    penalization_number_cvx,
    penalization_number_ncvx,
    mode,
):
    penal_name_cvx = "hq.omega_s" + str(penalization_number_cvx)
    penal_name_ncvx = "hq.omega_s" + str(penalization_number_ncvx)
    if mode == "learning_lambda_MM" or mode == "Deep_equilibrium":
        Diag_cvx = torch.diag_embed(
            eval(penal_name_cvx + "(torch.matmul(L,x.T).T,delta_s_cvx)")
        )
        Diag_ncvx = torch.diag_embed(
            eval(penal_name_ncvx + "(torch.matmul(L,x.T).T,delta_s_ncvx)")
        )
        A = torch.inverse(
            torch.bmm(torch.transpose(H, 1, 2), H)
            + torch.mul(Diag_cvx, lamda_cvx.unsqueeze(1))
            + torch.mul(Diag_ncvx, lamda_ncvx.unsqueeze(1))
        )
    return A


class Iter(torch.nn.Module):
    def __init__(self, n, m, mode, architecture_lambda):
        super(Iter, self).__init__()
        self.mode = mode
        self.architecture_name = architecture_lambda
        if mode == "learning_lambda_MM":
            self.architecture = R_Arch(n, m, architecture_lambda)

    def forward(
        self,
        x,
        z,
        Ht_x_degraded,
        H,
        L,
        delta_s_cvx,
        delta_s_ncvx,
        penalization_number_cvx,
        penalization_number_ncvx,
        Disp_param,
        lamda_cvx=None,
        lamda_ncvx=None,
    ):

        penal_name_cvx = "hq.phi_s" + str(penalization_number_cvx)
        penal_name_ncvx = "hq.phi_s" + str(penalization_number_ncvx)

        if self.mode == "learning_lambda_MM":
            gamma, lamda_cvx, lamda_ncvx = self.architecture(H, x, z)

            first_branch = (
                torch.bmm(torch.bmm(torch.transpose(H, 1, 2), H), x.unsqueeze(dim=2))
                - Ht_x_degraded
            )

            second_branch = eval(penal_name_cvx + "(x.unsqueeze(dim=2),delta_s_cvx)")
            second_branch1 = eval(penal_name_ncvx + "(x.unsqueeze(dim=2),delta_s_ncvx)")

            if self.architecture_name in [
                "lambda_Arch1",
                "lambda_Arch1_cvx",
                "lambda_Arch1_ncvx",
                "lamda_Arch1_overparam",
                "lamda_Arch1_cvx_overparam",
                "lamda_Arch1_ncvx_overparam",
            ]:
                summ = (
                    lamda_cvx * second_branch
                    + first_branch
                    + lamda_ncvx * second_branch1
                )
            else:
                summ = (
                    lamda_cvx.unsqueeze(dim=2) * second_branch
                    + first_branch
                    + lamda_ncvx.unsqueeze(dim=2) * second_branch1
                )

            inv_A = MM(
                x,
                H,
                L,
                delta_s_cvx,
                delta_s_ncvx,
                lamda_cvx,
                lamda_ncvx,
                penalization_number_cvx,
                penalization_number_ncvx,
                self.mode,
            )
            x = (x.unsqueeze(dim=2) - gamma * torch.bmm(inv_A, summ)).squeeze(dim=2)

        if self.mode == "Deep_equilibrium":

            first_branch = (
                torch.bmm(torch.bmm(torch.transpose(H, 1, 2), H), x.unsqueeze(dim=2))
                - Ht_x_degraded
            )

            second_branch = eval(penal_name_cvx + "(x.unsqueeze(dim=2),delta_s_cvx)")
            second_branch1 = eval(penal_name_ncvx + "(x.unsqueeze(dim=2),delta_s_ncvx)")
            summ = (
                lamda_cvx * second_branch + first_branch + lamda_ncvx * second_branch1
            )
            inv_A = MM(
                x,
                H,
                L,
                delta_s_cvx,
                delta_s_ncvx,
                lamda_cvx,
                lamda_ncvx,
                penalization_number_cvx,
                penalization_number_ncvx,
                self.mode,
            )
            x = (x.unsqueeze(dim=2) - torch.bmm(inv_A, summ)).squeeze(dim=2)

        if Disp_param == True:
            return x, lamda_cvx, lamda_ncvx, gamma

        if Disp_param == False:
            return x


class Block(torch.nn.Module):

    def __init__(self, n, m, mode, architecture_lambda):
        super(Block, self).__init__()

        self.Iter = Iter(n, m, mode, architecture_lambda)

    def forward(
        self,
        x,
        z,
        Ht_x_degraded,
        H,
        L,
        delta_s_cvx,
        delta_s_ncvx,
        penalization_num,
        penalization_num1,
        Disp_param,
        lamda_cvx=None,
        lamda_ncvx=None,
    ):

        return self.Iter(
            x,
            z,
            Ht_x_degraded,
            H,
            L,
            delta_s_cvx,
            delta_s_ncvx,
            penalization_num,
            penalization_num1,
            Disp_param,
            lamda_cvx,
            lamda_ncvx,
        )


class HalfQuadratic(torch.nn.Module):

    def __init__(
        self,
        n,
        m,
        n_layers,
        L,
        delta_s_cvx,
        delta_s_ncvx,
        mode,
        number_penalization_cvx,
        number_penalization_ncvx,
        architecture_lambda,
        learn_kernel=False,
        init_kernel="uniform",
        device="cpu",
        Disp_param=False,
    ):
        super(HalfQuadratic, self).__init__()

        self.n = n
        self.m = m
        self.Layers = nn.ModuleList()
        self.device = device
        self.L = L
        self.delta_s_cvx = delta_s_cvx
        self.delta_s_ncvx = delta_s_ncvx
        self.mode = mode
        self.Disp_param = Disp_param
        self.learn_kernel = learn_kernel
        self.number_penalization_cvx = number_penalization_cvx
        self.number_penalization_ncvx = number_penalization_ncvx
        self.architecture_lambda = architecture_lambda

        if self.learn_kernel:
            kernel_size = (m - n) + 1
            padding = ((m - 1) + kernel_size - n) // 2
            self.h = torch.nn.Conv1d(
                1, 1, kernel_size, padding_mode="zeros", bias=False, padding=padding
            )
            if init_kernel == "gaussian":
                # Initialize kernel to gaussian
                self.h.weight.data = (
                    torch.from_numpy(
                        (1 / 2 * np.pi)
                        * np.exp(
                            -np.arange(
                                -(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1
                            )
                            ** 2
                            / 2
                        )
                    )
                    .reshape(1, 1, -1)
                    .float()
                    .to(self.device)
                )

        for i in range(n_layers):
            self.Layers.append(Block(self.n, self.m, mode, architecture_lambda))

        if mode == "Deep_equilibrium" or mode == "Deep_equilibrium_3MG":
            self.architecture = R_Arch(architecture_lambda)

    def forward(self, z, H):
        mode, Disp_param = self.mode, self.Disp_param

        if Disp_param == True:
            lambdas_cvx = []
            lambdas_ncvx = []
            gammas = []
        
        x = torch.zeros(z.shape[0], self.n, requires_grad=False, device=self.device)

        if self.learn_kernel:
            H = convmtx_torch(self.h.weight.reshape(-1), x.shape[1], device=self.device).repeat(
                z.shape[0], 1, 1
            )
        
        Ht_x_degraded = torch.bmm(torch.transpose(
                        H, 1, 2), z.unsqueeze(dim=2))

        for i, l in enumerate(self.Layers):

            if mode == "learning_lambda_MM":

                if Disp_param == False:
                    x = self.Layers[i](
                        x,
                        z,
                        Ht_x_degraded,
                        H,
                        self.L,
                        self.delta_s_cvx,
                        self.delta_s_ncvx,
                        self.number_penalization_cvx,
                        self.number_penalization_ncvx,
                        Disp_param,
                    )
                if Disp_param == True:
                    x, lambda_cvx, lambda_ncvx, gamma = self.Layers[i](
                        x,
                        z,
                        Ht_x_degraded,
                        H,
                        self.L,
                        self.delta_s_cvx,
                        self.delta_s_ncvx,
                        self.number_penalization_cvx,
                        self.number_penalization_ncvx,
                        Disp_param,
                    )

                    lambdas_cvx.append(lambda_cvx)
                    lambdas_ncvx.append(lambda_ncvx)
                    gammas.append(gamma)

            if mode == "Deep_equilibrium":
                lamda_cvx, lamda_ncvx = self.architecture(H, x, z)
                x = self.Layers[i](
                    x,
                    z,
                    Ht_x_degraded,
                    H,
                    self.L,
                    self.delta_s_cvx,
                    self.delta_s_ncvx,
                    self.number_penalization_cvx,
                    self.number_penalization_ncvx,
                    Disp_param,
                    lamda_cvx,
                    lamda_ncvx,
                )

        if Disp_param == False:
            return r(x)
        if Disp_param == True:
            return r(x), lambdas_cvx, lambdas_ncvx, gammas
