import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from deepymod import Library


class PDLibraryExpanded(Library):
    """
    theta columns:
      0:1, 1:R, 2:C, 3:C^2, 4:Emax(C), 5:Hill(C), 6:C*R, 7:Emax(C)*R, 8:Hill(C)*R
    """
    def __init__(self):
        super().__init__()
        self.raw_ec50 = nn.Parameter(torch.tensor(4.0))
        self.raw_gamma = nn.Parameter(torch.tensor(2.0))

    def library(self, input):
        pred, data = input
        R = pred[:, 0:1]

        dR_ddata = grad(
            R, data,
            grad_outputs=torch.ones_like(R),
            create_graph=True,
            retain_graph=True
        )[0]
        dRdt = dR_ddata[:, 0:1]

        C = data[:, 1:2].clamp_min(1e-8)
        ec50 = F.softplus(self.raw_ec50) + 1e-8
        gamma = F.softplus(self.raw_gamma) + 1e-8

        emax = C / (ec50 + C)
        hill = (C ** gamma) / (ec50 ** gamma + C ** gamma)

        one = torch.ones_like(R)
        theta = torch.cat([one, R, C, C * C, emax, hill, C * R, emax * R, hill * R], dim=1)

        return [dRdt], [theta]

    @staticmethod
    def term_names():
        return ["1", "R", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"]