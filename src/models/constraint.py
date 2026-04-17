import torch
import torch.nn as nn


class RidgeConstraint(nn.Module):
    """
    对当前 mask 的 theta 做 ridge 闭式回归，得到稀疏系数向量
    """
    def __init__(self, lam=1e-4):
        super().__init__()
        self.lam = lam
        self.sparsity_masks = None
        self.coeff_vectors = None

    def forward(self, input):
        time_derivs, thetas = input

        if self.sparsity_masks is None:
            self.sparsity_masks = [
                torch.ones(theta.shape[1], dtype=torch.bool, device=theta.device)
                for theta in thetas
            ]

        coeffs = []
        for theta, dt, mask in zip(thetas, time_derivs, self.sparsity_masks):
            th = theta[:, mask]
            xtx = th.T @ th
            reg = self.lam * torch.eye(xtx.shape[0], device=xtx.device)
            xty = th.T @ dt

            w_small = torch.linalg.pinv(xtx + reg) @ xty

            full = torch.zeros((mask.shape[0], 1), device=theta.device)
            full[mask] = w_small
            coeffs.append(full)

        self.coeff_vectors = coeffs
        return coeffs