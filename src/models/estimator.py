import torch
import torch.nn as nn


class FixedMaskEstimator(nn.Module):
    """
    固定mask估计器：由外部 pruning/ranking 逻辑更新 mask
    """
    def __init__(self, init_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask_buf", init_mask.clone())

    def set_mask(self, new_mask: torch.Tensor):
        self.mask_buf = new_mask.clone()

    def forward(self, thetas, time_derivs):
        return [self.mask_buf]