import torch
import torch.nn as nn


class FixedMaskEstimator(nn.Module):
    """
    固定mask估计器：由外部 pruning/ranking 逻辑更新 mask

    支持：
    - 单一方程 mask (torch.Tensor)
    - 多方程 mask (list[torch.Tensor])
    """
    def __init__(self, init_mask):
        super().__init__()
        self._init_masks(init_mask)

    def _init_masks(self, init_mask):
        if isinstance(init_mask, (list, tuple)):
            self.mask_bufs = nn.ParameterList([
                nn.Parameter(m.clone(), requires_grad=False) for m in init_mask
            ])
        else:
            self.register_buffer("mask_buf", init_mask.clone())
            self.mask_bufs = None

    def set_mask(self, new_mask):
        if isinstance(new_mask, (list, tuple)):
            self.mask_bufs = nn.ParameterList([
                nn.Parameter(m.clone(), requires_grad=False) for m in new_mask
            ])
            if hasattr(self, "mask_buf"):
                delattr(self, "mask_buf")
        else:
            if self.mask_bufs is not None:
                self.mask_bufs = None
            self.register_buffer("mask_buf", new_mask.clone())

    def forward(self, thetas, time_derivs):
        if self.mask_bufs is not None:
            return [m for m in self.mask_bufs]
        return [self.mask_buf]
