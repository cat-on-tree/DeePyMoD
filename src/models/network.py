import torch
import torch.nn as nn


class TimeOnlyNet(nn.Module):
    """
    输入 x: [t_norm, C]
    仅用 t_norm 拟合 R(t)，再由 library 使用 x 中的 C 构造候选项
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        t = x[:, 0:1]
        pred = self.net(t)
        return pred, x