import torch
import torch.nn as nn

from src.configs.pd_module_registry import STATE_SPECS


def build_state_names(module_combo: str, state_specs=None):
    state_specs = state_specs or STATE_SPECS
    modules = module_combo.split("+") if module_combo else []
    latent = []
    for mod in modules:
        spec = state_specs.get(mod, {})
        latent.extend(spec.get("latent", []))
    names = ["R"]
    for s in latent:
        if s != "R" and s not in names:
            names.append(s)
    return names


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


class ModularTimeNet(nn.Module):
    """
    Module-aware time-only network that outputs multiple PD states.

    - Uses time as the only input (keeps consistency with prior design)
    - Output dimension is determined by module_combo latent state specs
    """
    def __init__(self, module_combo: str, hidden=64, state_specs=None):
        super().__init__()
        self.module_combo = module_combo
        self.state_names = build_state_names(module_combo, state_specs=state_specs)
        out_dim = len(self.state_names)

        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        t = x[:, 0:1]
        pred = self.net(t)
        return pred, x


class StateTimeNet(nn.Module):
    """
    Time-only network with explicit state names (for specialized libraries).
    """
    def __init__(self, state_names, hidden=64):
        super().__init__()
        self.state_names = list(state_names)
        out_dim = len(self.state_names)

        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        t = x[:, 0:1]
        pred = self.net(t)
        return pred, x
