import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from deepymod import Library

from src.configs.pd_module_registry import MODULE_TERMS, STATE_SPECS


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


class ModularPDLibrary(Library):
    """
    Module-aware PD library that builds a compact candidate set based on module combinations.

    Notes:
    - This library focuses on the primary PD state (R) for sparse discovery.
    - Latent states (e.g., T1/T2/T3, Tol, Ce) can be used in candidate terms
      but are not constrained by their own ODEs in this version.
    """
    def __init__(self, module_combo: str, module_terms=None, state_specs=None):
        super().__init__()
        self.module_combo = module_combo
        self.module_terms = module_terms or MODULE_TERMS
        self.state_specs = state_specs or STATE_SPECS
        self.raw_ec50 = nn.Parameter(torch.tensor(4.0))
        self.raw_gamma = nn.Parameter(torch.tensor(2.0))

        self.state_names = self._build_state_names(module_combo)
        self._term_names = self._build_term_names(module_combo)

    def _build_state_names(self, module_combo: str):
        modules = module_combo.split("+")
        latent = []
        for mod in modules:
            spec = self.state_specs.get(mod, {})
            latent.extend(spec.get("latent", []))
        # Always include R, then unique latent states
        names = ["R"]
        for s in latent:
            if s != "R" and s not in names:
                names.append(s)
        return names

    def _build_term_names(self, module_combo: str):
        modules = module_combo.split("+")
        terms = []
        for mod in modules:
            mod_terms = self.module_terms.get(mod, {})
            for t in mod_terms.get("R", []):
                if t not in terms:
                    terms.append(t)
        return terms

    def term_names(self):
        return list(self._term_names)

    def _eval_base_term(self, term, ctx, ec50, gamma):
        if term == "1":
            return torch.ones_like(ctx["R"])
        if term in ctx:
            return ctx[term]
        if term == "C^2":
            return ctx["C"] ** 2
        if term.startswith("Hill(") and term.endswith(")"):
            var = term[5:-1]
            x = ctx.get(var)
            if x is None:
                return torch.zeros_like(ctx["R"])
            return (x ** gamma) / (ec50 ** gamma + x ** gamma)
        if term.startswith("Emax(") and term.endswith(")"):
            var = term[5:-1]
            x = ctx.get(var)
            if x is None:
                return torch.zeros_like(ctx["R"])
            return x / (ec50 + x)
        if term == "cos(2pi*t/24)":
            return torch.cos(2 * np.pi * ctx["t"] / 24.0)
        if term == "sin(2pi*t/24)":
            return torch.sin(2 * np.pi * ctx["t"] / 24.0)
        raise ValueError(f"Unknown term: {term}")

    def _eval_term(self, term, ctx, ec50, gamma):
        if term.endswith("*R"):
            base = term[:-2]
            return self._eval_base_term(base, ctx, ec50, gamma) * ctx["R"]
        return self._eval_base_term(term, ctx, ec50, gamma)

    def library(self, input):
        pred, data = input
        device = pred.device

        # Map state outputs
        state_map = {name: pred[:, i:i+1] for i, name in enumerate(self.state_names)}

        # Inputs
        t = data[:, 0:1]
        C = data[:, 1:2].clamp_min(1e-8) if data.shape[1] >= 2 else torch.zeros_like(t)
        C_int = data[:, 2:3].clamp_min(1e-8) if data.shape[1] >= 3 else torch.zeros_like(t)

        ctx = {
            "t": t,
            "R": state_map.get("R"),
            "C": C,
            "C_int": C_int,
            **{k: v for k, v in state_map.items() if k != "R"}
        }

        # dR/dt
        R = ctx["R"]
        dR_ddata = grad(
            R, data,
            grad_outputs=torch.ones_like(R),
            create_graph=True,
            retain_graph=True
        )[0]
        dRdt = dR_ddata[:, 0:1]

        ec50 = F.softplus(self.raw_ec50) + 1e-8
        gamma = F.softplus(self.raw_gamma) + 1e-8

        theta_cols = [self._eval_term(term, ctx, ec50, gamma) for term in self._term_names]
        theta = torch.cat(theta_cols, dim=1)

        return [dRdt], [theta]
