import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from deepymod import Library


class AntibodyTMDDLibrary(Library):
    """
    Minimal TMDD/antibody library for mechanistic discovery.

    Equations (per state):
    - R:    1, R, C, C*R, CpR, CpR*R, Hill(C) (optional)
    - CpR:  C*R, CpR
    - Ct:   C, Ct, C-Ct (only when use_ct=True)
    """
    def __init__(
        self,
        use_ct: bool = False,
        include_hill: bool = False,
        include_c_term: bool = False,
        include_c_r_term: bool = False,
        enforce_safe_r_terms: bool = True,
    ):
        super().__init__()
        self.use_ct = bool(use_ct)
        self.include_hill = bool(include_hill)
        self.include_c_term = bool(include_c_term)
        self.include_c_r_term = bool(include_c_r_term)
        self.enforce_safe_r_terms = bool(enforce_safe_r_terms)
        self.raw_ec50 = nn.Parameter(torch.tensor(4.0))
        self.raw_gamma = nn.Parameter(torch.tensor(2.0))

        self.state_names = ["R", "CpR"] + (["Ct"] if self.use_ct else [])
        self.equation_order = list(self.state_names)

        if self.enforce_safe_r_terms:
            r_terms = ["1", "R", "CpR"]
            if self.include_c_term:
                r_terms.append("C")
            if self.include_c_r_term:
                r_terms.append("C*R")
            r_terms.append("CpR*R")
        else:
            r_terms = ["1", "R", "C", "C*R", "CpR", "CpR*R"]
        if self.include_hill:
            r_terms.append("Hill(C)")

        self.equation_terms = {
            "R": r_terms,
            "CpR": ["C*R", "CpR"],
        }
        if self.use_ct:
            self.equation_terms["Ct"] = ["C", "Ct", "C-Ct"]

    def term_names(self):
        return [list(self.equation_terms[s]) for s in self.equation_order]

    def metadata(self):
        return {
            "equation_order": list(self.equation_order),
            "equation_terms": {k: list(v) for k, v in self.equation_terms.items()},
            "term_groups": {
                "r_required": ["1", "R", "CpR"],
                "r_optional": ["C", "C*R", "Hill(C)"],
                "tmdd_interactions": ["CpR*R", "C*R", "C-Ct"],
            },
            "flags": {
                "use_ct": self.use_ct,
                "include_hill": self.include_hill,
                "include_c_term": self.include_c_term,
                "include_c_r_term": self.include_c_r_term,
                "enforce_safe_r_terms": self.enforce_safe_r_terms,
            },
        }

    def get_metadata(self):
        return self.metadata()

    def _eval_term(self, term, ctx, ec50, gamma):
        if term == "1":
            return torch.ones_like(ctx["R"])
        if term == "R":
            return ctx["R"]
        if term == "C":
            return ctx["C"]
        if term == "Ct":
            return ctx["Ct"]
        if term == "CpR":
            return ctx["CpR"]
        if term == "C*R":
            return ctx["C"] * ctx["R"]
        if term == "CpR*R":
            return ctx["CpR"] * ctx["R"]
        if term == "C-Ct":
            return ctx["C"] - ctx["Ct"]
        if term == "Hill(C)":
            return (ctx["C"] ** gamma) / (ec50 ** gamma + ctx["C"] ** gamma)
        raise ValueError(f"Unknown term: {term}")

    def library(self, input):
        pred, data = input

        state_map = {name: pred[:, i:i+1] for i, name in enumerate(self.state_names)}
        t = data[:, 0:1]
        C = data[:, 1:2].clamp_min(1e-8) if data.shape[1] >= 2 else torch.zeros_like(t)

        ctx = {
            "t": t,
            "C": C,
            "R": state_map["R"],
            "CpR": state_map["CpR"],
        }
        if self.use_ct:
            ctx["Ct"] = state_map["Ct"]

        ec50 = F.softplus(self.raw_ec50) + 1e-8
        gamma = F.softplus(self.raw_gamma) + 1e-8

        dts, thetas = [], []
        for s in self.equation_order:
            X = state_map[s]
            dX_ddata = grad(
                X, data,
                grad_outputs=torch.ones_like(X),
                create_graph=True,
                retain_graph=True
            )[0]
            dXdt = dX_ddata[:, 0:1]

            terms = self.equation_terms[s]
            theta_cols = [self._eval_term(term, ctx, ec50, gamma) for term in terms]
            theta = torch.cat(theta_cols, dim=1)
            dts.append(dXdt)
            thetas.append(theta)

        return dts, thetas
