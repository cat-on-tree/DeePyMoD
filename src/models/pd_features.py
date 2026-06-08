import numpy as np

from src.configs.pd_atom_registry import EXPANDED_PD_LIBRARY_TERMS


def build_pd_features(R, C, ec50, gamma):
    """
    与 PDLibraryExpanded 对齐的特征构造：
    1, R, C, C^2, Emax(C), Hill(C), C*R, Emax(C)*R, Hill(C)*R, R^2
    """
    C = max(float(C), 1e-10)
    ec50 = max(float(ec50), 1e-10)
    gamma = max(float(gamma), 1e-10)

    emax = C / (ec50 + C)
    hill = (C ** gamma) / (ec50 ** gamma + C ** gamma)

    return {
        "1": 1.0,
        "R": float(R),
        "C": float(C),
        "C^2": float(C ** 2),
        "Emax(C)": float(emax),
        "Hill(C)": float(hill),
        "C*R": float(C * R),
        "Emax(C)*R": float(emax * R),
        "Hill(C)*R": float(hill * R),
        "R^2": float(R * R),
    }


def get_pd_term_names():
    return [term for term in EXPANDED_PD_LIBRARY_TERMS if term not in {"exp(-t)", "exp(-t)*R"}]
