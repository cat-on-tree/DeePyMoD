import numpy as np


def has_redundant_pair(terms, redundancy_groups):
    """
    redundancy_groups: list[list[str]]
    若某组中命中 >=2 个term，视为冗余共线风险
    """
    s = set(terms)
    for g in redundancy_groups:
        hit = sum([1 for t in g if t in s])
        if hit >= 2:
            return True
    return False


def compute_selection_score(
    bic_val: float,
    k: int,
    bic_std: float = np.nan,
    terms=None,
    lambda_k: float = 1.0,
    lambda_collinear: float = 1.0,
    lambda_stability: float = 0.2,
    redundancy_groups=None,
):
    """
    温和通用评分：
      score = BIC + lambda_k*k + collinear_penalty + stability_penalty
    分数越小越好
    """
    if terms is None:
        terms = []
    if redundancy_groups is None:
        redundancy_groups = []

    if bic_val is None or not np.isfinite(bic_val):
        return np.inf, {"k_penalty": np.inf, "collinear_penalty": np.inf, "stability_penalty": np.inf}

    k_penalty = float(lambda_k * k)

    collinear_penalty = 0.0
    if has_redundant_pair(terms, redundancy_groups):
        collinear_penalty = float(lambda_collinear)

    stability_penalty = 0.0
    if bic_std is not None and np.isfinite(bic_std):
        stability_penalty = float(lambda_stability * bic_std)

    score = float(bic_val + k_penalty + collinear_penalty + stability_penalty)
    detail = {
        "k_penalty": k_penalty,
        "collinear_penalty": collinear_penalty,
        "stability_penalty": stability_penalty,
    }
    return score, detail