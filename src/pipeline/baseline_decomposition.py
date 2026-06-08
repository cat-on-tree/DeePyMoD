import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


BASELINE_TERMS: Tuple[str, ...] = ("1", "R", "R^2", "cos24", "sin24")
DEFAULT_BASELINE_TERMS: Tuple[str, ...] = ("1", "cos24", "sin24")


def _eval_terms(terms: List[str], t: np.ndarray, r: np.ndarray) -> np.ndarray:
    cols = []
    for name in terms:
        if name == "1":
            cols.append(np.ones_like(t, dtype=float))
        elif name == "R":
            cols.append(r.astype(float))
        elif name == "R^2":
            cols.append((r.astype(float) ** 2))
        elif name == "cos24":
            cols.append(np.cos(2.0 * np.pi * t / 24.0))
        elif name == "sin24":
            cols.append(np.sin(2.0 * np.pi * t / 24.0))
    if len(cols) == 0:
        return np.zeros((len(t), 0), dtype=float)
    return np.column_stack(cols).astype(float)


def _fit_subset_bic(t: np.ndarray, r: np.ndarray, drdt: np.ndarray, terms: List[str]) -> Dict:
    x = _eval_terms(terms, t, r)
    n = int(len(drdt))
    if x.shape[1] == 0 or n <= max(4, x.shape[1] + 1):
        return {"ok": False}
    try:
        coef, _, _, _ = np.linalg.lstsq(x, drdt, rcond=None)
    except Exception:
        return {"ok": False}
    yhat = x @ coef
    resid = drdt - yhat
    rss = float(np.sum(resid**2))
    rss = max(rss, 1e-12)
    k = int(x.shape[1])
    bic = float(n * np.log(rss / n) + k * np.log(max(n, 2)))
    rmse = float(np.sqrt(rss / n))
    return {
        "ok": True,
        "terms": list(terms),
        "coef": [float(v) for v in coef],
        "bic": bic,
        "rmse": rmse,
        "n_points": n,
    }


def fit_zero_dose_baseline(
    df_long: pd.DataFrame,
    c_zero_eps: float = 1e-8,
    min_points: int = 8,
    candidate_terms: Tuple[str, ...] = DEFAULT_BASELINE_TERMS,
) -> Dict:
    required = {"sid", "time", "C_obs", "R_obs"}
    if not required.issubset(df_long.columns):
        return {"enabled": False, "reason": "missing_required_columns"}

    d = df_long[["sid", "time", "C_obs", "R_obs"]].copy()
    d = d.sort_values(["sid", "time"]).reset_index(drop=True)
    sid_cmax = d.groupby("sid")["C_obs"].max()
    zero_sids = sid_cmax[sid_cmax <= float(c_zero_eps)].index.tolist()
    if len(zero_sids) == 0:
        return {"enabled": False, "reason": "no_zero_dose_subject"}

    z = d[d["sid"].isin(zero_sids)].copy()
    agg = z.groupby("time", as_index=False)["R_obs"].mean().sort_values("time")
    t = agg["time"].values.astype(float)
    r = agg["R_obs"].values.astype(float)
    if len(t) < max(min_points, 5):
        return {"enabled": False, "reason": "insufficient_zero_dose_points", "n_zero_points": int(len(t))}
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(r)):
        return {"enabled": False, "reason": "nonfinite_zero_dose_series"}
    if np.min(np.diff(np.unique(t))) <= 0:
        t_u, idx = np.unique(t, return_index=True)
        t = t_u
        r = r[idx]
    if len(t) < max(min_points, 5):
        return {"enabled": False, "reason": "insufficient_unique_time_points", "n_zero_points": int(len(t))}

    terms_universe = tuple(str(x) for x in candidate_terms if str(x) in BASELINE_TERMS)
    if len(terms_universe) == 0:
        return {"enabled": False, "reason": "no_valid_baseline_terms"}

    drdt = np.gradient(r, t).astype(float)
    results = []
    for k in range(1, len(terms_universe) + 1):
        for sub in itertools.combinations(terms_universe, k):
            rr = _fit_subset_bic(t, r, drdt, list(sub))
            if rr.get("ok", False):
                results.append(rr)
    if not results:
        return {"enabled": False, "reason": "baseline_fit_failed"}

    best = sorted(results, key=lambda x: (x["bic"], x["rmse"]))[0]
    return {
        "enabled": True,
        "reason": "ok",
        "terms": best["terms"],
        "coef": best["coef"],
        "bic": float(best["bic"]),
        "rmse": float(best["rmse"]),
        "n_points": int(best["n_points"]),
        "n_zero_subjects": int(len(zero_sids)),
        "n_subjects": int(d["sid"].nunique()),
        "c_zero_eps": float(c_zero_eps),
        "candidate_terms": list(terms_universe),
    }


def _rhs_from_model(t: float, r: float, model: Dict) -> float:
    terms = model.get("terms", [])
    coef = model.get("coef", [])
    if len(terms) != len(coef):
        return 0.0
    val = 0.0
    for name, b in zip(terms, coef):
        if name == "1":
            feat = 1.0
        elif name == "R":
            feat = float(r)
        elif name == "R^2":
            feat = float(r * r)
        elif name == "cos24":
            feat = float(np.cos(2.0 * np.pi * t / 24.0))
        elif name == "sin24":
            feat = float(np.sin(2.0 * np.pi * t / 24.0))
        else:
            feat = 0.0
        val += float(b) * feat
    return float(val)


def _simulate_baseline(model: Dict, t_grid: np.ndarray, r0: float) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    if t.size < 2:
        return np.full_like(t, float(r0), dtype=float)
    ord_idx = np.argsort(t)
    t_sorted = t[ord_idx]
    try:
        sol = solve_ivp(
            fun=lambda tt, rr: [_rhs_from_model(tt, float(rr[0]), model)],
            t_span=(float(t_sorted[0]), float(t_sorted[-1])),
            y0=[float(r0)],
            t_eval=t_sorted,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if (not sol.success) or (sol.y.shape[1] != t_sorted.size):
            y = np.full_like(t_sorted, float(r0), dtype=float)
        else:
            y = sol.y[0].astype(float)
    except Exception:
        y = np.full_like(t_sorted, float(r0), dtype=float)

    out = np.zeros_like(y, dtype=float)
    out[:] = y
    inv = np.empty_like(ord_idx)
    inv[ord_idx] = np.arange(ord_idx.size)
    return out[inv]


def apply_zero_dose_baseline_correction(df_long: pd.DataFrame, model: Dict) -> pd.DataFrame:
    d = df_long.copy()
    d["R_baseline_hat"] = d["R_obs"].astype(float)
    d["R_baseline_drift"] = 0.0
    d["R_obs_basecorr"] = d["R_obs"].astype(float)
    if not model.get("enabled", False):
        return d

    for sid, grp in d.groupby("sid"):
        g = grp.sort_values("time")
        t = g["time"].values.astype(float)
        y = g["R_obs"].values.astype(float)
        if len(t) == 0:
            continue
        r0 = float(y[0])
        b = _simulate_baseline(model, t, r0=r0)
        drift = b - r0
        d.loc[g.index, "R_baseline_hat"] = b
        d.loc[g.index, "R_baseline_drift"] = drift
    # Recompute correction from stored drift to guarantee effective application.
    d["R_obs_basecorr"] = d["R_obs"].astype(float) - d["R_baseline_drift"].astype(float)
    return d
