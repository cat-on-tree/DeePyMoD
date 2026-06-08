import numpy as np
import pandas as pd

from src.configs.pkpd_registry import MODEL_REGISTRY, tv_pk
from src.data.pk_profiles import simulate_pk_profile


def _simulate_known_pk(model_name, times):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}")
    return simulate_pk_profile(times, route="oral", compartments=1, pk_params=tv_pk)


def _fit_positive_scale(c_obs, c_model, min_scale=0.0, max_scale=5.0):
    obs = np.asarray(c_obs, dtype=float)
    mdl = np.asarray(c_model, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(mdl)
    if not mask.any():
        return 1.0
    # Keep placebo/zero-exposure subjects at zero instead of forcing a positive scale.
    if float(np.max(np.abs(obs[mask]))) <= 1e-8:
        return 0.0
    den = float(np.dot(mdl[mask], mdl[mask]))
    if den <= 1e-12:
        return 1.0
    raw = float(np.dot(obs[mask], mdl[mask]) / den)
    lo = max(0.0, float(min_scale))
    hi = max(lo, float(max_scale))
    return float(np.clip(raw, lo, hi))


def impute_pk_from_known_model(
    df: pd.DataFrame,
    model_name: str,
    pk_route: str = None,
    pk_compartments: int = None,
    fit_subject_scale: bool = True,
    min_scale: float = 0.0,
    max_scale: float = 5.0,
):
    required = {"sid", "time", "R_obs"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input data must contain columns {required}, got {list(df.columns)}")

    out = df.copy()
    out["sid"] = out["sid"].astype(int)
    out["time"] = out["time"].astype(float)
    out["R_obs"] = out["R_obs"].astype(float)
    has_c_obs = "C_obs" in out.columns
    if has_c_obs:
        out["C_obs_raw"] = pd.to_numeric(out["C_obs"], errors="coerce")

    rows = []
    for sid, grp in out.groupby("sid", sort=True):
        sg = grp.sort_values("time").copy()
        t = sg["time"].values.astype(float)
        use_explicit_pk = (pk_route is not None) or (pk_compartments is not None)
        if not use_explicit_pk:
            c_model = _simulate_known_pk(model_name, t)
        else:
            c_model = simulate_pk_profile(
                t,
                route=(pk_route or "oral"),
                compartments=int(pk_compartments or 1),
                pk_params=tv_pk,
            )

        if fit_subject_scale and has_c_obs:
            scale = _fit_positive_scale(
                c_obs=sg["C_obs_raw"].values.astype(float),
                c_model=c_model,
                min_scale=min_scale,
                max_scale=max_scale,
            )
        else:
            scale = 1.0

        sg["pk_scale"] = scale
        sg["C_obs_model"] = c_model
        sg["C_obs"] = np.clip(c_model * scale, 0.0, None)
        rows.append(sg)

    merged = pd.concat(rows, ignore_index=True)
    keep = ["sid", "time", "C_obs", "R_obs", "C_obs_model", "pk_scale"] + (["C_obs_raw"] if has_c_obs else [])
    passthrough = [c for c in merged.columns if c.endswith("_obs") and c not in {"C_obs", "R_obs"}]
    return merged[keep + passthrough]
