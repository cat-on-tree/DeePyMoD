import numpy as np
import torch

from src.configs.pkpd_registry import MODEL_REGISTRY, tv_pk, omega_pk, t_dense, t_obs
from src.data.subject_simulator import sample_individual_params, simulate_subject


def _pd_shape_metrics(r_obs):
    r = np.asarray(r_obs, dtype=float).reshape(-1)
    if r.size < 4 or (not np.isfinite(r).all()):
        return {"rel_span": 0.0, "curvature": 0.0}
    q95 = float(np.percentile(r, 95))
    q05 = float(np.percentile(r, 5))
    mid = float(np.median(np.abs(r))) + 1e-8
    rel_span = (q95 - q05) / mid
    d1 = np.diff(r)
    d2 = np.diff(r, n=2)
    curv = float(np.std(d2) / (np.std(d1) + 1e-8)) if d1.size > 0 and d2.size > 0 else 0.0
    return {"rel_span": float(rel_span), "curvature": curv}


def _observable_identifiability_metrics(c_obs, r_obs):
    c = np.asarray(c_obs, dtype=float).reshape(-1)
    r = np.asarray(r_obs, dtype=float).reshape(-1)
    if c.size < 4 or r.size < 4 or (not np.isfinite(c).all()) or (not np.isfinite(r).all()):
        return {"c_rel_span": 0.0, "r_rel_span": 0.0, "abs_corr": 0.0, "slope_var": 0.0}

    c_q95 = float(np.percentile(c, 95))
    c_q05 = float(np.percentile(c, 5))
    c_mid = float(np.median(np.abs(c))) + 1e-8
    c_rel_span = (c_q95 - c_q05) / c_mid

    r_q95 = float(np.percentile(r, 95))
    r_q05 = float(np.percentile(r, 5))
    r_mid = float(np.median(np.abs(r))) + 1e-8
    r_rel_span = (r_q95 - r_q05) / r_mid

    c_std = float(np.std(c))
    r_std = float(np.std(r))
    if c_std < 1e-10 or r_std < 1e-10:
        abs_corr = 0.0
    else:
        abs_corr = float(np.abs(np.corrcoef(c, r)[0, 1]))
        if not np.isfinite(abs_corr):
            abs_corr = 0.0

    dr = np.diff(r)
    slope_var = float(np.std(dr) / (np.mean(np.abs(dr)) + 1e-8)) if dr.size > 0 else 0.0
    return {
        "c_rel_span": float(c_rel_span),
        "r_rel_span": float(r_rel_span),
        "abs_corr": float(abs_corr),
        "slope_var": float(slope_var),
    }


def _is_pd_shape_reasonable(family, r_obs):
    m = _pd_shape_metrics(r_obs)
    rel_span = m["rel_span"]
    curv = m["curvature"]

    # Delay-chain / latent-mechanism families should not look nearly linear-flat.
    if family in {"transduction", "feedback", "biophase", "tolerance", "precursor"}:
        return (rel_span >= 0.08) and (curv >= 0.12)
    # Others keep a mild minimum dynamic requirement.
    return rel_span >= 0.03


def _is_observable_family_identifiable(family, c_obs, r_obs):
    m = _observable_identifiability_metrics(c_obs, r_obs)
    if family == "direct":
        return (m["c_rel_span"] >= 0.08) and (m["r_rel_span"] >= 0.06) and (m["abs_corr"] >= 0.30)
    if family == "idr":
        return (m["c_rel_span"] >= 0.08) and (m["r_rel_span"] >= 0.08) and (m["abs_corr"] >= 0.18) and (m["slope_var"] >= 0.08)
    if family == "tgi":
        return (m["c_rel_span"] >= 0.08) and (m["r_rel_span"] >= 0.08) and (m["abs_corr"] >= 0.16) and (m["slope_var"] >= 0.08)
    return True


def _clamp_observable_params(p, family):
    q = dict(p)

    def _clip(name, lo=None, hi=None):
        if name not in q:
            return
        val = float(q[name])
        if lo is not None:
            val = max(float(lo), val)
        if hi is not None:
            val = min(float(hi), val)
        q[name] = float(val)

    if family == "direct":
        _clip("S", 0.05, 1.2)
        _clip("Emax", 0.4, 16.0)
        _clip("EC50", 0.35, 10.0)
        _clip("gamma", 0.9, 1.6)
        _clip("k_resp", 1.0, 6.0)
    elif family == "idr":
        _clip("Imax", 0.2, 0.9)
        _clip("Smax", 0.2, 0.9)
        _clip("EC50", 0.35, 10.0)
        _clip("gamma", 0.9, 2.2)
    elif family == "tgi":
        _clip("k_grow", 0.08, 0.28)
        _clip("Emax_kill", 0.2, 1.4)
        _clip("EC50_kill", 0.3, 8.0)
        _clip("gamma_kill", 0.9, 2.2)
        _clip("lambda_kill", 0.005, 0.02)
    return q


def _is_observable_shape_bounded(family, r_obs):
    r = np.asarray(r_obs, dtype=float).reshape(-1)
    if r.size < 4 or (not np.isfinite(r).all()):
        return False
    baseline = float(np.percentile(r[: min(3, r.size)], 50))
    scale = max(abs(baseline), 1e-6)
    max_ratio = float(np.max(r) / scale)
    min_ratio = float(np.min(r) / scale)

    if family == "direct":
        return (max_ratio <= 1.38) and (min_ratio >= 0.72)
    if family == "idr":
        return (max_ratio <= 1.20) and (min_ratio >= 0.10)
    if family == "tgi":
        return (max_ratio <= 25.0) and (min_ratio >= 0.01)
    return True


def _boost_pd_params_for_family(p, family, rng):
    q = dict(p)

    def _mul(name, lo, hi, min_v=1e-8, max_v=None):
        if name not in q:
            return
        q[name] = float(max(min_v, q[name] * float(rng.uniform(lo, hi))))
        if max_v is not None:
            q[name] = float(min(max_v, q[name]))

    if family == "transduction":
        _mul("kt1", 0.9, 2.0, min_v=0.15, max_v=2.5)
        _mul("Emax1", 1.3, 2.8, min_v=0.2, max_v=4.0)
        _mul("EC50", 0.45, 0.90, min_v=0.2, max_v=12.0)
        _mul("gamma", 1.1, 1.8, min_v=0.8, max_v=7.0)
    elif family == "feedback":
        _mul("kt2", 0.9, 1.8, min_v=0.12, max_v=2.0)
        _mul("Emax1", 1.2, 2.0, min_v=0.2, max_v=4.0)
        _mul("Emax2", 1.2, 2.4, min_v=0.15, max_v=3.5)
        _mul("EC50", 0.55, 0.95, min_v=0.2, max_v=12.0)
        _mul("gamma", 1.0, 1.6, min_v=0.8, max_v=7.0)
    elif family == "biophase":
        _mul("ke0", 1.1, 2.2, min_v=0.08, max_v=3.0)
        _mul("Emax", 1.1, 1.8, min_v=0.5, max_v=60.0)
        _mul("EC50", 0.6, 1.0, min_v=0.2, max_v=12.0)
    elif family == "tolerance":
        _mul("k_in_tol", 1.1, 2.0, min_v=0.05, max_v=2.5)
        _mul("Smax_tol", 1.2, 2.2, min_v=0.2, max_v=4.0)
        _mul("Emax1", 1.1, 1.8, min_v=0.2, max_v=4.0)
    elif family == "precursor":
        _mul("ktr", 1.0, 2.0, min_v=0.08, max_v=2.5)
        _mul("Emax1", 1.2, 2.0, min_v=0.2, max_v=4.0)
        _mul("EC50", 0.55, 0.95, min_v=0.2, max_v=12.0)
    elif family == "direct":
        _mul("S", 1.0, 1.10, min_v=0.05, max_v=1.2)
        _mul("Emax", 0.95, 1.08, min_v=0.4, max_v=16.0)
        _mul("EC50", 0.90, 1.10, min_v=0.35, max_v=10.0)
        _mul("gamma", 0.92, 1.03, min_v=0.9, max_v=1.6)
        _mul("k_resp", 0.92, 1.03, min_v=1.0, max_v=6.0)
    elif family == "idr":
        _mul("Imax", 1.0, 1.12, min_v=0.2, max_v=0.9)
        _mul("Smax", 1.0, 1.12, min_v=0.2, max_v=0.9)
        _mul("EC50", 0.90, 1.10, min_v=0.35, max_v=10.0)
        _mul("gamma", 0.95, 1.08, min_v=0.9, max_v=2.2)
    elif family == "tgi":
        _mul("k_grow", 0.95, 1.10, min_v=0.08, max_v=0.28)
        _mul("Emax_kill", 1.0, 1.15, min_v=0.2, max_v=1.4)
        _mul("EC50_kill", 0.90, 1.10, min_v=0.3, max_v=8.0)
        _mul("gamma_kill", 0.95, 1.08, min_v=0.9, max_v=2.2)
        _mul("lambda_kill", 0.95, 1.08, min_v=0.005, max_v=0.02)
    else:
        _mul("Emax1", 1.1, 1.6, min_v=0.1, max_v=4.0)
        _mul("Emax", 1.1, 1.6, min_v=0.1, max_v=60.0)

    return q


def _simulate_subject_with_quality_guard(
    cfg,
    p,
    rng,
    pk_route=None,
    pk_compartments=None,
    max_tries=4,
    observation_times=None,
    noise_scale=1.0,
):
    family = str(cfg.get("family", ""))
    observable_family = family in {"direct", "idr", "tgi"}
    best = None
    best_score = -np.inf
    cur_p = dict(p)
    if observable_family:
        cur_p = _clamp_observable_params(cur_p, family)

    if observable_family:
        max_tries = min(int(max_tries), 2)

    for i in range(max_tries + 1):
        c_obs, r_obs = simulate_subject(
            cfg,
            cur_p,
            pk_route=pk_route,
            pk_compartments=pk_compartments,
            dose_scale=float(cur_p.get("_dose_scale", 1.0)),
            observation_times=observation_times,
            noise_scale=noise_scale,
        )
        m = _pd_shape_metrics(r_obs)
        idm = _observable_identifiability_metrics(c_obs, r_obs)
        score = float(
            m["rel_span"]
            + 0.5 * m["curvature"]
            + 0.4 * idm["r_rel_span"]
            + 0.3 * idm["abs_corr"]
            + 0.2 * idm["slope_var"]
        )
        if score > best_score:
            best = (dict(cur_p), c_obs, r_obs)
            best_score = score
        pd_ok = _is_pd_shape_reasonable(family, r_obs)
        identifiable = _is_observable_family_identifiable(family, c_obs, r_obs)
        bounded = (True if not observable_family else _is_observable_shape_bounded(family, r_obs))
        if pd_ok and identifiable and bounded:
            return cur_p, c_obs, r_obs
        if i < max_tries:
            cur_p = _boost_pd_params_for_family(cur_p, family, rng)
            if observable_family:
                cur_p = _clamp_observable_params(cur_p, family)

    # Fallback to the most dynamic candidate if none satisfies thresholds.
    # For direct family, enforce a bounded peak by attenuating the dominant
    # peak-driving terms before returning.
    if observable_family and family == "direct" and best is not None:
        best_p, best_c, best_r = best
        if not _is_observable_shape_bounded(family, best_r):
            tune_p = dict(best_p)
            for _ in range(8):
                tune_p["Emax"] = max(0.4, float(tune_p.get("Emax", 0.4)) * 0.88)
                tune_p["gamma"] = max(0.9, float(tune_p.get("gamma", 0.9)) * 0.93)
                tune_p["k_resp"] = max(1.0, float(tune_p.get("k_resp", 1.0)) * 0.92)
                tune_p = _clamp_observable_params(tune_p, family)
                c_obs, r_obs = simulate_subject(
                    cfg,
                    tune_p,
                    pk_route=pk_route,
                    pk_compartments=pk_compartments,
                    dose_scale=float(tune_p.get("_dose_scale", 1.0)),
                    observation_times=observation_times,
                    noise_scale=noise_scale,
                )
                if _is_observable_shape_bounded(family, r_obs):
                    return tune_p, c_obs, r_obs
            return tune_p, c_obs, r_obs
    return best


def generate_population_data(
    model_name,
    seed=42,
    n_subjects=12,
    extra_pk_iiv_sigma=0.0,
    return_pk_scale=False,
    pk_route=None,
    pk_compartments=None,
    dose_design_enabled=False,
    dose_levels=None,
    observable_enhanced_design=False,
    observable_dose_levels=None,
    observation_times=None,
    disable_iiv=False,
    disable_quality_guard=False,
):
    """
    extra_pk_iiv_sigma:
      0.0 表示不额外加
      >0 表示对每个受试者 C_obs 乘以 exp(N(0, sigma))
    """
    cfg = MODEL_REGISTRY[model_name]
    tv_pd = cfg["tv_pd"]
    omega_pd = cfg["omega_pd"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed + 10086)
    use_dose_design = bool(dose_design_enabled) or bool(observable_enhanced_design)
    merged_levels = dose_levels if dose_levels is not None else observable_dose_levels
    merged_levels = list(merged_levels or [0.0, 0.2, 1.0, 5.0])
    if not merged_levels:
        merged_levels = [1.0]
    merged_levels = [max(0.0, float(x)) for x in merged_levels]
    n_dose = len(merged_levels)
    obs_times = np.asarray(t_obs if observation_times is None else observation_times, dtype=float).reshape(-1)
    obs_times = np.unique(np.clip(obs_times, float(np.min(t_dense)), float(np.max(t_dense))))
    if obs_times.size == 0:
        obs_times = np.asarray(t_obs, dtype=float)
    omega_pk_eff = ({k: 0.0 for k in omega_pk.keys()} if disable_iiv else omega_pk)
    omega_pd_eff = ({k: 0.0 for k in omega_pd.keys()} if disable_iiv else omega_pd)

    all_rows = []
    subject_params = []
    pk_scale_by_sid = {}

    for sid in range(n_subjects):
        p0 = sample_individual_params(tv_pk, omega_pk_eff, tv_pd, omega_pd_eff)
        p0["model_name"] = model_name
        p0["family"] = cfg["family"]
        if use_dose_design:
            p0["_dose_scale"] = merged_levels[int(sid) % n_dose]
        else:
            p0["_dose_scale"] = 1.0

        if bool(disable_quality_guard):
            p = dict(p0)
            C_obs, R_obs = simulate_subject(
                cfg,
                p,
                pk_route=pk_route,
                pk_compartments=pk_compartments,
                dose_scale=float(p.get("_dose_scale", 1.0)),
                observation_times=obs_times,
                noise_scale=(0.35 if disable_iiv else 1.0),
            )
        else:
            p, C_obs, R_obs = _simulate_subject_with_quality_guard(
                cfg=cfg,
                p=p0,
                rng=rng,
                pk_route=pk_route,
                pk_compartments=pk_compartments,
                max_tries=(0 if (use_dose_design and float(p0["_dose_scale"]) == 0.0) else 4),
                observation_times=obs_times,
                noise_scale=(0.35 if disable_iiv else 1.0),
            )
        subject_params.append(p)

        if extra_pk_iiv_sigma > 0:
            scale = float(np.exp(rng.normal(0.0, extra_pk_iiv_sigma)))
            C_obs = np.clip(C_obs * scale, 0.0, None)
        else:
            scale = 1.0
        pk_scale_by_sid[int(sid)] = scale

        for j, tt in enumerate(obs_times):
            all_rows.append([sid, tt, C_obs[j], R_obs[j]])

    pop_data = np.array(all_rows, dtype=float)

    if return_pk_scale:
        return pop_data, subject_params, cfg, pk_scale_by_sid
    return pop_data, subject_params, cfg

def self_check_models(seed=42, n_subjects=2, verbose=True):
    """
    遍历 MODEL_REGISTRY，尝试生成少量数据。
    """
    from src.configs.pkpd_registry import MODEL_REGISTRY
    ok, failed = [], {}
    for name in MODEL_REGISTRY.keys():
        try:
            _ = generate_population_data(
                model_name=name,
                seed=seed,
                n_subjects=n_subjects,
                extra_pk_iiv_sigma=0.0,
                return_pk_scale=False,
            )
            ok.append(name)
            if verbose:
                print(f"[OK] {name}")
        except Exception as e:
            failed[name] = str(e)
            if verbose:
                print(f"[FAIL] {name} -> {e}")

    return {"ok": ok, "failed": failed}
