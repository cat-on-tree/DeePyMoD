import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.data.preprocess_pkpd import to_population_mean
from src.models.pd_features import build_pd_features, get_pd_term_names
from src.pipeline.model_selection import compute_selection_score


def _simulate_on_grid(theta, terms, t_grid, c_grid, r0, ec50, gamma):
    t_grid = np.asarray(t_grid, dtype=float)
    c_grid = np.asarray(c_grid, dtype=float)

    def c_of_t(tt):
        return np.interp(tt, t_grid, c_grid)

    def rhs(tt, rr):
        R = rr[0]
        C = c_of_t(tt)
        feats = build_pd_features(R, C, ec50, gamma)
        dR = 0.0
        for i, term in enumerate(terms):
            dR += theta[i] * feats[term]
        return [dR]

    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), [float(r0)], t_eval=t_grid,
        method="RK45", rtol=1e-5, atol=1e-7
    )
    if (not sol.success) or (sol.y.shape[1] != len(t_grid)):
        return np.full_like(t_grid, np.nan, dtype=float)
    return sol.y[0]


def _build_agg_splits(df_long, train_frac=0.7, keep_time_order=True, seed=42):
    pop_data = df_long[["sid", "time", "C_obs", "R_obs"]].values
    agg = to_population_mean(pop_data).sort_values("time").reset_index(drop=True)

    t = agg["time"].values.astype(float)
    c = agg["C_mean"].values.astype(float)
    r = agg["R_mean"].values.astype(float)

    n = len(t)
    n_train = int(np.floor(train_frac * n))
    if n_train <= 1 or n_train >= n:
        raise ValueError(f"Invalid split: n={n}, n_train={n_train}, train_frac={train_frac}")

    if keep_time_order:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

    tr_idx = idx[:n_train]
    va_idx = idx[n_train:]

    r_mean = float(np.mean(r[tr_idx]))
    r_std = float(np.std(r[tr_idx]) + 1e-8)

    sid_u = np.sort(df_long["sid"].astype(int).unique())
    n_sid = len(sid_u)
    n_sid_train = int(np.floor(train_frac * n_sid))
    if n_sid_train <= 0:
        n_sid_train = 1
    if n_sid_train >= n_sid:
        n_sid_train = n_sid - 1

    if keep_time_order:
        sid_idx = np.arange(n_sid)
    else:
        rng = np.random.default_rng(seed + 17)
        sid_idx = rng.permutation(n_sid)
    sid_tr = sid_u[sid_idx[:n_sid_train]]
    sid_va = sid_u[sid_idx[n_sid_train:]]

    return dict(
        t=t,
        c=c,
        r=r,
        tr_idx=tr_idx,
        va_idx=va_idx,
        r_mean=r_mean,
        r_std=r_std,
        sid_tr=sid_tr,
        sid_va=sid_va,
    )


def _build_subject_cache(df_long: pd.DataFrame):
    cache = {}
    for sid, df_sid in df_long.groupby("sid"):
        d = df_sid.sort_values("time")
        cache[int(sid)] = {
            "t": d["time"].values.astype(float),
            "c": d["C_obs"].values.astype(float),
            "r": d["R_obs"].values.astype(float),
        }
    return cache


def _dose_labels_from_subject_cache(subject_cache: dict):
    sids = sorted(subject_cache.keys())
    cmax = np.array([float(np.max(subject_cache[s]["c"])) for s in sids], dtype=float)
    eps = 1e-10
    labels = {}
    zero_mask = cmax <= eps
    for sid, z in zip(sids, zero_mask):
        if z:
            labels[sid] = "zero"
    nz_vals = cmax[~zero_mask]
    nz_sids = [sid for sid, z in zip(sids, zero_mask) if not z]
    if len(nz_sids) == 0:
        return labels
    if len(nz_sids) == 1:
        labels[nz_sids[0]] = "nz_mid"
        return labels
    q1, q2 = np.percentile(nz_vals, [33.0, 66.0])
    if q2 <= q1 + 1e-12:
        q1, q2 = np.percentile(nz_vals, [25.0, 75.0])
    for sid, v in zip(nz_sids, nz_vals):
        if v <= q1:
            labels[sid] = "nz_low"
        elif v <= q2:
            labels[sid] = "nz_mid"
        else:
            labels[sid] = "nz_high"
    return labels


def _dose_stratified_rmse(theta, terms, ec50, gamma, subject_cache, sid_val, dose_labels):
    bucket_sq = {}
    bucket_n = {}
    all_sq = 0.0
    all_n = 0
    for sid in sid_val:
        sid = int(sid)
        if sid not in subject_cache:
            continue
        s = subject_cache[sid]
        t, c, r = s["t"], s["c"], s["r"]
        if len(t) < 2:
            continue
        yhat = _simulate_on_grid(theta, terms, t, c, float(r[0]), ec50, gamma)
        if np.any(~np.isfinite(yhat)):
            continue
        resid = yhat - r
        sq = float(np.sum(resid**2))
        n = int(len(resid))
        all_sq += sq
        all_n += n
        b = dose_labels.get(sid, "unknown")
        bucket_sq[b] = bucket_sq.get(b, 0.0) + sq
        bucket_n[b] = bucket_n.get(b, 0) + n
    if all_n <= 0:
        return np.nan
    bucket_rmses = []
    for b, sq in bucket_sq.items():
        n = bucket_n.get(b, 0)
        if n > 0:
            bucket_rmses.append(float(np.sqrt(sq / n)))
    if len(bucket_rmses) == 0:
        return float(np.sqrt(all_sq / all_n))
    return float(np.mean(bucket_rmses))


def _temporal_shape_penalty(t_grid, yhat, ytrue):
    t = np.asarray(t_grid, dtype=float)
    yh = np.asarray(yhat, dtype=float)
    yt = np.asarray(ytrue, dtype=float)
    m = np.isfinite(t) & np.isfinite(yh) & np.isfinite(yt)
    if np.sum(m) < 6:
        return np.nan, np.nan, np.nan
    t = t[m]
    resid = (yh[m] - yt[m]).astype(float)
    ord_idx = np.argsort(t)
    t = t[ord_idx]
    resid = resid[ord_idx]
    if resid.size < 6:
        return np.nan, np.nan, np.nan

    q1 = np.quantile(t, 1.0 / 3.0)
    q2 = np.quantile(t, 2.0 / 3.0)
    early = resid[t <= q1]
    late = resid[t >= q2]
    if early.size < 2 or late.size < 2:
        mid = resid.size // 2
        early = resid[:mid]
        late = resid[mid:]
    if early.size < 2 or late.size < 2:
        return np.nan, np.nan, np.nan

    mu_e = float(np.mean(early))
    mu_l = float(np.mean(late))
    scale = float(np.std(resid) + 1e-8)
    # Generic mismatch score:
    # 1) penalize opposite-sign early/late bias (front/back shape mismatch),
    # 2) penalize large early-vs-late drift regardless of sign.
    sign_flip = max(0.0, -mu_e * mu_l) / (scale * scale)
    drift = abs(mu_l - mu_e) / scale
    return float(sign_flip + 0.5 * drift), mu_e, mu_l


def _global_residual_bias_penalty(yhat, ytrue):
    yh = np.asarray(yhat, dtype=float)
    yt = np.asarray(ytrue, dtype=float)
    m = np.isfinite(yh) & np.isfinite(yt)
    if np.sum(m) < 4:
        return np.nan, np.nan
    resid = (yh[m] - yt[m]).astype(float)
    scale = float(np.std(yt[m]) + 1e-8)
    mu = float(np.mean(resid))
    return float(abs(mu) / scale), mu


def _boundary_proximity_penalty(theta, lb, ub, margin_ratio: float = 0.05):
    th = np.asarray(theta, dtype=float)
    lo = np.asarray(lb, dtype=float)
    hi = np.asarray(ub, dtype=float)
    if th.size == 0 or lo.size != th.size or hi.size != th.size:
        return np.nan
    span = hi - lo
    finite = np.isfinite(th) & np.isfinite(lo) & np.isfinite(hi) & (span > 1e-12)
    if not np.any(finite):
        return np.nan
    d_lo = (th[finite] - lo[finite]) / span[finite]
    d_hi = (hi[finite] - th[finite]) / span[finite]
    d_min = np.minimum(d_lo, d_hi)
    mr = max(float(margin_ratio), 1e-6)
    prox = np.clip((mr - d_min) / mr, 0.0, 1.0)
    return float(np.mean(prox))


def _feature_identifiability_penalty(r_obs, c_obs, terms, ec50, gamma):
    rr = np.asarray(r_obs, dtype=float)
    cc = np.asarray(c_obs, dtype=float)
    if rr.size < 6 or cc.size != rr.size or len(terms) <= 1:
        return np.nan
    X = np.empty((rr.size, len(terms)), dtype=float)
    for i in range(rr.size):
        feats = build_pd_features(float(rr[i]), float(cc[i]), float(ec50), float(gamma))
        X[i, :] = [float(feats[t]) for t in terms]
    m = np.all(np.isfinite(X), axis=1)
    X = X[m]
    if X.shape[0] < max(6, X.shape[1] + 2):
        return np.nan
    X = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    X = X / scale
    try:
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan
    if s.size == 0 or not np.isfinite(s[0]):
        return np.nan
    s_min = float(max(np.min(s), 1e-8))
    cond = float(np.max(s) / s_min)
    # Penalty starts when condition number exceeds ~30 and grows smoothly.
    return float(max(0.0, np.log10(cond) - 1.5))


def _late_overprediction_penalty(mu_late, ytrue):
    if not np.isfinite(mu_late):
        return np.nan
    yt = np.asarray(ytrue, dtype=float)
    m = np.isfinite(yt)
    if np.sum(m) < 4:
        return np.nan
    scale = float(np.std(yt[m]) + 1e-8)
    return float(max(0.0, float(mu_late)) / scale)


def _build_hill_seed_pairs(
    n_restarts: int,
    ec50_0: float,
    gamma_0: float,
    ec50_lb: float,
    ec50_ub: float,
    gamma_lb: float,
    gamma_ub: float,
    c_values,
    gamma_grid=None,
):
    c_vals = np.asarray(c_values, dtype=float)
    c_pos = c_vals[np.isfinite(c_vals) & (c_vals > 1e-8)]
    if c_pos.size >= 4:
        ec50_grid = np.percentile(c_pos, [20.0, 40.0, 60.0, 80.0])
    elif c_pos.size >= 2:
        ec50_grid = np.percentile(c_pos, [25.0, 50.0, 75.0])
    else:
        ec50_grid = np.linspace(ec50_lb, ec50_ub, num=3)
    ec50_candidates = np.unique(
        np.clip(np.append(ec50_grid, [ec50_0, 0.5 * (ec50_lb + ec50_ub)]), ec50_lb, ec50_ub)
    )
    if gamma_grid is None:
        gamma_candidates = np.array([gamma_lb, 0.5 * (gamma_lb + gamma_ub), gamma_ub, gamma_0], dtype=float)
    else:
        gamma_candidates = np.asarray(gamma_grid, dtype=float)
        gamma_candidates = gamma_candidates[np.isfinite(gamma_candidates)]
        gamma_candidates = gamma_candidates[(gamma_candidates >= gamma_lb) & (gamma_candidates <= gamma_ub)]
        if gamma_candidates.size == 0:
            gamma_candidates = np.array([gamma_0], dtype=float)
    gamma_candidates = np.unique(np.clip(gamma_candidates, gamma_lb, gamma_ub))
    pairs = [(float(np.clip(ec50_0, ec50_lb, ec50_ub)), float(np.clip(gamma_0, gamma_lb, gamma_ub)))]
    for e in ec50_candidates:
        for g in gamma_candidates:
            pairs.append((float(e), float(g)))
    uniq = []
    seen = set()
    for e, g in pairs:
        key = (round(e, 8), round(g, 8))
        if key not in seen:
            uniq.append((e, g))
            seen.add(key)
    if len(uniq) >= n_restarts:
        return uniq[:n_restarts]
    out = []
    i = 0
    while len(out) < n_restarts:
        out.append(uniq[i % len(uniq)])
        i += 1
    return out


def _response_scaled_theta_bound(r_values, base_bound: float, min_bound: float):
    vals = np.asarray(r_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    candidates = [float(base_bound), float(min_bound)]
    if vals.size:
        max_abs = float(np.max(np.abs(vals)))
        dyn = float(np.max(vals) - np.min(vals))
        # Structure coefficients are dR/dt-scale. Direct-effect bridge models need
        # theta_1 ~= k_resp * baseline, which can be several times the response level.
        candidates.extend([6.0 * max_abs, 12.0 * dyn])
    bound = max(candidates)
    return float(min(max(bound, 30.0), 5000.0))


def _direct_term_bounds(terms, large_bound: float):
    bounds = []
    for term in terms:
        key = str(term).replace(" ", "").lower()
        if key == "1" or key in {"emax(c)", "hill(c)"}:
            b = large_bound
        elif key == "r":
            b = min(large_bound, 50.0)
        elif key == "c":
            b = min(large_bound, 50.0)
        elif key == "c^2":
            b = min(large_bound, 10.0)
        elif "*r" in key:
            b = min(large_bound, 10.0)
        else:
            b = min(large_bound, 50.0)
        bounds.append(float(max(b, 1e-6)))
    return np.asarray(bounds, dtype=float)


def _early_late_residual_means(t_grid, resid):
    t = np.asarray(t_grid, dtype=float)
    r = np.asarray(resid, dtype=float)
    m = np.isfinite(t) & np.isfinite(r)
    if np.sum(m) < 6:
        return np.nan, np.nan
    t = t[m]
    r = r[m]
    ord_idx = np.argsort(t)
    t = t[ord_idx]
    r = r[ord_idx]
    q1 = np.quantile(t, 1.0 / 3.0)
    q2 = np.quantile(t, 2.0 / 3.0)
    early = r[t <= q1]
    late = r[t >= q2]
    if early.size < 2 or late.size < 2:
        mid = r.size // 2
        early = r[:mid]
        late = r[mid:]
    if early.size < 2 or late.size < 2:
        return np.nan, np.nan
    return float(np.mean(early)), float(np.mean(late))


def run_multistart_refit(
    df_long: pd.DataFrame,
    topk_payload: dict,
    n_restarts: int = 12,
    seed: int = 20260419,
    theta_scale: float = 0.3,
    max_nfev: int = 3000,
    train_frac: float = 0.7,
    keep_time_order: bool = True,
    use_bounds: bool = True,
    theta_abs_bound: float = 30.0,
    # 通用温和选择参数（默认偏简单，但不过分）
    lambda_k: float = 1.0,
    lambda_collinear: float = 1.0,
    lambda_stability: float = 0.2,
    lambda_dose_stratified: float = 2.0,
    lambda_temporal_shape: float = 3.0,
    lambda_late_overpredict: float = 10.0,
    lambda_residual_bias: float = 2.0,
    lambda_identifiability: float = 3.0,
    lambda_boundary_proximity: float = 4.0,
    lambda_nonconverged: float = 80.0,
    lambda_train_residual_bias: float = 2.0,
    lambda_train_temporal_bias: float = 1.5,
    redundancy_groups=None,
    direct_hill_constraints: bool = False,
    direct_ec50_lb: float = 0.5,
    direct_ec50_ub: float = 8.0,
    direct_gamma_lb: float = 0.5,
    direct_gamma_ub: float = 3.0,
    direct_theta_abs_bound: float = 400.0,
    hill_gamma_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
):
    required = {"sid", "time", "C_obs", "R_obs"}
    if not required.issubset(df_long.columns):
        raise ValueError(f"df_long must contain {required}, got {list(df_long.columns)}")

    cands = topk_payload.get("candidates", [])
    if not isinstance(cands, list) or len(cands) == 0:
        raise ValueError("topk_payload['candidates'] is empty or invalid.")

    ec50 = float(topk_payload.get("ec50_hat", 4.0) or 4.0)
    gamma = float(topk_payload.get("gamma_hat", 2.0) or 2.0)

    if redundancy_groups is None:
        # 默认仅做“同族项”温和惩罚，适度通用
        redundancy_groups = [
            ["C", "C^2", "Emax(C)", "Hill(C)"],
            ["C*R", "Emax(C)*R", "Hill(C)*R"],
        ]

    legal_terms = set(get_pd_term_names())

    agg = _build_agg_splits(df_long.copy(), train_frac, keep_time_order, seed)
    t, c, r = agg["t"], agg["c"], agg["r"]
    tr_idx, va_idx = agg["tr_idx"], agg["va_idx"]
    r_mean, r_std = agg["r_mean"], agg["r_std"]
    sid_va = agg["sid_va"]
    subject_cache = _build_subject_cache(df_long.copy())
    dose_labels = _dose_labels_from_subject_cache(subject_cache)
    scaled_theta_abs_bound = _response_scaled_theta_bound(r, theta_abs_bound, direct_theta_abs_bound)

    rng = np.random.default_rng(seed)
    all_rows, best_rows = [], []

    for i, cand in enumerate(cands, 1):
        cid = int(cand.get("candidate_id", i))
        rank = int(cand.get("rank", i))
        terms = [str(x) for x in cand.get("terms", [])]
        if not terms:
            continue

        unknown = [x for x in terms if x not in legal_terms]
        if unknown:
            all_rows.append({
                "candidate_id": cid, "rank": rank, "restart": 0,
                "terms": " + ".join(terms), "k": len(terms),
                "success": 0, "status": -101, "train_cost": np.nan,
                "sse_val": np.inf, "rmse_val": np.nan, "logLik_val": np.nan,
                "AIC_val": np.nan, "BIC_val": np.nan,
                "selection_score": np.inf,
                "k_penalty": np.inf, "collinear_penalty": np.inf, "stability_penalty": np.inf,
                "theta0_json": "[]", "theta_hat_json": "[]",
                "message": f"unknown_terms:{unknown}",
                "ec50_hat": ec50, "gamma_hat": gamma,
            })
            continue

        has_hill = any(t in ("Hill(C)", "Hill(C)*R") for t in terms)

        # EC50/gamma 初值：优先读 per-candidate JSON 值（Step 4 写入的），否则用 Step 3 全局值
        ec50_0 = float(cand.get("ec50_hat", ec50) or ec50)
        gamma_0 = float(cand.get("gamma_hat", gamma) or gamma)

        init = np.array(cand.get("init_params", cand.get("theta0", [0.0] * len(terms))), dtype=float)
        if init.size != len(terms):
            init = np.zeros(len(terms), dtype=float)

        # R0：取群体均值数据第一个时间点的响应值
        r0 = float(r[0])  # tr_idx[0] 是索引，直接用 r[0] 更直接

        theta_abs_bound_eff = scaled_theta_abs_bound if direct_hill_constraints else float(theta_abs_bound)
        struct_bounds = (
            _direct_term_bounds(terms, theta_abs_bound_eff)
            if direct_hill_constraints
            else np.full(len(terms), theta_abs_bound_eff)
        )

        if has_hill:
            # theta = [theta_struct(1..p), EC50, gamma]，参与拟合
            n_struct = len(terms)
            ec50_lb, ec50_ub = 0.5, 20.0
            gamma_lb, gamma_ub = 0.5, 6.0
            if direct_hill_constraints:
                c_pos = c[c > 1e-8]
                if c_pos.size > 0:
                    ec50_lb = max(float(direct_ec50_lb), float(np.percentile(c_pos, 10.0) * 0.5))
                    ec50_ub_data = float(np.percentile(c_pos, 95.0) * 1.2)
                    ec50_ub = min(float(direct_ec50_ub), max(ec50_lb + 0.2, ec50_ub_data))
                else:
                    ec50_lb = float(direct_ec50_lb)
                    ec50_ub = float(direct_ec50_ub)
                gamma_lb = float(direct_gamma_lb)
                gamma_ub = float(direct_gamma_ub)
            ec50_0 = float(np.clip(ec50_0, ec50_lb, ec50_ub))
            gamma_0 = float(np.clip(gamma_0, gamma_lb, gamma_ub))
            gamma_grid_eff = np.asarray(hill_gamma_grid, dtype=float)
            gamma_grid_eff = gamma_grid_eff[np.isfinite(gamma_grid_eff)]
            gamma_grid_eff = gamma_grid_eff[(gamma_grid_eff >= gamma_lb) & (gamma_grid_eff <= gamma_ub)]
            if gamma_grid_eff.size == 0:
                gamma_grid_eff = np.array([gamma_0], dtype=float)
            gamma_0 = float(gamma_grid_eff[np.argmin(np.abs(gamma_grid_eff - gamma_0))])
            theta_full_bound_lo = np.append(-struct_bounds, [ec50_lb])
            theta_full_bound_hi = np.append(struct_bounds, [ec50_ub])
            theta_full_init = np.append(init, [ec50_0])
            hill_seed_pairs = _build_hill_seed_pairs(
                n_restarts=n_restarts,
                ec50_0=ec50_0,
                gamma_0=gamma_0,
                ec50_lb=ec50_lb,
                ec50_ub=ec50_ub,
                gamma_lb=gamma_lb,
                gamma_ub=gamma_ub,
                c_values=c,
                gamma_grid=gamma_grid_eff,
            )
        else:
            theta_full_bound_lo = -struct_bounds
            theta_full_bound_hi = struct_bounds
            theta_full_init = init
            hill_seed_pairs = None
        fixed_gamma = {"value": gamma_0}

        def train_residual(theta_full):
            if has_hill:
                th_struct = theta_full[:-1]
                eg_ec50 = theta_full[-1]
                eg_gamma = fixed_gamma["value"]
            else:
                th_struct = theta_full
                eg_ec50 = ec50_0
                eg_gamma = gamma_0
            yhat = _simulate_on_grid(th_struct, terms, t, c, r0, eg_ec50, eg_gamma)
            if np.any(~np.isfinite(yhat)):
                return np.ones(len(tr_idx)) * 1e3
            y_tr = (r[tr_idx] - r_mean) / r_std
            yhat_tr = (yhat[tr_idx] - r_mean) / r_std
            resid_tr = yhat_tr - y_tr
            penalty_terms = []
            if lambda_train_residual_bias > 0.0:
                mu = float(np.mean(resid_tr))
                penalty_terms.append(np.sqrt(lambda_train_residual_bias) * mu)
            if lambda_train_temporal_bias > 0.0:
                mu_e, mu_l = _early_late_residual_means(t[tr_idx], resid_tr)
                if np.isfinite(mu_e) and np.isfinite(mu_l):
                    penalty_terms.append(np.sqrt(lambda_train_temporal_bias) * (mu_l - mu_e))
            if penalty_terms:
                return np.concatenate([resid_tr, np.asarray(penalty_terms, dtype=float)])
            return resid_tr

        def val_metrics(theta_full):
            if has_hill:
                th_struct = theta_full[:-1]
                eg_ec50 = theta_full[-1]
                eg_gamma = fixed_gamma["value"]
            else:
                th_struct = theta_full
                eg_ec50 = ec50_0
                eg_gamma = gamma_0
            yhat = _simulate_on_grid(th_struct, terms, t, c, r0, eg_ec50, eg_gamma)
            if np.any(~np.isfinite(yhat)):
                return (
                    np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                )
            y_va = (r[va_idx] - r_mean) / r_std
            yhat_va = (yhat[va_idx] - r_mean) / r_std
            resid = yhat_va - y_va
            n = resid.size
            sse = float(np.sum(resid**2))
            rmse = float(np.sqrt(sse / max(n, 1)))
            sigma2 = max(sse / max(n, 1), 1e-12)
            loglik = float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))
            k_full = len(th_struct) + (2 if has_hill else 0)
            dose_rmse = _dose_stratified_rmse(
                th_struct,
                terms,
                eg_ec50,
                eg_gamma,
                subject_cache=subject_cache,
                sid_val=sid_va,
                dose_labels=dose_labels,
            )
            temporal_pen, mu_early, mu_late = _temporal_shape_penalty(t, yhat, r)
            residual_bias_pen, resid_mu = _global_residual_bias_penalty(yhat, r)
            late_overpredict_pen = _late_overprediction_penalty(mu_late, r)
            identifiability_pen = _feature_identifiability_penalty(r, c, terms, eg_ec50, eg_gamma)
            return (
                sse,
                rmse,
                loglik,
                n,
                eg_ec50,
                eg_gamma,
                dose_rmse,
                temporal_pen,
                mu_early,
                mu_late,
                residual_bias_pen,
                resid_mu,
                late_overpredict_pen,
                identifiability_pen,
            )

        if use_bounds:
            lb = theta_full_bound_lo
            ub = theta_full_bound_hi
        else:
            lb, ub = -np.inf, np.inf

        restart_rows = []

        for rs in range(1, n_restarts + 1):
            if rs == 1:
                theta0 = theta_full_init.copy()
                if has_hill:
                    fixed_gamma["value"] = hill_seed_pairs[0][1]
            else:
                # 非 Hill 模型：只扰动结构参数
                if has_hill:
                    pert = rng.normal(0, theta_scale, size=len(init))
                    ec50_seed, gamma_seed = hill_seed_pairs[(rs - 1) % len(hill_seed_pairs)]
                    fixed_gamma["value"] = gamma_seed
                    theta0 = np.append(init + pert, [ec50_seed])
                else:
                    theta0 = init + rng.normal(0, theta_scale, size=len(init))
            if use_bounds:
                theta0 = np.clip(theta0, lb, ub)

            try:
                out = least_squares(
                    train_residual, theta0, method="trf", bounds=(lb, ub),
                    max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8
                )

                (
                    sse_val,
                    rmse_val,
                    loglik_val,
                    n_val,
                    ec50_fit,
                    gamma_fit,
                    dose_rmse,
                    temporal_pen,
                    mu_early,
                    mu_late,
                    residual_bias_pen,
                    resid_mu,
                    late_overpredict_pen,
                    identifiability_pen,
                ) = val_metrics(out.x)
                k_full = len(terms) + (2 if has_hill else 0)

                if np.isfinite(loglik_val):
                    aic_val = float(-2 * loglik_val + 2 * k_full)
                    bic_val = float(-2 * loglik_val + np.log(max(n_val, 1)) * k_full)
                else:
                    aic_val = np.nan
                    bic_val = np.nan

                row = {
                    "candidate_id": cid, "rank": rank, "restart": rs,
                    "terms": " + ".join(terms), "k": k_full,
                    "success": int(out.success), "status": int(out.status),
                    "train_cost": float(out.cost),
                    "sse_val": sse_val, "rmse_val": rmse_val, "logLik_val": loglik_val,
                    "AIC_val": aic_val, "BIC_val": bic_val,
                    "selection_score": np.nan,
                    "k_penalty": np.nan, "collinear_penalty": np.nan, "stability_penalty": np.nan,
                    "dose_stratified_rmse": dose_rmse,
                    "dose_penalty": np.nan,
                    "temporal_shape_penalty": temporal_pen,
                    "temporal_penalty": np.nan,
                    "residual_bias_penalty": residual_bias_pen,
                    "residual_bias_weighted": np.nan,
                    "late_overpredict_penalty": late_overpredict_pen,
                    "late_overpredict_weighted": np.nan,
                    "identifiability_penalty": identifiability_pen,
                    "identifiability_weighted": np.nan,
                    "boundary_proximity_penalty": _boundary_proximity_penalty(out.x, lb, ub),
                    "boundary_penalty": np.nan,
                    "nonconverged_penalty": np.nan,
                    "residual_mean_all": resid_mu,
                    "resid_mean_early": mu_early,
                    "resid_mean_late": mu_late,
                    "theta0_json": json.dumps(
                        (np.append(theta0, fixed_gamma["value"]) if has_hill else theta0).tolist(),
                        ensure_ascii=False,
                    ),
                    "theta_hat_json": json.dumps(
                        (np.append(out.x, fixed_gamma["value"]) if has_hill else out.x).tolist(),
                        ensure_ascii=False,
                    ),
                    "message": str(out.message),
                    "ec50_hat": ec50_fit, "gamma_hat": gamma_fit,
                }
            except Exception as e:
                row = {
                    "candidate_id": cid, "rank": rank, "restart": rs,
                    "terms": " + ".join(terms), "k": len(terms) + (2 if has_hill else 0),
                    "success": 0, "status": -999, "train_cost": np.nan,
                    "sse_val": np.inf, "rmse_val": np.nan, "logLik_val": np.nan,
                    "AIC_val": np.nan, "BIC_val": np.nan,
                    "selection_score": np.inf,
                    "k_penalty": np.inf, "collinear_penalty": np.inf, "stability_penalty": np.inf,
                    "dose_stratified_rmse": np.nan,
                    "dose_penalty": np.nan,
                    "temporal_shape_penalty": np.nan,
                    "temporal_penalty": np.nan,
                    "residual_bias_penalty": np.nan,
                    "residual_bias_weighted": np.nan,
                    "late_overpredict_penalty": np.nan,
                    "late_overpredict_weighted": np.nan,
                    "identifiability_penalty": np.nan,
                    "identifiability_weighted": np.nan,
                    "boundary_proximity_penalty": np.nan,
                    "boundary_penalty": np.nan,
                    "nonconverged_penalty": np.nan,
                    "residual_mean_all": np.nan,
                    "resid_mean_early": np.nan,
                    "resid_mean_late": np.nan,
                    "theta0_json": json.dumps(
                        (np.append(theta0, fixed_gamma["value"]) if has_hill else theta0).tolist(),
                        ensure_ascii=False,
                    ),
                    "theta_hat_json": "[]",
                    "message": repr(e),
                    "ec50_hat": ec50_0,
                    "gamma_hat": gamma_0,
                }

            restart_rows.append(row)
            all_rows.append(row)

        # 候选内稳定性（基于BIC波动）
        valid_bics = [r["BIC_val"] for r in restart_rows if np.isfinite(r["BIC_val"])]
        bic_std = float(np.std(valid_bics)) if len(valid_bics) > 0 else np.nan
        bic_mean = float(np.mean(valid_bics)) if len(valid_bics) > 0 else np.nan
        n_valid = int(len(valid_bics))

        # 给每个重启补 selection_score（同一候选共享稳定性项）
        for rr in restart_rows:
            score, detail = compute_selection_score(
                bic_val=rr["BIC_val"],
                k=rr["k"],
                bic_std=bic_std,
                terms=terms,
                lambda_k=lambda_k,
                lambda_collinear=lambda_collinear,
                lambda_stability=lambda_stability,
                redundancy_groups=redundancy_groups,
            )
            dose_rmse = rr.get("dose_stratified_rmse", np.nan)
            dose_penalty = float(lambda_dose_stratified * dose_rmse) if np.isfinite(dose_rmse) else 0.0
            temporal_shape = rr.get("temporal_shape_penalty", np.nan)
            temporal_penalty = float(lambda_temporal_shape * temporal_shape) if np.isfinite(temporal_shape) else 0.0
            residual_bias = rr.get("residual_bias_penalty", np.nan)
            residual_bias_weighted = float(lambda_residual_bias * residual_bias) if np.isfinite(residual_bias) else 0.0
            late_raw = rr.get("late_overpredict_penalty", np.nan)
            late_weight_mult = 1.5 if direct_hill_constraints else 1.0
            late_weighted = (
                float(lambda_late_overpredict * late_weight_mult * late_raw) if np.isfinite(late_raw) else 0.0
            )
            ident_raw = rr.get("identifiability_penalty", np.nan)
            ident_weighted = float(lambda_identifiability * ident_raw) if np.isfinite(ident_raw) else 0.0
            boundary_raw = rr.get("boundary_proximity_penalty", np.nan)
            boundary_penalty = float(lambda_boundary_proximity * boundary_raw) if np.isfinite(boundary_raw) else 0.0
            is_converged = bool(rr.get("success", 0) == 1 and rr.get("status", 0) > 0)
            nonconverged_penalty = 0.0 if is_converged else float(lambda_nonconverged)
            rr["selection_score"] = float(
                score
                + dose_penalty
                + temporal_penalty
                + residual_bias_weighted
                + late_weighted
                + ident_weighted
                + boundary_penalty
                + nonconverged_penalty
            )
            rr["k_penalty"] = detail["k_penalty"]
            rr["collinear_penalty"] = detail["collinear_penalty"]
            rr["stability_penalty"] = detail["stability_penalty"]
            rr["dose_penalty"] = dose_penalty
            rr["temporal_penalty"] = temporal_penalty
            rr["residual_bias_weighted"] = residual_bias_weighted
            rr["late_overpredict_weighted"] = late_weighted
            rr["identifiability_weighted"] = ident_weighted
            rr["boundary_penalty"] = boundary_penalty
            rr["nonconverged_penalty"] = nonconverged_penalty
            rr["BIC_val_std_across_restarts"] = bic_std
            rr["BIC_val_mean_across_restarts"] = bic_mean
            rr["n_valid_restarts"] = n_valid

        # 候选最佳重启
        best = sorted(
            restart_rows,
            key=lambda x: (
                np.inf if not np.isfinite(x["selection_score"]) else x["selection_score"],
                np.inf if not np.isfinite(x["BIC_val"]) else x["BIC_val"],
                np.inf if not np.isfinite(x["AIC_val"]) else x["AIC_val"],
            )
        )[0]
        best_rows.append(best)

    df_all = pd.DataFrame(all_rows)
    df_best = pd.DataFrame(best_rows)

    if len(df_best):
        df_best = df_best.sort_values(
            ["selection_score", "BIC_val", "AIC_val"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    return df_all, df_best
