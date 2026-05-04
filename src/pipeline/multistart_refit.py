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

    return dict(t=t, c=c, r=r, tr_idx=tr_idx, va_idx=va_idx, r_mean=r_mean, r_std=r_std)


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
    theta_abs_bound: float = 50.0,
    # 通用温和选择参数（默认偏简单，但不过分）
    lambda_k: float = 1.0,
    lambda_collinear: float = 1.0,
    lambda_stability: float = 0.2,
    redundancy_groups=None,
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

        if has_hill:
            # theta = [theta_struct(1..p), EC50, gamma]，参与拟合
            n_struct = len(terms)
            theta_full_bound_lo = np.append(np.full(n_struct, -theta_abs_bound), [0.5, 1.0])
            theta_full_bound_hi = np.append(np.full(n_struct, theta_abs_bound), [20.0, 6.0])
            theta_full_init = np.append(init, [ec50_0, gamma_0])
        else:
            theta_full_bound_lo = np.full(len(terms), -theta_abs_bound)
            theta_full_bound_hi = np.full(len(terms), theta_abs_bound)
            theta_full_init = init

        def train_residual(theta_full):
            if has_hill:
                th_struct = theta_full[:-2]
                eg_ec50 = theta_full[-2]
                eg_gamma = theta_full[-1]
            else:
                th_struct = theta_full
                eg_ec50 = ec50_0
                eg_gamma = gamma_0
            yhat = _simulate_on_grid(th_struct, terms, t, c, r0, eg_ec50, eg_gamma)
            if np.any(~np.isfinite(yhat)):
                return np.ones(len(tr_idx)) * 1e3
            y_tr = (r[tr_idx] - r_mean) / r_std
            yhat_tr = (yhat[tr_idx] - r_mean) / r_std
            return yhat_tr - y_tr

        def val_metrics(theta_full):
            if has_hill:
                th_struct = theta_full[:-2]
                eg_ec50 = theta_full[-2]
                eg_gamma = theta_full[-1]
            else:
                th_struct = theta_full
                eg_ec50 = ec50_0
                eg_gamma = gamma_0
            yhat = _simulate_on_grid(th_struct, terms, t, c, r0, eg_ec50, eg_gamma)
            if np.any(~np.isfinite(yhat)):
                return np.inf, np.nan, np.nan, np.nan, np.nan, np.nan
            y_va = (r[va_idx] - r_mean) / r_std
            yhat_va = (yhat[va_idx] - r_mean) / r_std
            resid = yhat_va - y_va
            n = resid.size
            sse = float(np.sum(resid**2))
            rmse = float(np.sqrt(sse / max(n, 1)))
            sigma2 = max(sse / max(n, 1), 1e-12)
            loglik = float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))
            k_full = len(th_struct) + (2 if has_hill else 0)
            return sse, rmse, loglik, n, eg_ec50, eg_gamma

        if use_bounds:
            lb = theta_full_bound_lo
            ub = theta_full_bound_hi
        else:
            lb, ub = -np.inf, np.inf

        restart_rows = []

        for rs in range(1, n_restarts + 1):
            if rs == 1:
                theta0 = theta_full_init.copy()
            else:
                # 非 Hill 模型：只扰动结构参数
                if has_hill:
                    pert = rng.normal(0, theta_scale, size=len(init))
                    theta0 = np.append(init + pert, [ec50_0, gamma_0])
                else:
                    theta0 = init + rng.normal(0, theta_scale, size=len(init))
            if use_bounds:
                theta0 = np.clip(theta0, lb, ub)

            try:
                out = least_squares(
                    train_residual, theta0, method="trf", bounds=(lb, ub),
                    max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8
                )

                sse_val, rmse_val, loglik_val, n_val, ec50_fit, gamma_fit = val_metrics(out.x)
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
                    "theta0_json": json.dumps(theta0.tolist(), ensure_ascii=False),
                    "theta_hat_json": json.dumps(out.x.tolist(), ensure_ascii=False),
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
                    "theta0_json": json.dumps(theta0.tolist(), ensure_ascii=False),
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
            rr["selection_score"] = score
            rr["k_penalty"] = detail["k_penalty"]
            rr["collinear_penalty"] = detail["collinear_penalty"]
            rr["stability_penalty"] = detail["stability_penalty"]
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