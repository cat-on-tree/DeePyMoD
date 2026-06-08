import json
import numpy as np
import pandas as pd


def _flatten_terms(raw_terms):
    if isinstance(raw_terms, (list, tuple)):
        if len(raw_terms) > 0 and isinstance(raw_terms[0], (list, tuple)):
            flat = []
            for eq in raw_terms:
                flat.extend([str(x) for x in eq])
            return flat
        return [str(x) for x in raw_terms]
    return [str(raw_terms)]


def _candidate_has_hill(terms):
    for term in terms:
        key = str(term).replace(" ", "").lower()
        if "hill(" in key:
            return True
    return False


def _to_float_list(values):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return None
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def _sanitize_candidate_params(terms, theta_like, ec50_default, gamma_default):
    flat_terms = _flatten_terms(terms)
    p = len(flat_terms)
    has_hill = _candidate_has_hill(flat_terms)
    theta_arr = _to_float_list(theta_like)
    ec50_val = float(ec50_default)
    gamma_val = float(gamma_default)

    if theta_arr is None:
        return None, ec50_val, gamma_val

    if has_hill and len(theta_arr) == p + 2:
        ec50_val = float(theta_arr[-2])
        gamma_val = float(theta_arr[-1])
        return theta_arr[:p], ec50_val, gamma_val

    if len(theta_arr) == p:
        return theta_arr, ec50_val, gamma_val

    return None, ec50_val, gamma_val


def to_population_mean_df(pop_data):
    """将个体数据聚合成群体均值（按 time 分组）。"""
    df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
    agg = (
        df.groupby("time", as_index=False)
          .agg(sid=("sid", "min"), C_obs=("C_obs", "mean"), R_obs=("R_obs", "mean"))
          .sort_values("time")
          .reset_index(drop=True)
    )
    # 群体均值数据只保留 sid=0（代表群体），用于 MATLAB 拟合
    return agg[["sid", "time", "C_obs", "R_obs"]].copy()


def export_nlme_inputs(
    pop_data,
    top_results,
    out_dir="artifacts/nlme",
    use_population_mean=False,
    nlme_mode="screen",
    nlme_multistart_on_fail=True,
    fit_hints=None,
):
    """
    pop_data columns: [sid, time, C_obs, R_obs]
    top_results: run_single_discovery 返回的 top_results
    use_population_mean: True → 输出群体均值数据（与 Step 4 一致的优化问题）
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
    df_raw["sid"] = df_raw["sid"].astype(int)

    if use_population_mean:
        df_out = to_population_mean_df(pop_data)
        df_out["sid"] = df_out["sid"].astype(int)
    else:
        df_out = df_raw

    data_csv = f"{out_dir}/pkpd_long.csv"
    cand_json = f"{out_dir}/topk_candidates.json"

    df_out.to_csv(data_csv, index=False)

    candidates = []
    for i, r in enumerate(top_results, 1):
        bic_raw = r.get("score_bic_val") if "score_bic_val" in r else None
        if bic_raw is None:
            bic_raw = r.get("score") if "score" in r else None
        bic_val_python = float(bic_raw) if bic_raw is not None else float("nan")

        mse_raw = r.get("mse_val") if "mse_val" in r else None
        if mse_raw is None:
            mse_raw = r.get("mse_val_python") if "mse_val_python" in r else None
        mse_val_python = float(mse_raw) if mse_raw is not None else float("nan")

        ec50_raw = r.get("ec50_hat") if "ec50_hat" in r else None
        gamma_raw = r.get("gamma_hat") if "gamma_hat" in r else None
        ec50_val = float(ec50_raw) if ec50_raw is not None and not np.isnan(float(ec50_raw)) else 4.0
        gamma_val = float(gamma_raw) if gamma_raw is not None and not np.isnan(float(gamma_raw)) else 2.0

        theta_hat_struct, ec50_val, gamma_val = _sanitize_candidate_params(
            terms=r["terms"],
            theta_like=r.get("theta_hat"),
            ec50_default=ec50_val,
            gamma_default=gamma_val,
        )
        init_params_struct, _, _ = _sanitize_candidate_params(
            terms=r["terms"],
            theta_like=r.get("init_params"),
            ec50_default=ec50_val,
            gamma_default=gamma_val,
        )

        cand = {
            "rank": i,
            "terms": r["terms"],
            "k": int(r["k"]),
            "bic_val_python": bic_val_python,
            "mse_val_python": mse_val_python,
            "ec50_hat": ec50_val,
            "gamma_hat": gamma_val,
        }
        if theta_hat_struct is not None:
            cand["theta_hat"] = theta_hat_struct
        if init_params_struct is not None:
            cand["init_params"] = init_params_struct
        candidates.append(cand)

    with open(cand_json, "w", encoding="utf-8") as f:
        payload = {
            "nlme_mode": nlme_mode,
            "nlme_multistart_on_fail": bool(nlme_multistart_on_fail),
            "candidates": candidates,
        }
        if isinstance(fit_hints, dict) and fit_hints:
            payload["fit_hints"] = fit_hints
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return data_csv, cand_json
