import json
import numpy as np
import pandas as pd


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


def export_nlme_inputs(pop_data, top_results, out_dir="artifacts/nlme", use_population_mean=False):
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

        candidates.append({
            "rank": i,
            "terms": r["terms"],
            "k": int(r["k"]),
            "bic_val_python": bic_val_python,
            "mse_val_python": mse_val_python,
            "ec50_hat": ec50_val,
            "gamma_hat": gamma_val,
        })

    with open(cand_json, "w", encoding="utf-8") as f:
        json.dump({"candidates": candidates}, f, ensure_ascii=False, indent=2)

    return data_csv, cand_json