import json
import pandas as pd


def export_nlme_inputs(pop_data, top_results, out_dir="artifacts/nlme"):
    """
    pop_data columns: [sid, time, C_obs, R_obs]
    top_results: run_single_discovery 返回的 top_results
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
    df["sid"] = df["sid"].astype(int)

    data_csv = f"{out_dir}/pkpd_long.csv"
    cand_json = f"{out_dir}/topk_candidates.json"

    df.to_csv(data_csv, index=False)

    candidates = []
    for i, r in enumerate(top_results, 1):
        candidates.append({
            "rank": i,
            "terms": r["terms"],
            "k": int(r["k"]),
            "bic_val_python": float(r["score"]),
            "mse_val_python": float(r["mse_val"]),
        })

    with open(cand_json, "w", encoding="utf-8") as f:
        json.dump({"candidates": candidates}, f, ensure_ascii=False, indent=2)

    return data_csv, cand_json