import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_pkpd_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "sid" in df.columns:
        df["sid"] = df["sid"].astype(int)
    return df


def load_optional_json(json_path: str) -> dict:
    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_data_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["sid"].nunique()) if "sid" in df.columns else None,
        "time_min": float(df["time"].min()),
        "time_max": float(df["time"].max()),
        "C_min": float(df["C_obs"].min()),
        "C_max": float(df["C_obs"].max()),
        "R_min": float(df["R_obs"].min()),
        "R_max": float(df["R_obs"].max()),
    }


def save_snapshot_and_summary(df: pd.DataFrame, tables_dir: str):
    os.makedirs(tables_dir, exist_ok=True)
    snap_csv = os.path.join(tables_dir, "pkpd_long_snapshot.csv")
    summary_json = os.path.join(tables_dir, "data_summary.json")

    df.to_csv(snap_csv, index=False)
    summary = compute_data_summary(df)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return snap_csv, summary_json, summary


def plot_pkpd_scatter(df: pd.DataFrame, fig_dir: str, filename: str = "pkpd_scatter.png") -> str:
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, filename)

    plt.figure(figsize=(14, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(df["time"], df["C_obs"], s=14, alpha=0.55)
    ax1.set_title("PK Scatter: C_obs vs time")
    ax1.set_xlabel("time")
    ax1.set_ylabel("C_obs")

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(df["time"], df["R_obs"], s=14, alpha=0.55, color="tab:orange")
    ax2.set_title("PD Scatter: R_obs vs time")
    ax2.set_xlabel("time")
    ax2.set_ylabel("R_obs")

    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(df["C_obs"], df["R_obs"], s=14, alpha=0.55, color="tab:green")
    ax3.set_title("Exposure-Response: R_obs vs C_obs")
    ax3.set_xlabel("C_obs")
    ax3.set_ylabel("R_obs")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close()

    return fig_path