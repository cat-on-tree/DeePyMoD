"""
诊断图模块：GOF / 残差 / VPC / Bootstrap
支持任意 terms + theta 组合，自动处理归一化尺度 refit。

调用方式：
    from src.report.diagnostics import run_diagnostics
    results = run_diagnostics(df, df_cmp, df_ms, df_ms_all, ...)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from src.models.pd_features import build_pd_features


# ------------------------------------------------------------------
# 核心工具
# ------------------------------------------------------------------

def denormalize_theta(theta_norm, r_mean, r_std):
    """
    将归一化尺度的 theta 反归一化到原始物理尺度。

    归一化过程：z = (R - r_mean) / r_std
    目标函数中 dR/dt = sum(theta_raw[i] * feat_i(R, C))
    在归一化空间中：dz/dt = sum(theta_norm[i] * feat_i(R_norm, C))
    其中 R = r_mean + r_std * R_norm

    对线性项（常数项、线性R项）有：
      theta_norm = theta_raw * r_std
    对 Emax/Hill 等非线性项（参数ec50固定下为常数因子）：
      近似 theta_norm ≈ theta_raw * r_std（足够好的初始化）

    所以反归一化为：theta_raw ≈ theta_norm / r_std
    """
    arr = np.asarray(theta_norm, dtype=float)
    return np.clip(arr / r_std, -50, 50)


def fit_theta_on_original_scale(terms_list, ec50, gamma, df_in, theta_init_raw, max_nfev=2000):
    """
    用原始尺度残差（无归一化）refit theta。
    返回 (theta_fit, success)。

    Parameters
    ----------
    terms_list : list of str
    ec50, gamma : float
    df_in : pd.DataFrame with sid/time/C_obs/R_obs
    theta_init_raw : array, 反归一化后的初始猜测
    """
    k = len(terms_list)
    term_idx = {t: i for i, t in enumerate(terms_list)}

    # 按 sid 预建插值与索引
    sid_data = {}
    for sid, grp in df_in.groupby("sid"):
        sg = grp.sort_values("time")
        sid_data[sid] = {
            "t": sg["time"].values.astype(float),
            "C": sg["C_obs"].values.astype(float),
            "R": sg["R_obs"].values.astype(float),
        }

    def residuals_raw(theta):
        total = []
        for sid, sd in sid_data.items():
            t_s, C_s, R_s = sd["t"], sd["C"], sd["R"]
            c_interp = interp1d(t_s, C_s, kind="linear",
                                fill_value="extrapolate", assume_sorted=True)

            def rhs(t, R):
                return [sum(theta[term_idx[t_]] *
                            build_pd_features(R[0], float(c_interp(t)), ec50, gamma)[t_]
                            for t_ in terms_list)]

            sol = solve_ivp(rhs, (t_s[0], t_s[-1]), [R_s[0]],
                           t_eval=t_s, method="RK45", rtol=1e-4, atol=1e-6)
            pred = sol.y[0] if sol.success else np.full_like(t_s, np.nan)
            total.extend(R_s - pred)
        return np.array(total)

    try:
        out = least_squares(residuals_raw, theta_init_raw, method="trf",
                            bounds=([-50] * k, [50] * k),
                            max_nfev=max_nfev, xtol=1e-7, ftol=1e-7)
        return out.x, bool(out.success)
    except Exception:
        return theta_init_raw, False


def predict_R_per_subject(sid_df, theta, ec50, gamma, terms_list):
    """ODE 积分预测 R(t)，返回与 sid_df 行序一致的预测数组。"""
    t_arr = sid_df.sort_values("time")["time"].values.astype(float)
    C_arr = sid_df.sort_values("time")["C_obs"].values.astype(float)
    c_interp = interp1d(t_arr, C_arr, kind="linear",
                        fill_value="extrapolate", assume_sorted=True)
    term_idx = {t: i for i, t in enumerate(terms_list)}

    def rhs(t, R):
        feats = build_pd_features(R[0], float(c_interp(t)), ec50, gamma)
        return [sum(theta[term_idx[t_]] * feats[t_] for t_ in terms_list)]

    R0 = float(sid_df.sort_values("time")["R_obs"].iloc[0])
    sol = solve_ivp(rhs, (t_arr[0], t_arr[-1]), [R0],
                    t_eval=t_arr, method="RK45", rtol=1e-4, atol=1e-6)
    return sol.y[0] if sol.success else np.full_like(t_arr, np.nan, dtype=float)


# ------------------------------------------------------------------
# 诊断图生成
# ------------------------------------------------------------------

def plot_gof(pred_df, prefix, fig_dir):
    """观测 vs 预测散点图 + 按 subject 上色 + residuals vs time。"""
    lims = [pred_df["R_obs"].min() * 0.95, pred_df["R_obs"].max() * 1.05]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.scatter(pred_df["R_pred"], pred_df["R_obs"], s=18, alpha=0.55, color="steelblue")
    ax.plot(lims, lims, "r--", lw=1.5, label="identity")
    ax.set_xlabel("Predicted R"); ax.set_ylabel("Observed R")
    ax.set_title(f"GOF [{prefix}]"); ax.legend(); ax.grid(alpha=0.2)

    ax = axes[1]
    sids = sorted(pred_df["sid"].unique())
    cmap = plt.get_cmap("tab20", len(sids))
    for i, sid in enumerate(sids):
        g = pred_df[pred_df["sid"] == sid]
        ax.scatter(g["R_pred"], g["R_obs"], color=cmap(i), s=14, alpha=0.8)
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Predicted R"); ax.set_ylabel("Observed R")
    ax.set_title(f"GOF by subject [{prefix}]"); ax.grid(alpha=0.2)

    ax = axes[2]
    ax.scatter(pred_df["time"], pred_df["residual"], s=14, alpha=0.55, color="tab:orange")
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set_xlabel("time"); ax.set_ylabel("Residual")
    ax.set_title(f"Residuals vs Time [{prefix}]"); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_gof.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_residual(res, prefix, fig_dir):
    """Normal QQ + 直方图。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    (osm, osr), (sl, ic, r) = stats.probplot(res, dist="norm")
    ax.scatter(osm, osr, s=18, alpha=0.55, color="steelblue")
    ax.plot(osm, sl * osm + ic, "r--", lw=1.5)
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"Normal QQ [{prefix}] (R={r:.3f})"); ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(res, bins=25, density=True, alpha=0.65, color="tab:orange", edgecolor="white")
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std()),
            "b-", lw=2, label=f"N({res.mean():.2f},{res.std():.2f})")
    ax.set_xlabel("Residual"); ax.set_ylabel("Density")
    ax.set_title(f"Residual Dist [{prefix}]"); ax.legend(); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_residual.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_vpc(df, theta_runs_raw, terms_list, ec50, gamma, bin_edges, prefix, fig_dir):
    """
    VPC：模型不确定性用 theta_runs 积分 R(t) 轨迹的代表性值；
    观测不确定性用分箱 percentile。

    bin_edges : list of float, 时间分箱边界
    """
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}"
                  for i in range(len(bin_edges) - 1)]
    df = df.copy()
    df["time_bin"] = pd.cut(df["time"], bins=bin_edges,
                            labels=bin_labels, include_lowest=True)

    # 用 r_mean 作为 dR/dt 评估的代表性 R 值（而非 R=0）
    r_mean = float(df["R_obs"].mean())

    tmb, slo, sme, shi = [], [], [], []
    for tb in bin_labels:
        tgrp = df[df["time_bin"] == tb]
        if len(tgrp) == 0:
            continue
        tmb.append(float(tgrp["time"].mean()))
        Cmean = float(tgrp["C_obs"].mean())
        preds = []
        for th in theta_runs_raw:
            ti = {t: i for i, t in enumerate(terms_list)}
            feats = build_pd_features(r_mean, Cmean, ec50, gamma)
            preds.append(sum(th[ti[t_]] * feats[t_] for t_ in terms_list))
        slo.append(np.percentile(preds, 5))
        sme.append(np.percentile(preds, 50))
        shi.append(np.percentile(preds, 95))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(tmb, slo, shi, color="lightcoral", alpha=0.25,
                    label="Model 5-95% PI")
    ax.plot(tmb, sme, "r-", lw=2, label="Model median")
    ax.plot(tmb, slo, "r--", lw=1); ax.plot(tmb, shi, "r--", lw=1)

    for p, col, al in [(5, "skyblue", 0.3),
                        (50, "steelblue", 0.5),
                        (95, "skyblue", 0.3)]:
        lo, hi, tm = [], [], []
        for _, grp in df.groupby("time_bin"):
            tm.append(float(grp["time"].mean()))
            lo.append(np.percentile(grp["R_obs"], 50 - p / 2))
            hi.append(np.percentile(grp["R_obs"], 50 + p / 2))
        ax.fill_between(tm, lo, hi, color=col, alpha=al, label=f"Obs P{p}")

    tmo, mo = [], []
    for _, grp in df.groupby("time_bin"):
        tmo.append(float(grp["time"].mean()))
        mo.append(np.percentile(grp["R_obs"], 50))
    ax.plot(tmo, mo, "ko-", ms=5, lw=1.5, label="Obs median")

    ax.set_xlabel("Time"); ax.set_ylabel("R_obs")
    ax.set_title(f"VPC [{prefix}]"); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_vpc.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_bootstrap(boot_rows, theta_fit, terms_list, prefix, fig_dir):
    """Bootstrap 直方图 + CI。"""
    if len(boot_rows) == 0:
        k = len(terms_list)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
        if k == 1:
            axes = [axes]
        for i, term in enumerate(terms_list):
            axes[i].text(0.5, 0.5,
                         f"No convergence\n(theta={theta_fit[i]:.3f})",
                         ha="center", va="center",
                         transform=axes[i].transAxes, fontsize=10)
            axes[i].set_title(f"Bootstrap: {term} [{prefix}]")
            axes[i].grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{prefix}_bootstrap.png"),
                   dpi=160, bbox_inches="tight")
        plt.close()
        return 0

    boot_arr = np.array(boot_rows)
    bci = np.percentile(boot_arr, [2.5, 97.5], axis=0)
    bse = np.std(boot_arr, axis=0)
    k = boot_arr.shape[1]
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]
    for i, term in enumerate(terms_list):
        ax = axes[i]
        ax.hist(boot_arr[:, i], bins=30, density=True,
                alpha=0.65, color="steelblue", edgecolor="white")
        ax.axvline(theta_fit[i], color="red", lw=2,
                   label=f"Est={theta_fit[i]:.3f}")
        ax.axvline(bci[0, i], color="orange", ls="--", lw=1.5,
                   label=f"2.5%={bci[0,i]:.3f}")
        ax.axvline(bci[1, i], color="orange", ls="--", lw=1.5,
                   label=f"97.5%={bci[1,i]:.3f}")
        ax.set_title(f"Bootstrap: {term} [{prefix}]")
        ax.set_xlabel(f"θ[{term}]"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_bootstrap.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    return len(boot_rows)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def run_diagnostics(
    df: pd.DataFrame,
    df_cmp: pd.DataFrame,
    df_ms: pd.DataFrame,
    df_ms_all: pd.DataFrame,
    fig_dir: str,
    n_bootstrap: int = 200,
    seed_bootstrap: int = 20260420,
    bin_edges: list = None,
    boot_max_nfev: int = 800,
    fit_max_nfev: int = 2000,
):
    """
    遍历 df_cmp 中所有候选，生成 GOF / 残差 / VPC / Bootstrap 图。

    Parameters
    ----------
    df : pd.DataFrame  [sid, time, C_obs, R_obs]
    df_cmp : pd.DataFrame  NLME 结果，含 terms, BIC, AIC 列
    df_ms : pd.DataFrame  multistart_summary，含 theta_hat_json, ec50_hat, gamma_hat
    df_ms_all : pd.DataFrame  multistart_all_runs，含所有重启的 theta
    fig_dir : str
    n_bootstrap : int
    seed_bootstrap : int
    bin_edges : list of float  VPC 时间分箱，默认 [0, 1, 4, 8, 24]
    boot_max_nfev : int
    fit_max_nfev : int

    Returns
    -------
    dict  {rank: {"terms", "k", "BIC", "AIC", "n_boot", "theta_fit", ...}}
    """
    if bin_edges is None:
        bin_edges = [0.0, 1.0, 4.0, 8.0, 24.0]

    # 构建 terms -> (theta_norm, r_mean, r_std, ec50, gamma) 映射
    # r_mean/r_std 从 df_ms 同一 terms 的行中取（若不存在则估计）
    _build_normalization = lambda df_ref, terms_str: (
        float(df_ref["R_obs"].mean()),
        float(df_ref["R_obs"].std() + 1e-8)
    )

    theta_meta = {}  # terms -> {theta_norm, r_mean, r_std, ec50, gamma}
    for _, row in df_ms.iterrows():
        ts = row["terms"]
        if ts in theta_meta:
            continue
        theta_meta[ts] = {
            "theta_norm": json.loads(row["theta_hat_json"]),
            "ec50": float(row["ec50_hat"]),
            "gamma": float(row["gamma_hat"]),
        }
        # r_mean/r_std 用训练数据估计
        r_mean = float(df["R_obs"].mean())
        r_std = float(df["R_obs"].std() + 1e-8)
        theta_meta[ts]["r_mean"] = r_mean
        theta_meta[ts]["r_std"] = r_std

    results = {}
    rng_boot = np.random.default_rng(seed_bootstrap)

    for rank, (_, crow) in enumerate(df_cmp.iterrows(), 1):
        terms_str = crow["terms"]
        terms_list = [t.strip() for t in terms_str.split(" + ")]
        k = len(terms_list)
        bic_val = crow.get("BIC") or crow.get("bic") or crow.get("BIC_val")
        aic_val = crow.get("AIC") or crow.get("aic") or crow.get("AIC_val")
        prefix = f"m{rank}"

        meta = theta_meta.get(terms_str)
        if meta is None:
            print(f"  [Rank {rank}] {terms_str}: no theta in ms_sum, skip.")
            continue

        theta_norm = meta["theta_norm"]
        r_mean = meta["r_mean"]
        r_std = meta["r_std"]
        ec50 = meta["ec50"]
        gamma = meta["gamma"]

        # 1. 反归一化 + refit（原始尺度）
        theta_init_raw = denormalize_theta(theta_norm, r_mean, r_std)
        theta_fit, fit_ok = fit_theta_on_original_scale(
            terms_list, ec50, gamma, df, theta_init_raw, max_nfev=fit_max_nfev)

        # 2. ODE 预测
        rows = []
        for sid, grp in df.groupby("sid"):
            r_pred = predict_R_per_subject(grp, theta_fit, ec50, gamma, terms_list)
            for j, (_, rr) in enumerate(grp.sort_values("time").iterrows()):
                rows.append({
                    "sid": sid, "time": rr["time"],
                    "R_pred": r_pred[j], "R_obs": rr["R_obs"]
                })
        pred_df = pd.DataFrame(rows)
        pred_df["residual"] = pred_df["R_obs"] - pred_df["R_pred"]

        # 3. GOF + 残差图
        plot_gof(pred_df, prefix, fig_dir)
        res = pred_df["residual"].dropna()
        plot_residual(res, prefix, fig_dir)

        # 4. VPC：收集同 terms 所有重启的 theta（反归一化后）
        theta_runs_raw = []
        for _, r2 in df_ms_all.iterrows():
            if r2["terms"] != terms_str:
                continue
            try:
                th_n = json.loads(r2["theta_hat_json"])
                rm = float(r2.get("r_mean", r_mean))
                rs = float(r2.get("r_std", r_std))
                theta_runs_raw.append(denormalize_theta(th_n, rm, rs))
            except Exception:
                pass
        if not theta_runs_raw:
            theta_runs_raw = [theta_fit]

        plot_vpc(df, theta_runs_raw, terms_list, ec50, gamma,
                 bin_edges, prefix, fig_dir)

        # 5. Bootstrap（原始尺度残差）
        boot_rows, n_ok = [], 0
        for b in range(1, n_bootstrap + 1):
            bsids = rng_boot.choice(
                sorted(df["sid"].unique()),
                size=len(df["sid"].unique()), replace=True)
            bdf = pd.concat([df[df["sid"] == s] for s in bsids], ignore_index=True)

            def bresid_raw(th):
                pa = np.full(bdf.shape[0], np.nan)
                idx = 0
                for sid, grp in bdf.groupby("sid"):
                    rp = predict_R_per_subject(grp, th, ec50, gamma, terms_list)
                    for j, (_, rr) in enumerate(grp.sort_values("time").iterrows()):
                        pa[idx] = rp[j]; idx += 1
                if np.any(~np.isfinite(pa)):
                    return np.ones(bdf.shape[0]) * 1e3
                return pa - bdf.sort_values(["sid", "time"])["R_obs"].values

            try:
                out = least_squares(bresid_raw, theta_fit, method="trf",
                                    bounds=([-50] * k, [50] * k),
                                    max_nfev=boot_max_nfev,
                                    xtol=1e-5, ftol=1e-5)
                if out.success:
                    boot_rows.append(out.x.tolist())
                    n_ok += 1
            except Exception:
                pass

        plot_bootstrap(boot_rows, theta_fit, terms_list, prefix, fig_dir)

        # 6. Bootstrap 汇总表
        boot_table = (
            f"**Rank {rank}** | `{terms_str}` | k={k} | "
            f"BIC={bic_val:.4f} | AIC={aic_val:.4f} | "
            f"Bootstrap {n_ok}/{n_bootstrap}\n\n"
        )
        if n_ok > 0:
            boot_arr = np.array(boot_rows)
            bci = np.percentile(boot_arr, [2.5, 97.5], axis=0)
            bse = np.std(boot_arr, axis=0)
            boot_table += (
                "| 参数 | 点估计 | 2.5% | 97.5% | SE |\n"
                "|---:|---:|---:|---:|---:|\n"
            )
            for i, term in enumerate(terms_list):
                boot_table += (
                    f"| {term} | {theta_fit[i]:.4f} | "
                    f"{bci[0, i]:.4f} | {bci[1, i]:.4f} | {bse[i]:.4f} |\n"
                )
        else:
            boot_table += "_Bootstrap无收敛，参数估计仅供参考_"

        results[rank] = {
            "terms": terms_str, "k": k,
            "BIC": bic_val, "AIC": aic_val,
            "n_boot": n_ok,
            "theta_fit": theta_fit.tolist(),
            "theta_init_raw": theta_init_raw.tolist(),
            "fit_ok": fit_ok,
            "gof": f"![{prefix} GOF](figures/{prefix}_gof.png)",
            "residual": f"![{prefix} Residual](figures/{prefix}_residual.png)",
            "vpc": f"![{prefix} VPC](figures/{prefix}_vpc.png)",
            "bootstrap": boot_table,
        }
        print(f"  [Rank {rank}] {terms_str} done. "
              f"theta_fit={theta_fit.round(3).tolist()}. "
              f"Boot={n_ok}/{n_bootstrap}")

    return results