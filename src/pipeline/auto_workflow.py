import json
import os
import time
from typing import Optional

import pandas as pd

from src.configs.defaults import DEFAULTS, TMDD_DISCOVERY_DEFAULTS, TMDD_NLME_DEFAULTS
from src.pipeline.baseline_decomposition import fit_zero_dose_baseline, apply_zero_dose_baseline_correction
from src.pipeline.discovery import run_single_discovery
from src.pipeline.export_for_nlme import export_nlme_inputs
from src.pipeline.hidden_state_fast_confirm import confirm_hidden_states_fast
from src.pipeline.hidden_state_confirmation import confirm_top_hidden_mechanisms
from src.pipeline.mechanism_hints import (
    build_fixed_surrogate_protocol_evidence,
    build_residual_hidden_evidence,
    confirm_mechanism_presence,
    merge_residual_hidden_evidence,
    score_hidden_mechanisms,
)
from src.pipeline.multistart_refit import run_multistart_refit
from src.pipeline.pk_imputation import impute_pk_from_known_model
from src.pipeline.run_simbiology_validation import run_simbiology_diagnostics, run_simbiology_validation
from src.agent.llm_interface import build_llm_agent_payload


def _init_run_workspace(project_root=".", run_name=None):
    project_root = os.path.abspath(project_root)
    run_id = run_name or "run"
    run_dir = os.path.join(project_root, "reports", run_id)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return {
        "run_id": run_id,
        "paths": {
            "run_dir": run_dir,
            "artifacts_dir": artifacts_dir,
            "report_md": os.path.join(run_dir, "report.md"),
        },
    }


def _normalize_terms_for_md(terms):
    if isinstance(terms, (list, tuple)) and terms and isinstance(terms[0], (list, tuple)):
        parts = []
        for i, eq_terms in enumerate(terms, 1):
            parts.append(f"Eq{i}: {' + '.join(str(x) for x in eq_terms)}")
        return " || ".join(parts)
    if isinstance(terms, (list, tuple)):
        return " + ".join(str(x) for x in terms)
    return str(terms)


def _to_top_results(discovery_results):
    ec50 = float(discovery_results["ec50_hat"])
    gamma = float(discovery_results["gamma_hat"])
    cands = []
    for i, cand in enumerate(discovery_results["top_results"], 1):
        cands.append(
            {
                "rank": i,
                "terms": cand["terms"],
                "k": int(cand["k"]),
                "score": float(cand["score"]),
                "mse_val": float(cand["mse_val"]),
                "mse_train": float(cand["mse_train"]),
                "ec50_hat": ec50,
                "gamma_hat": gamma,
            }
        )
    return cands


def _inject_multistart_theta(top_results, df_ms_best: pd.DataFrame):
    if df_ms_best is None or df_ms_best.empty:
        return top_results
    by_rank = {}
    for _, row in df_ms_best.iterrows():
        try:
            rank = int(row["rank"])
            theta = json.loads(row["theta_hat_json"])
            if isinstance(theta, list):
                by_rank[rank] = theta
        except Exception:
            continue
    merged = []
    for cand in top_results:
        item = dict(cand)
        rank = int(item.get("rank", 0))
        if rank in by_rank:
            item["theta_hat"] = by_rank[rank]
            item["init_params"] = by_rank[rank]
        merged.append(item)
    return merged


def _rerank_top_results_by_multistart(top_results, df_ms_best: pd.DataFrame):
    if df_ms_best is None or df_ms_best.empty:
        return top_results
    by_rank = {int(c.get("rank", i + 1)): dict(c) for i, c in enumerate(top_results)}
    ordered = []
    seen = set()
    if "rank" in df_ms_best.columns:
        for _, row in df_ms_best.iterrows():
            try:
                rk = int(row["rank"])
            except Exception:
                continue
            if rk in by_rank and rk not in seen:
                ordered.append(by_rank[rk])
                seen.add(rk)
    for i, c in enumerate(top_results, 1):
        rk = int(c.get("rank", i))
        if rk not in seen:
            ordered.append(dict(c))
            seen.add(rk)

    reranked = []
    for i, c in enumerate(ordered, 1):
        item = dict(c)
        item["discovery_rank"] = int(c.get("rank", i))
        item["rank"] = i
        reranked.append(item)
    return reranked


def _flatten_terms_for_gate(raw_terms):
    if isinstance(raw_terms, (list, tuple)) and raw_terms and isinstance(raw_terms[0], (list, tuple)):
        flat = []
        for eq in raw_terms:
            flat.extend([str(x) for x in eq])
        return flat
    if isinstance(raw_terms, (list, tuple)):
        return [str(x) for x in raw_terms]
    return [str(raw_terms)]


def _use_direct_hill_constraints_from_terms(top_results) -> bool:
    # Blind-safe gate: inspect discovered terms only (no pd_data_label usage).
    for cand in top_results:
        terms = _flatten_terms_for_gate(cand.get("terms", []))
        norm = [str(t).replace(" ", "").lower() for t in terms]
        has_direct_nonlinear_exposure = any(("hill(" in t) or ("emax(" in t) for t in norm)
        has_tmdd_state = any(("cpr" in t) or ("ct" in t) or ("c-ct" in t) for t in norm)
        if has_direct_nonlinear_exposure and (not has_tmdd_state):
            return True
    return False


def _estimate_struct_theta_abs_bound(df: pd.DataFrame, cfg: dict, enabled: bool) -> float:
    base = float(cfg.get("nlme_struct_theta_abs_bound", 30.0))
    if not enabled or "R_obs" not in df.columns:
        return base
    r = pd.to_numeric(df["R_obs"], errors="coerce").dropna()
    if r.empty:
        return base
    max_abs = float(r.abs().max())
    dyn = float(r.max() - r.min())
    return float(min(max(base, 6.0 * max_abs, 12.0 * dyn, 30.0), 5000.0))


def _terms_to_key(terms) -> str:
    if isinstance(terms, (list, tuple)):
        return " + ".join(str(x) for x in terms)
    return str(terms)


def _rerank_top_results_by_nlme(top_results, simbio_df: pd.DataFrame):
    if simbio_df is None or simbio_df.empty or ("terms" not in simbio_df.columns):
        return top_results
    by_key = {}
    for c in top_results:
        key = _terms_to_key(c.get("terms", []))
        by_key[key] = dict(c)
    ranked = []
    used = set()
    cols = [c for c in ["BIC", "AIC", "RMSE"] if c in simbio_df.columns]
    if len(cols) == 0:
        return top_results
    df_ord = simbio_df.sort_values(cols, ascending=[True] * len(cols))
    for _, row in df_ord.iterrows():
        key = str(row.get("terms", ""))
        if key in by_key and key not in used:
            ranked.append(by_key[key])
            used.add(key)
    for c in top_results:
        key = _terms_to_key(c.get("terms", []))
        if key not in used:
            ranked.append(dict(c))
            used.add(key)
    out = []
    for i, c in enumerate(ranked, 1):
        item = dict(c)
        item["pre_nlme_rank"] = int(c.get("rank", i))
        item["rank"] = i
        out.append(item)
    return out


def _write_report(
    report_path: str,
    active_model: str,
    pd_data_label: Optional[str],
    pk_model_name: str,
    imputed_csv_path: str,
    top_results,
    simbio_df: pd.DataFrame,
    artifacts_dir: str,
    initial_confirmation_note: str,
    mechanism_hint: Optional[dict] = None,
    mechanism_confirmation: Optional[dict] = None,
    nlme_skipped: bool = False,
    timing: Optional[dict] = None,
    baseline_note: Optional[str] = None,
):
    timing = timing or {}
    lines = [
        "# PKPD 自动发现报告",
        "",
        f"- **PD发现模型**: `{active_model}`",
        f"- **PD数据标签**: `{pd_data_label or active_model}`",
        f"- **已知PK模型**: `{pk_model_name}`",
        f"- **插补后数据**: `{imputed_csv_path}`",
        f"- **Discovery耗时(秒)**: `{float(timing.get('discovery_seconds', 0.0)):.3f}`",
        f"- **NLME耗时(秒)**: `{float(timing.get('nlme_seconds', 0.0)):.3f}`",
        f"- **总耗时(秒)**: `{float(timing.get('total_workflow_seconds', 0.0)):.3f}`",
        "",
        "## 1. 回归出的模型结构（Top-K）",
        "",
        "| Rank | k | 结构 | BIC(val, python) | MSE(val) |",
        "|---:|---:|---|---:|---:|",
    ]
    if baseline_note:
        lines.insert(8, f"- **0剂量基线分解**: {baseline_note}")
    for cand in top_results:
        lines.append(
            f"| {int(cand['rank'])} | {int(cand['k'])} | {_normalize_terms_for_md(cand['terms'])} | "
            f"{float(cand['score']):.6f} | {float(cand['mse_val']):.6f} |"
        )

    if mechanism_hint:
        lines.extend(
            [
                "",
                "## 2. 隐状态机制增强提示",
                "",
                f"- **MES(0-100)**: `{float(mechanism_hint.get('overall_mes', 0.0)):.2f}`",
                f"- **Top1机制**: `{mechanism_hint.get('top_mechanism')}`",
                f"- **Top1概率**: `{float(mechanism_hint.get('top_probability', 0.0)):.4f}`",
                f"- **Top1-Top2 概率差**: `{float(mechanism_hint.get('top_probability_gap', 0.0)):.4f}`",
                f"- **是否机制歧义**: `{bool(mechanism_hint.get('ambiguous', False))}`",
                f"- **DDI提示证据级别**: `{mechanism_hint.get('interaction_evidence', 'none')}`",
                "- **默认判读策略**: `现象证据分型，不默认执行H0/H1隐状态确认`",
                "",
                "| 机制 | 最优模块 | 分数 | 概率 | ΔBIC(相对基线) | ΔMSE(相对基线) | 标记命中 |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in mechanism_hint.get("mechanisms", []):
            lines.append(
                f"| {row.get('mechanism')} | {row.get('best_combo')} | "
                f"{float(row.get('score', 0.0)):.2f} | {float(row.get('probability', 0.0)):.4f} | "
                f"{float(row.get('delta_bic', 0.0)):.6f} | {float(row.get('delta_mse', 0.0)):.6f} | "
                f"{', '.join(row.get('marker_hits', []))} |"
            )
        ool = mechanism_hint.get("out_of_library", {})
        lines.extend(
            [
                "",
                "### 2.1 库外机制风险与开放元件重组建议",
                "",
                f"- **是否可能存在库外机制**: `{bool(ool.get('likely', False))}`",
                f"- **库外机制置信度**: `{float(ool.get('confidence', 0.0)):.4f}`",
                f"- **机制库最大ΔBIC**: `{float(ool.get('max_delta_bic', 0.0)):.6f}`",
                f"- **机制库最大ΔMSE**: `{float(ool.get('max_delta_mse', 0.0)):.6f}`",
                f"- **判定依据**: {ool.get('rationale', '')}",
            ]
        )
        phenomenon_assessment = mechanism_hint.get("hidden_phenomena_assessment", {})
        if phenomenon_assessment:
            surrogate = phenomenon_assessment.get("observable_surrogate_assessment", {})
            lines.extend(
                [
                    "",
                    "### 2.2 隐空间现象证据分型",
                    "",
                    f"- **总体结论**: `{phenomenon_assessment.get('overall_verdict', '')}`",
                    f"- **Observable surrogate状态**: `{surrogate.get('status', '')}`",
                    f"- **Surrogate判定原因**: `{surrogate.get('reason', '')}`",
                    f"- **Surrogate是否足够**: `{bool(surrogate.get('sufficient', False))}`",
                    f"- **poor_fit_score**: `{float(surrogate.get('poor_fit_score', 0.0)):.4f}`",
                    f"- **residual_structure_score**: `{float(surrogate.get('residual_structure_score', 0.0)):.4f}`",
                    f"- **最强现象**: `{phenomenon_assessment.get('strongest_phenomenon', '')}`",
                    f"- **最强分数**: `{float(phenomenon_assessment.get('strongest_score', 0.0)):.2f}`",
                    f"- **是否使用H0/H1检验**: `{bool(phenomenon_assessment.get('uses_h0_h1_test', False))}`",
                    f"- **适用范围**: `{phenomenon_assessment.get('scope', '')}`",
                    f"- **解释**: {phenomenon_assessment.get('interpretation', '')}",
                    "",
                    "| 隐空间现象 | 证据等级 | 是否提示存在 | 分数 | 残差证据 | 模块证据 | 增益证据 | 关联机制 |",
                    "|---|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for row in phenomenon_assessment.get("phenomena", []):
                lines.append(
                    f"| {row.get('label', row.get('phenomenon', ''))} | "
                    f"{row.get('evidence_level', '')} | {bool(row.get('present', False))} | "
                    f"{float(row.get('score', 0.0)):.2f} | {float(row.get('residual_score', 0.0)):.2f} | "
                    f"{float(row.get('mechanism_score', 0.0)):.2f} | {float(row.get('gain_score', 0.0)):.2f} | "
                    f"{', '.join(row.get('linked_mechanisms', []))} |"
                )
        residual = mechanism_hint.get("residual_evidence", {})
        if residual:
            protocol = residual.get("surrogate_protocol", {})
            minimal = protocol.get("minimal", {}) if protocol else {}
            full = protocol.get("full", {}) if protocol else {}
            lines.extend(
                [
                    "",
                    "### 2.3 残差证据",
                    "",
                    f"- **残差证据来源**: `{residual.get('residual_source', 'na')}`",
                    f"- **Surrogate协议版本**: `{protocol.get('protocol_version', '')}`",
                    f"- **Minimal surrogate terms**: `{', '.join(minimal.get('terms', []) or [])}`",
                    f"- **Full surrogate terms**: `{', '.join(full.get('terms', []) or [])}`",
                    f"- **Minimal MSE**: `{float(minimal.get('mse', 0.0) or 0.0):.6f}`",
                    f"- **Full MSE**: `{float(full.get('mse', 0.0) or 0.0):.6f}`",
                    f"- **Residual absorption(minimal→full)**: `{float(protocol.get('residual_absorption_score', 0.0) or 0.0):.4f}`",
                    f"- **hysteresis_score**: `{float(residual.get('hysteresis_score', 0.0)):.4f}`",
                    f"- **zero_dose_trend_score**: `{float(residual.get('zero_dose_trend_score', 0.0)):.4f}`",
                    f"- **early_late_drift_score**: `{float(residual.get('early_late_drift_score', 0.0)):.4f}`",
                    f"- **late_overprediction_score**: `{float(residual.get('late_overprediction_score', 0.0)):.4f}`",
                    f"- **poor_fit_score**: `{float(residual.get('poor_fit_score', 0.0)):.4f}`",
                ]
            )
        suggestions = ool.get("suggested_open_component_equations", [])
        if suggestions:
            lines.extend(
                [
                    "",
                    "| 建议序号 | 开放元件候选方程 |",
                    "|---:|---|",
                ]
            )
            for i, item in enumerate(suggestions, 1):
                lines.append(f"| {i} | {item.get('equation', '')} |")
        if mechanism_confirmation and mechanism_confirmation.get("type") == "top_hidden_confirmation":
            lines.extend(
                [
                    "",
                    "### 2.4 Top机制确认拟合（仅显式启用时）",
                    "",
                    f"- **隐状态最终判定**: `{mechanism_confirmation.get('hidden_state_verdict', '')}`",
                    f"- **判定原因**: `{mechanism_confirmation.get('verdict_reason', '')}`",
                    f"- **推荐确认动作**: `{mechanism_confirmation.get('recommended_confirmation', '')}`",
                    f"- **确认阈值 ΔBIC**: `{float(mechanism_confirmation.get('thresholds', {}).get('delta_bic', 0.0)):.4f}`",
                    f"- **确认阈值 ΔMSE**: `{float(mechanism_confirmation.get('thresholds', {}).get('delta_mse', 0.0)):.4f}`",
                    f"- **BIC差值门限 T=BIC_obs-BIC_hidden**: `{float(mechanism_confirmation.get('thresholds', {}).get('t_bic_gate', 0.0)):.4f}`",
                    f"- **Bootstrap alpha**: `{float(mechanism_confirmation.get('thresholds', {}).get('bootstrap_alpha', 0.0)):.4f}`",
                    "",
                    "| 机制 | 是否支持 | 最优确认模块 | T(BIC差) | p值 | R方程隐状态耦合 | ΔBIC | ΔMSE | 原因 |",
                    "|---|---|---|---:|---:|---|---:|---:|---|",
                ]
            )
            for row in mechanism_confirmation.get("confirmations", []):
                p_value = row.get("bootstrap_p_value", None)
                p_display = "" if p_value is None else f"{float(p_value):.4f}"
                lines.append(
                    f"| {row.get('mechanism')} | {bool(row.get('supported', False))} | "
                    f"{row.get('best_combo', '')} | "
                    f"{float(row.get('t_bic', 0.0)):.6f} | {p_display} | "
                    f"{bool(row.get('valid_hidden_r_coupling', False))} | "
                    f"{float(row.get('delta_bic', 0.0)):.6f} | {float(row.get('delta_mse', 0.0)):.6f} | "
                    f"{row.get('reason', '')} |"
                )
        elif mechanism_confirmation:
            lines.extend(
                [
                    "",
                    "### 2.4 指定机制判别",
                    "",
                    f"- **指定机制**: `{mechanism_confirmation.get('mechanism')}`",
                    f"- **是否支持该机制**: `{bool(mechanism_confirmation.get('supported', False))}`",
                    f"- **判别原因**: `{mechanism_confirmation.get('reason')}`",
                    f"- **该机制概率**: `{float(mechanism_confirmation.get('probability', 0.0)):.4f}`",
                    f"- **该机制分数**: `{float(mechanism_confirmation.get('score', 0.0)):.2f}`",
                ]
            )

    lines.extend(
        [
            "",
            "## 3. 初值确认",
            "",
            initial_confirmation_note,
            "",
            "## 4. NLME 拟合值与拟合优度",
            "",
        ]
    )
    if nlme_skipped:
        lines.append("已按机制判别模式跳过 NLME 与诊断。")

    display_cols = ["rank", "terms", "converged", "AIC", "BIC", "RMSE", "SSE", "message"]
    cols = [c for c in display_cols if c in simbio_df.columns]
    if cols:
        simbio_display = simbio_df
        sort_cols = [c for c in ["BIC", "AIC", "RMSE"] if c in simbio_display.columns]
        if sort_cols:
            simbio_display = simbio_display.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        lines.extend(
            [
                "|" + "|".join(cols) + "|",
                "|" + "|".join(["---"] * len(cols)) + "|",
            ]
        )
        for _, row in simbio_display.iterrows():
            vals = [str(row[c]) for c in cols]
            lines.append("|" + "|".join(vals) + "|")

    lines.extend(
        [
            "",
            "## 5. 诊断图与工件",
            "",
            f"- NLME 目录: `{os.path.join(artifacts_dir, 'nlme')}`",
            f"- 诊断图目录: `{os.path.join(artifacts_dir, 'figures')}`",
            f"- MATLAB 结果: `{os.path.join(artifacts_dir, 'nlme', 'simbiology_results.csv')}`",
            "",
        ]
    )
    if mechanism_hint:
        lines.insert(-1, f"- 机制提示结果: `{os.path.join(artifacts_dir, 'mechanism_hinting.json')}`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_discovery_workflow(
    input_csv: str,
    active_model: str,
    pk_model_name: str,
    out_root: str,
    pd_data_label: Optional[str] = None,
    pk_route: str = None,
    pk_compartments: int = None,
    run_name: Optional[str] = None,
    device: Optional[str] = None,
    use_population_mean_for_nlme: bool = False,
    run_initial_confirmation: bool = True,
    nlme_mode: Optional[str] = None,
    nlme_multistart_on_fail: Optional[bool] = None,
    skip_bootstrap: bool = True,
    discovery_overrides: Optional[dict] = None,
    enable_mechanism_hints: bool = False,
    specified_mechanism: Optional[str] = None,
    mechanism_confirm_only: bool = False,
    matlab_engine_reuse: bool = False,
):
    workflow_t0 = time.perf_counter()
    manifest = _init_run_workspace(project_root=out_root, run_name=run_name)
    paths = manifest["paths"]
    artifacts_dir = paths["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df_imputed = impute_pk_from_known_model(
        df,
        model_name=pk_model_name,
        pk_route=pk_route,
        pk_compartments=pk_compartments,
        fit_subject_scale=True,
    )
    imputed_csv_path = os.path.join(artifacts_dir, "pkpd_long_imputed.csv")
    base_cols = ["sid", "time", "C_obs", "R_obs"]
    extra_signal_cols = [c for c in ["Ct_obs", "C_int_obs"] if c in df_imputed.columns]
    discovery_cols = base_cols + extra_signal_cols
    df_imputed[discovery_cols].to_csv(imputed_csv_path, index=False)

    cfg = dict(DEFAULTS)
    if active_model in {"TMDD_BASE", "ANTIBODY_PKPD"}:
        cfg.update(TMDD_DISCOVERY_DEFAULTS)
    if discovery_overrides:
        cfg.update(discovery_overrides)

    baseline_note = "未启用。"
    baseline_model = {"enabled": False, "reason": "disabled"}
    df_discovery = df_imputed.copy()
    nlme_df = df_imputed.copy()
    enable_baseline_decomp = bool(
        cfg.get(
            "enable_zero_dose_baseline_decomposition",
            active_model not in {"TMDD_BASE", "ANTIBODY_PKPD"},
        )
    )
    if enable_baseline_decomp and active_model not in {"TMDD_BASE", "ANTIBODY_PKPD"}:
        baseline_candidate_terms = tuple(cfg.get("baseline_candidate_terms", ["1", "cos24", "sin24"]))
        baseline_model = fit_zero_dose_baseline(
            df_imputed[base_cols],
            c_zero_eps=float(cfg.get("baseline_zero_eps", 1e-8)),
            candidate_terms=baseline_candidate_terms,
        )
        baseline_json = os.path.join(artifacts_dir, "baseline_decomposition.json")
        if baseline_model.get("enabled", False):
            df_discovery = apply_zero_dose_baseline_correction(df_imputed, baseline_model)
            corr_delta = (df_discovery["R_obs_basecorr"].astype(float) - df_imputed["R_obs"].astype(float)).abs()
            baseline_model["max_abs_correction"] = float(corr_delta.max()) if len(corr_delta) else 0.0
            baseline_model["n_corrected_points"] = int((corr_delta > 1e-12).sum())
            r_span = float(df_imputed["R_obs"].max() - df_imputed["R_obs"].min()) if len(df_imputed) else 0.0
            ratio = (baseline_model["max_abs_correction"] / r_span) if r_span > 1e-12 else 0.0
            baseline_model["correction_ratio"] = float(ratio)
            max_ratio = float(cfg.get("baseline_max_correction_ratio", 0.25))
            baseline_model["max_correction_ratio_allowed"] = max_ratio
            if ratio > max_ratio:
                baseline_model["enabled"] = False
                baseline_model["reason"] = "correction_ratio_exceeds_guard"
                baseline_model["n_corrected_points"] = 0
                baseline_note = (
                    f"已尝试但禁用（correction_ratio={ratio:.3f} > {max_ratio:.3f}，"
                    f"terms={baseline_model.get('terms', [])}）。"
                )
            else:
                df_discovery["R_obs"] = df_discovery["R_obs_basecorr"].astype(float)
                nlme_df = df_discovery.copy()
                baseline_note = (
                    f"已启用（terms={baseline_model.get('terms', [])}, "
                    f"zero_sids={baseline_model.get('n_zero_subjects', 0)}/{baseline_model.get('n_subjects', 0)}, "
                    f"corrected_points={baseline_model.get('n_corrected_points', 0)}）。"
                )
            discovery_csv_path = os.path.join(artifacts_dir, "pkpd_long_for_discovery.csv")
            save_cols = discovery_cols + ["R_baseline_hat", "R_baseline_drift", "R_obs_basecorr"]
            df_discovery[save_cols].to_csv(discovery_csv_path, index=False)
        else:
            baseline_note = f"已尝试但未启用（{baseline_model.get('reason', 'unknown')}）。"
        with open(baseline_json, "w", encoding="utf-8") as f:
            json.dump(baseline_model, f, ensure_ascii=False, indent=2)

    discovery_t0 = time.perf_counter()
    discovery = run_single_discovery(
        pop_data=df_discovery[discovery_cols],
        active_model=active_model,
        config=cfg,
        device=device,
        module_combo=None,
    )
    discovery_seconds = float(time.perf_counter() - discovery_t0)
    top_results = _to_top_results(discovery)
    mechanism_hint = None
    mechanism_hint_path = ""
    mechanism_confirmation = None
    mechanism_confirmation_path = ""
    has_cint_signal = ("C_int_obs" in df_imputed.columns)
    if has_cint_signal:
        cint_sum = float(pd.to_numeric(df_imputed["C_int_obs"], errors="coerce").fillna(0.0).abs().sum())
        has_cint_signal = bool(cint_sum > 0.0)
    has_coadmin_signal = any(
        kw in col.lower()
        for col in df_imputed.columns
        for kw in ["coadmin", "concom", "comed", "drug2", "co_drug"]
    )
    interaction_evidence = "cint" if has_cint_signal else ("coadmin" if has_coadmin_signal else "none")
    should_hint = (enable_mechanism_hints or bool(specified_mechanism)) and active_model not in {"TMDD_BASE", "ANTIBODY_PKPD"}
    hinting_mode = str(cfg.get("hidden_hinting_mode", "residual_only") or "residual_only").lower()
    if should_hint and hinting_mode != "residual_only":
        try:
            hint_cfg = dict(cfg)
            hint_cfg["n_epochs_warmup"] = min(int(hint_cfg.get("n_epochs_warmup", 1800)), 600)
            hint_cfg["n_epochs_prune"] = min(int(hint_cfg.get("n_epochs_prune", 700)), 250)
            hint_cfg["max_prune_rounds"] = min(int(hint_cfg.get("max_prune_rounds", 10)), 5)
            hint_cfg["candidate_refit_epochs"] = min(int(hint_cfg.get("candidate_refit_epochs", 400)), 180)
            mechanism_hint = score_hidden_mechanisms(
                pop_data=df_discovery[discovery_cols],
                active_model=active_model,
                baseline_discovery=discovery,
                config=hint_cfg,
                device=device,
                interaction_evidence=interaction_evidence,
            )
        except Exception:
            mechanism_hint = None
    initial_confirmation_note = "已跳过（当前配置未启用）。"
    nlme_skipped = bool(mechanism_confirm_only and bool(specified_mechanism))
    df_ms_best = pd.DataFrame()

    if run_initial_confirmation and not nlme_skipped:
        is_direct_sigmax_case = _use_direct_hill_constraints_from_terms(top_results)
        topk_payload = {
            "ec50_hat": float(discovery["ec50_hat"]),
            "gamma_hat": float(discovery["gamma_hat"]),
            "candidates": [{"rank": c["rank"], "terms": c["terms"], "k": c["k"]} for c in top_results],
        }
        try:
            df_ms_all, df_ms_best = run_multistart_refit(
                df_discovery[base_cols],
                topk_payload,
                direct_hill_constraints=is_direct_sigmax_case,
                lambda_stability=float(cfg.get("multistart_lambda_stability", 1.0)),
                lambda_temporal_shape=float(cfg.get("multistart_lambda_temporal", 3.0)),
                lambda_late_overpredict=float(cfg.get("multistart_lambda_late_overpredict", 10.0)),
                lambda_identifiability=float(cfg.get("multistart_lambda_identifiability", 3.0)),
                lambda_boundary_proximity=float(cfg.get("multistart_lambda_boundary", 6.0)),
                lambda_nonconverged=float(cfg.get("multistart_lambda_nonconverged", 120.0)),
                lambda_train_residual_bias=float(cfg.get("train_bias_penalty", 3.0)),
                lambda_train_temporal_bias=float(cfg.get("train_temporal_penalty", 2.0)),
                direct_theta_abs_bound=float(cfg.get("multistart_direct_theta_abs_bound", 400.0)),
                hill_gamma_grid=cfg.get("hill_gamma_grid", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            )
            ms_all_csv = os.path.join(artifacts_dir, "multistart_all_runs.csv")
            ms_best_csv = os.path.join(artifacts_dir, "multistart_best.csv")
            df_ms_all.to_csv(ms_all_csv, index=False)
            df_ms_best.to_csv(ms_best_csv, index=False)
            top_results = _inject_multistart_theta(top_results, df_ms_best)
            top_results = _rerank_top_results_by_multistart(top_results, df_ms_best)
            initial_confirmation_note = f"已执行 multistart 初值确认，结果保存在 `{ms_best_csv}`。"
        except Exception as exc:
            initial_confirmation_note = f"初值确认跳过（当前候选不支持该步骤或执行失败）：`{exc}`"

    if should_hint:
        if bool(cfg.get("use_fixed_surrogate_protocol", True)):
            residual_evidence = build_fixed_surrogate_protocol_evidence(
                pop_data=df_discovery[base_cols],
                config=cfg,
                interaction_evidence=interaction_evidence,
            )
        else:
            residual_evidence = build_residual_hidden_evidence(
                pop_data=df_discovery[base_cols],
                baseline_discovery=discovery,
                multistart_best=df_ms_best,
                interaction_evidence=interaction_evidence,
            )
        mechanism_hint = merge_residual_hidden_evidence(mechanism_hint, residual_evidence, config=cfg)
    if should_hint and mechanism_hint is not None and bool(cfg.get("enable_hidden_state_confirmation", True)):
        try:
            if bool(cfg.get("use_fast_hidden_confirmation", True)):
                mechanism_confirmation = confirm_hidden_states_fast(
                    pop_data=df_discovery[discovery_cols],
                    baseline_discovery=discovery,
                    mechanism_hint=mechanism_hint,
                    config=cfg,
                )
            else:
                mechanism_confirmation = confirm_top_hidden_mechanisms(
                    pop_data=df_discovery[discovery_cols],
                    active_model=active_model,
                    baseline_discovery=discovery,
                    mechanism_hint=mechanism_hint,
                    config=cfg,
                    device=device,
                )
            if isinstance(mechanism_confirmation, dict):
                mechanism_hint["recommended_confirmation"] = mechanism_confirmation.get("recommended_confirmation", "")
        except Exception:
            mechanism_confirmation = None
    if mechanism_hint is not None and specified_mechanism:
        if mechanism_confirmation and mechanism_confirmation.get("type") == "top_hidden_confirmation":
            row = next(
                (
                    r for r in mechanism_confirmation.get("confirmations", [])
                    if str(r.get("mechanism", "")).strip() == str(specified_mechanism).strip()
                ),
                None,
            )
            if row is not None:
                mechanism_confirmation = {
                    "mechanism": specified_mechanism,
                    "supported": bool(row.get("supported", False)),
                    "reason": str(row.get("reason", "from_top_hidden_confirmation")),
                    "probability": float(row.get("screen_probability", 0.0)),
                    "score": float(row.get("screen_score", 0.0)),
                    "top_probability_gap": float(mechanism_hint.get("top_probability_gap", 0.0)),
                    "best_combo": row.get("best_combo", ""),
                    "delta_bic": float(row.get("delta_bic", 0.0)),
                    "delta_mse": float(row.get("delta_mse", 0.0)),
                }
            else:
                mechanism_confirmation = confirm_mechanism_presence(mechanism_hint, specified_mechanism)
        else:
            mechanism_confirmation = confirm_mechanism_presence(mechanism_hint, specified_mechanism)
    if mechanism_hint is not None:
        mechanism_hint_path = os.path.join(artifacts_dir, "mechanism_hinting.json")
        with open(mechanism_hint_path, "w", encoding="utf-8") as f:
            json.dump(mechanism_hint, f, ensure_ascii=False, indent=2)
    if mechanism_confirmation is not None:
        mechanism_confirmation_path = os.path.join(artifacts_dir, "hidden_state_confirmation.json")
        with open(mechanism_confirmation_path, "w", encoding="utf-8") as f:
            json.dump(mechanism_confirmation, f, ensure_ascii=False, indent=2)

    if nlme_mode is None:
        nlme_mode = TMDD_NLME_DEFAULTS["nlme_mode"] if active_model in {"TMDD_BASE", "ANTIBODY_PKPD"} else "screen"
    if nlme_multistart_on_fail is None:
        nlme_multistart_on_fail = (
            TMDD_NLME_DEFAULTS["nlme_multistart_on_fail"] if active_model in {"TMDD_BASE", "ANTIBODY_PKPD"} else True
        )

    nlme_dir = os.path.join(artifacts_dir, "nlme")
    simbio_df = pd.DataFrame()
    nlme_seconds = 0.0
    if not nlme_skipped:
        is_direct_sigmax_case = _use_direct_hill_constraints_from_terms(top_results)
        struct_theta_abs_bound = _estimate_struct_theta_abs_bound(nlme_df[base_cols], cfg, is_direct_sigmax_case)
        fit_hints = {
            "direct_sigemax_case": bool(is_direct_sigmax_case),
            "struct_theta_abs_bound": struct_theta_abs_bound,
            "direct_hill_bounds": {
                "ec50_lb": 0.5,
                "ec50_ub": 8.0,
                "gamma_lb": 0.5,
                "gamma_ub": 3.0,
            },
            "hill_gamma_grid": cfg.get("hill_gamma_grid", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        }
        nlme_t0 = time.perf_counter()
        os.makedirs(nlme_dir, exist_ok=True)
        simbio_df = run_simbiology_validation(
            pop_data=nlme_df[base_cols].values,
            top_results=top_results,
            project_root=".",
            out_dir=nlme_dir,
            use_population_mean=use_population_mean_for_nlme,
            nlme_mode=nlme_mode,
            nlme_multistart_on_fail=bool(nlme_multistart_on_fail),
            fit_hints=fit_hints,
            matlab_engine_reuse=bool(matlab_engine_reuse),
        )

        data_csv, topk_json = export_nlme_inputs(
            pop_data=nlme_df[base_cols].values,
            top_results=top_results,
            out_dir=nlme_dir,
            use_population_mean=use_population_mean_for_nlme,
            nlme_mode=nlme_mode,
            nlme_multistart_on_fail=bool(nlme_multistart_on_fail),
            fit_hints=fit_hints,
        )

        fig_dir = os.path.join(artifacts_dir, "figures")
        diag_ok = run_simbiology_diagnostics(
            data_csv=data_csv,
            simbio_csv=os.path.join(nlme_dir, "simbiology_results.csv"),
            topk_json=topk_json,
            fig_dir=fig_dir,
            project_root=".",
            skip_bootstrap=skip_bootstrap,
            matlab_engine_reuse=bool(matlab_engine_reuse),
        )
        if not diag_ok:
            diag_note_path = os.path.join(fig_dir, "diagnostics_note.txt")
            with open(diag_note_path, "w", encoding="utf-8") as f:
                f.write("Diagnostics plotting failed and was skipped. See diagnostics_error.txt for details.\n")
        nlme_seconds = float(time.perf_counter() - nlme_t0)
        top_results = _rerank_top_results_by_nlme(top_results, simbio_df)
    total_workflow_seconds = float(time.perf_counter() - workflow_t0)
    timing = {
        "discovery_seconds": discovery_seconds,
        "nlme_seconds": nlme_seconds,
        "total_workflow_seconds": total_workflow_seconds,
    }

    report_path = paths["report_md"]
    _write_report(
        report_path=report_path,
        active_model=active_model,
        pd_data_label=pd_data_label,
        pk_model_name=pk_model_name,
        imputed_csv_path=imputed_csv_path,
        top_results=top_results,
        simbio_df=simbio_df,
        artifacts_dir=artifacts_dir,
        initial_confirmation_note=initial_confirmation_note,
        mechanism_hint=mechanism_hint,
        mechanism_confirmation=mechanism_confirmation,
        nlme_skipped=nlme_skipped,
        timing=timing,
        baseline_note=baseline_note,
    )
    llm_payload = build_llm_agent_payload(
        active_model=active_model,
        pk_model_name=pk_model_name,
        discovery_top_results=top_results,
        mechanism_hint=mechanism_hint,
        mechanism_confirmation=mechanism_confirmation,
    )
    llm_payload_path = os.path.join(artifacts_dir, "llm_agent_payload.json")
    with open(llm_payload_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, ensure_ascii=False, indent=2)

    meta_path = os.path.join(artifacts_dir, "workflow_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "active_model": active_model,
                "pd_data_label": pd_data_label,
                "pk_model_name": pk_model_name,
                "pk_route": pk_route,
                "pk_compartments": (int(pk_compartments) if pk_compartments is not None else None),
                "input_csv": os.path.abspath(input_csv),
                "imputed_csv": imputed_csv_path,
                "baseline_decomposition": baseline_model,
                "report_md": report_path,
                "nlme_mode": nlme_mode,
                "nlme_multistart_on_fail": bool(nlme_multistart_on_fail),
                "skip_bootstrap": bool(skip_bootstrap),
                "enable_mechanism_hints": bool(enable_mechanism_hints),
                "discovery_profile": str(cfg.get("discovery_profile", "unknown")),
                "residual_hinting_enabled": bool(should_hint),
                "specified_mechanism": specified_mechanism,
                "mechanism_confirm_only": bool(mechanism_confirm_only),
                "matlab_engine_reuse": bool(matlab_engine_reuse),
                "llm_agent_payload_json": llm_payload_path,
                "mechanism_hint_json": mechanism_hint_path,
                "mechanism_confirmation_json": mechanism_confirmation_path,
                "timing": timing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "run_dir": paths["run_dir"],
        "report_md": report_path,
        "imputed_csv": imputed_csv_path,
        "nlme_csv": (os.path.join(nlme_dir, "simbiology_results.csv") if not nlme_skipped else ""),
        "meta_json": meta_path,
        "mechanism_confirmation": mechanism_confirmation,
        "mechanism_hint_json": mechanism_hint_path,
        "mechanism_confirmation_json": mechanism_confirmation_path,
        "llm_payload_json": llm_payload_path,
        "timing": timing,
    }
