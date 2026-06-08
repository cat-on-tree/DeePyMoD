import argparse
import json
import os

import numpy as np
import pandas as pd


def _read_csv(path, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _basic_success(df):
    if "repair_success" in df.columns:
        return _as_bool(df["repair_success"])
    delta = pd.to_numeric(df.get("proposed_delta_bic", np.nan), errors="coerce")
    gain = pd.to_numeric(df.get("proposed_rel_mse_gain", np.nan), errors="coerce")
    return (delta > 0.0) & (gain > 0.02)


def _repair_basic_success(df):
    delta = pd.to_numeric(df.get("delta_bic", np.nan), errors="coerce")
    gain = pd.to_numeric(df.get("rel_mse_gain", np.nan), errors="coerce")
    ok = _as_bool(df["ok"]) if "ok" in df.columns else pd.Series(True, index=df.index)
    return ok & (delta > 0.0) & (gain > 0.02)


def _write_table(lines, title, df, max_rows=20):
    lines.extend([f"## {title}", ""])
    if df.empty:
        lines.extend(["No rows available.", ""])
        return
    shown = df.head(max_rows)
    lines.append("|" + "|".join(str(c) for c in shown.columns) + "|")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(shown.columns) - 1)) + "|")
    for _, row in shown.iterrows():
        vals = []
        for val in row:
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("|" + "|".join(vals) + "|")
    lines.append("")


def _classifier_confusion(cases, guide):
    hidden = cases[cases["status"].eq("success") & ~_as_bool(cases["expected_h0"])].copy()
    if not guide.empty and {"true_mechanism", "predicted_mechanism"}.issubset(guide.columns):
        confusion = pd.crosstab(guide["true_mechanism"], guide["predicted_mechanism"])
    elif {"true_mechanism", "proposed_mechanism"}.issubset(hidden.columns):
        confusion = pd.crosstab(hidden["true_mechanism"], hidden["proposed_mechanism"])
    else:
        confusion = pd.DataFrame()
    return confusion


def _summary_by_mechanism(cases):
    hidden = cases[cases["status"].eq("success") & ~_as_bool(cases["expected_h0"])].copy()
    if hidden.empty:
        return pd.DataFrame()
    hidden["repair_success_basic"] = _basic_success(hidden)
    if "specificity_success" in hidden.columns:
        hidden["specificity_success_bool"] = _as_bool(hidden["specificity_success"])
    else:
        hidden["specificity_success_bool"] = False
    aggregations = {
        "n": ("case_id", "count"),
        "proposal_acc": ("proposal_matches_true", "mean"),
        "repair_success": ("repair_success_basic", "mean"),
        "specificity_success": ("specificity_success_bool", "mean"),
        "proposed_delta_bic": ("proposed_delta_bic", "mean"),
        "proposed_rel_mse_gain": ("proposed_rel_mse_gain", "mean"),
    }
    if "proposed_gate_supported" in hidden.columns:
        hidden["proposed_gate_supported_bool"] = _as_bool(hidden["proposed_gate_supported"])
        aggregations["gate_support"] = ("proposed_gate_supported_bool", "mean")
    if "proposed_gate_score" in hidden.columns:
        aggregations["proposed_gate_score"] = ("proposed_gate_score", "mean")
    if "proposed_gate_penalty" in hidden.columns:
        aggregations["proposed_gate_penalty"] = ("proposed_gate_penalty", "mean")
    if "best_repair_matches_true" in hidden.columns:
        aggregations["best_specificity_acc"] = ("best_repair_matches_true", "mean")
    if "best_bic_repair_matches_true" in hidden.columns:
        aggregations["best_bic_acc"] = ("best_bic_repair_matches_true", "mean")
    if "proposed_specificity_score" in hidden.columns:
        aggregations["proposed_specificity_score"] = ("proposed_specificity_score", "mean")
    return hidden.groupby("true_mechanism").agg(**aggregations).reset_index()


def _top_off_diagonal(repairs, top_n):
    if repairs.empty or not {"true_mechanism", "module", "delta_bic", "rel_mse_gain"}.issubset(repairs.columns):
        return pd.DataFrame()
    work = repairs.copy()
    work["repair_success_basic"] = _repair_basic_success(work)
    work = work[
        work["repair_success_basic"]
        & work["true_mechanism"].ne("observable")
        & work["true_mechanism"].ne(work["module"])
    ].copy()
    if work.empty:
        return work
    cols = ["case_id", "true_mechanism", "module", "delta_bic", "rel_mse_gain"]
    if "specificity_score" in work.columns:
        cols.append("specificity_score")
    return work.sort_values(["delta_bic", "rel_mse_gain"], ascending=False)[cols].head(top_n)


def _observable_false_positives(repairs):
    if repairs.empty or "true_mechanism" not in repairs.columns:
        return pd.DataFrame()
    obs = repairs[repairs["true_mechanism"].eq("observable")].copy()
    if obs.empty:
        return pd.DataFrame()
    obs["repair_success_basic"] = _repair_basic_success(obs)
    agg = obs.groupby("module").agg(
        n=("case_id", "count"),
        false_positive_rate=("repair_success_basic", "mean"),
        mean_delta_bic=("delta_bic", "mean"),
        mean_rel_mse_gain=("rel_mse_gain", "mean"),
    )
    if "specificity_score" in obs.columns:
        agg["mean_specificity_score"] = obs.groupby("module")["specificity_score"].mean()
    return agg.reset_index().sort_values(["false_positive_rate", "mean_delta_bic"], ascending=False)


def _module_specificity(repairs):
    if repairs.empty:
        return pd.DataFrame()
    work = repairs.copy()
    work["is_true"] = work["true_mechanism"].eq(work["module"])
    work["repair_success_basic"] = _repair_basic_success(work)
    hidden = work[work["true_mechanism"].ne("observable")].copy()
    if hidden.empty:
        return pd.DataFrame()
    rows = []
    for module, grp in hidden.groupby("module"):
        diag = grp[grp["is_true"]]
        off = grp[~grp["is_true"]]
        row = {
            "module": module,
            "true_gain": float(pd.to_numeric(diag["delta_bic"], errors="coerce").mean()) if not diag.empty else np.nan,
            "offdiag_gain": float(pd.to_numeric(off["delta_bic"], errors="coerce").mean()) if not off.empty else np.nan,
            "offdiag_success_rate": float(off["repair_success_basic"].mean()) if not off.empty else np.nan,
        }
        if "module_gate_score" in grp.columns:
            row["true_gate_score"] = float(pd.to_numeric(diag["module_gate_score"], errors="coerce").mean()) if not diag.empty else np.nan
            row["offdiag_gate_score"] = float(pd.to_numeric(off["module_gate_score"], errors="coerce").mean()) if not off.empty else np.nan
        if "module_gate_penalty" in grp.columns:
            row["true_gate_penalty"] = float(pd.to_numeric(diag["module_gate_penalty"], errors="coerce").mean()) if not diag.empty else np.nan
        if "specificity_score" in grp.columns:
            row["true_specificity_score"] = float(pd.to_numeric(diag["specificity_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if not diag.empty else np.nan
            row["offdiag_specificity_score"] = float(pd.to_numeric(off["specificity_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if not off.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["specificity_margin"] = out["true_gain"] - out["offdiag_gain"]
        out = out.sort_values("specificity_margin", ascending=True)
    return out


def _recommendations(summary, offdiag, obs_fp, module_spec):
    lines = ["## Suggested next edits", ""]
    if summary.empty and offdiag.empty and obs_fp.empty:
        lines.extend(["Insufficient repair artifacts for recommendations.", ""])
        return lines
    if not summary.empty:
        weak = summary.sort_values(["specificity_success", "repair_success"]).head(3)
        for _, row in weak.iterrows():
            lines.append(
                f"- `{row['true_mechanism']}`: low specificity/repair signal "
                f"(specificity_success={row.get('specificity_success', np.nan):.3f}, "
                f"repair_success={row.get('repair_success', np.nan):.3f}); inspect or add mechanism-specific atoms."
            )
        if "gate_support" in summary.columns:
            gate_weak = summary.sort_values("gate_support").iloc[0]
            lines.append(
                f"- Weakest residual evidence gate: `{gate_weak['true_mechanism']}` "
                f"(gate_support={float(gate_weak['gate_support']):.3f}, "
                f"gate_score={float(gate_weak.get('proposed_gate_score', np.nan)):.3f}); refine gate features or mechanism atoms."
            )
    if not offdiag.empty:
        top = offdiag.iloc[0]
        lines.append(
            f"- Strongest off-diagonal absorption: true `{top['true_mechanism']}` repaired by `{top['module']}` "
            f"(delta_bic={float(top['delta_bic']):.3f}); split or gate `{top['module']}` atoms."
        )
    if not obs_fp.empty:
        top_fp = obs_fp.iloc[0]
        if float(top_fp["false_positive_rate"]) > 0:
            lines.append(
                f"- Highest observable false-positive module: `{top_fp['module']}` "
                f"(rate={float(top_fp['false_positive_rate']):.3f}); increase penalty or require residual evidence gate."
            )
    if not module_spec.empty:
        bad = module_spec.iloc[0]
        lines.append(
            f"- Lowest specificity margin module: `{bad['module']}` "
            f"(margin={float(bad['specificity_margin']):.3f}); this is the first atom set to tighten."
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Analyze repair benchmark artifacts and write model-atom refinement diagnostics.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    result_dir = args.result_dir
    cases = _read_csv(os.path.join(result_dir, "repair_cases.csv"), required=True)
    repairs = _read_csv(os.path.join(result_dir, "repair_predictions_long.csv"), required=True)
    guide = _read_csv(os.path.join(result_dir, "guide_classifier_predictions.csv"), required=False)
    manifest_path = os.path.join(result_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    confusion = _classifier_confusion(cases, guide)
    summary = _summary_by_mechanism(cases)
    offdiag = _top_off_diagonal(repairs, int(args.top_n))
    obs_fp = _observable_false_positives(repairs)
    module_spec = _module_specificity(repairs)

    confusion.to_csv(os.path.join(result_dir, "repair_analysis_classifier_confusion.csv"))
    summary.to_csv(os.path.join(result_dir, "repair_analysis_by_mechanism.csv"), index=False)
    offdiag.to_csv(os.path.join(result_dir, "repair_analysis_top_off_diagonal.csv"), index=False)
    obs_fp.to_csv(os.path.join(result_dir, "repair_analysis_observable_false_positives.csv"), index=False)
    module_spec.to_csv(os.path.join(result_dir, "repair_analysis_module_specificity.csv"), index=False)

    lines = [
        "# Repair benchmark analysis",
        "",
        f"- result_dir: `{result_dir}`",
        f"- guide_mode: `{manifest.get('guide_mode', 'unknown')}`",
        f"- cases: {len(cases)}",
        f"- repairs: {len(repairs)}",
        "",
    ]
    lines.extend(_recommendations(summary, offdiag, obs_fp, module_spec))
    _write_table(lines, "Mechanism summary", summary, max_rows=50)
    if not confusion.empty:
        lines.extend(["## Classifier/proposal confusion", ""])
        lines.append(confusion.to_markdown())
        lines.append("")
    _write_table(lines, "Top off-diagonal repair absorptions", offdiag, max_rows=int(args.top_n))
    _write_table(lines, "Observable false-positive modules", obs_fp, max_rows=50)
    _write_table(lines, "Module specificity margins", module_spec, max_rows=50)

    report_path = os.path.join(result_dir, "repair_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(report_path)
    print(os.path.join(result_dir, "repair_analysis_by_mechanism.csv"))
    print(os.path.join(result_dir, "repair_analysis_top_off_diagonal.csv"))


if __name__ == "__main__":
    main()
