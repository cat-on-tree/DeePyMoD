import argparse
import concurrent.futures as cf
import json
import os
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.defaults import DEFAULTS
from src.data.simulate_pkpd import generate_population_data
from src.pipeline.mechanism_hints import build_fixed_surrogate_protocol_evidence


DEFAULT_MODELS = [
    "DIRECT_SIGEMAX",
    "IDR_INHIB_KIN_SIG",
    "TGI_BASIC",
    "BIOPHASE_EMAX",
    "BIOPHASE_SIGEMAX",
    "TRANSDUCTION_DELAY",
    "FEEDBACK_REGULATION",
    "CIRCADIAN_REGULATION",
    "DISEASE_PROGRESSION",
    "TOLERANCE_ADAPTATION",
    "PRECURSOR_POOL",
]

RESIDUAL_FEATURE_COLUMNS = [
    "mechanistic_mse",
    "empirical_mse",
    "minimal_to_mechanistic_gain",
    "mechanistic_to_empirical_gain",
    "poor_fit_score",
    "raw_residual_structure_score",
    "amplitude_weighted_structure_score",
    "hysteresis_score",
    "zero_dose_trend_score",
    "early_late_drift_score",
    "late_overprediction_score",
    "residual_bias_score",
    "temporal_shape_score",
    "dose_stratified_rmse_score",
    "response_peak_lag_score",
    "residual_peak_lag_score",
    "residual_loop_area_score",
    "response_loop_area_score",
    "residual_autocorr_score",
    "residual_sign_persistence_score",
    "late_residual_slope_score",
    "dose_residual_correlation_score",
    "positive_residual_fraction_mean",
    "positive_residual_fraction_sd",
    "residual_iqr_score",
]


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, float) and (not np.isfinite(obj)):
        return None
    return obj


def _label_for_model(model_name):
    if model_name in {"DIRECT_SIGEMAX", "IDR_INHIB_KIN_SIG", "TGI_BASIC"}:
        return "observable_sufficient"
    if model_name.startswith("BIOPHASE"):
        return "biophase_like"
    if model_name == "TRANSDUCTION_DELAY":
        return "transduction_like"
    if model_name == "FEEDBACK_REGULATION":
        return "feedback_like"
    if model_name == "CIRCADIAN_REGULATION":
        return "circadian_like"
    if model_name == "DISEASE_PROGRESSION":
        return "disease_like"
    if model_name == "TOLERANCE_ADAPTATION":
        return "tolerance_like"
    if model_name == "PRECURSOR_POOL":
        return "precursor_like"
    return "unknown"


def _flatten_case(model_name, seed, evidence):
    protocol = evidence.get("surrogate_protocol", {}) or {}
    h0_gate = evidence.get("h0_gate", {}) or {}
    minimal = protocol.get("minimal", {}) or {}
    mechanistic = protocol.get("mechanistic", {}) or {}
    empirical = protocol.get("empirical", protocol.get("full", {}) or {}) or {}
    label = _label_for_model(model_name)
    expected_h0 = bool(label == "observable_sufficient")
    h0_accepted = bool(h0_gate.get("accepted", False))
    row = {
        "pd_model": model_name,
        "label": label,
        "expected_h0": expected_h0,
        "seed": int(seed),
        "status": "success",
        "protocol_version": protocol.get("protocol_version", ""),
        "h0_accepted": h0_accepted,
        "h0_expected_correct": bool(h0_accepted == expected_h0),
        "h0_verdict": h0_gate.get("verdict", evidence.get("h0_verdict", "")),
        "residual_screen_verdict": evidence.get("residual_screen_verdict", ""),
        "h0_reason": h0_gate.get("reason", ""),
        "h0_mechanistic_poor_fit_score": float(h0_gate.get("mechanistic_poor_fit_score", evidence.get("poor_fit_score", 0.0)) or 0.0),
        "h0_mechanistic_structure_score": float(h0_gate.get("mechanistic_structure_score", evidence.get("amplitude_weighted_structure_score", 0.0)) or 0.0),
        "h0_mechanistic_to_empirical_gain": float(h0_gate.get("mechanistic_to_empirical_gain", protocol.get("mechanistic_to_empirical_gain", 0.0)) or 0.0),
        "minimal_mse": float(minimal.get("mse", 0.0) or 0.0),
        "minimal_rmse": float(minimal.get("rmse", 0.0) or 0.0),
        "mechanistic_mse": float(mechanistic.get("mse", 0.0) or 0.0),
        "mechanistic_rmse": float(mechanistic.get("rmse", 0.0) or 0.0),
        "empirical_mse": float(empirical.get("mse", 0.0) or 0.0),
        "empirical_rmse": float(empirical.get("rmse", 0.0) or 0.0),
        "full_mse": float(empirical.get("mse", 0.0) or 0.0),
        "full_rmse": float(empirical.get("rmse", 0.0) or 0.0),
        "residual_absorption_score": float(protocol.get("residual_absorption_score", 0.0) or 0.0),
        "minimal_to_mechanistic_gain": float(protocol.get("minimal_to_mechanistic_gain", 0.0) or 0.0),
        "mechanistic_to_empirical_gain": float(protocol.get("mechanistic_to_empirical_gain", 0.0) or 0.0),
        "minimal_to_full_mse_ratio": protocol.get("minimal_to_full_mse_ratio", None),
        "hysteresis_score": float(evidence.get("hysteresis_score", 0.0) or 0.0),
        "zero_dose_trend_score": float(evidence.get("zero_dose_trend_score", 0.0) or 0.0),
        "early_late_drift_score": float(evidence.get("early_late_drift_score", 0.0) or 0.0),
        "late_overprediction_score": float(evidence.get("late_overprediction_score", 0.0) or 0.0),
        "residual_bias_score": float(evidence.get("residual_bias_score", 0.0) or 0.0),
        "temporal_shape_score": float(evidence.get("temporal_shape_score", 0.0) or 0.0),
        "dose_stratified_rmse_score": float(evidence.get("dose_stratified_rmse_score", 0.0) or 0.0),
        "poor_fit_score": float(evidence.get("poor_fit_score", 0.0) or 0.0),
        "raw_residual_structure_score": float(evidence.get("raw_residual_structure_score", 0.0) or 0.0),
        "amplitude_weighted_structure_score": float(evidence.get("amplitude_weighted_structure_score", 0.0) or 0.0),
        "response_peak_lag_score": float(evidence.get("response_peak_lag_score", 0.0) or 0.0),
        "residual_peak_lag_score": float(evidence.get("residual_peak_lag_score", 0.0) or 0.0),
        "residual_loop_area_score": float(evidence.get("residual_loop_area_score", 0.0) or 0.0),
        "response_loop_area_score": float(evidence.get("response_loop_area_score", 0.0) or 0.0),
        "residual_autocorr_score": float(evidence.get("residual_autocorr_score", 0.0) or 0.0),
        "residual_sign_persistence_score": float(evidence.get("residual_sign_persistence_score", 0.0) or 0.0),
        "late_residual_slope_score": float(evidence.get("late_residual_slope_score", 0.0) or 0.0),
        "dose_residual_correlation_score": float(evidence.get("dose_residual_correlation_score", 0.0) or 0.0),
        "positive_residual_fraction_mean": float(evidence.get("positive_residual_fraction_mean", 0.0) or 0.0),
        "positive_residual_fraction_sd": float(evidence.get("positive_residual_fraction_sd", 0.0) or 0.0),
        "residual_iqr_score": float(evidence.get("residual_iqr_score", 0.0) or 0.0),
    }
    return row


def _run_one_case(args):
    model_name = args["model_name"]
    seed = int(args["seed"])
    try:
        pop_data, _, _ = generate_population_data(
            model_name=model_name,
            seed=seed,
            n_subjects=int(args["n_subjects"]),
            extra_pk_iiv_sigma=0.0,
            return_pk_scale=False,
            pk_route=args["pk_route"],
            pk_compartments=int(args["pk_compartments"]),
            dose_design_enabled=True,
            dose_levels=list(args["dose_levels"]),
            disable_iiv=bool(args["disable_iiv"]),
            disable_quality_guard=bool(args.get("disable_quality_guard", False)),
        )
        df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
        cfg = dict(DEFAULTS)
        evidence = build_fixed_surrogate_protocol_evidence(
            df,
            config=cfg,
            interaction_evidence="none",
        )
        if not evidence.get("available", False):
            return {
                "pd_model": model_name,
                "label": _label_for_model(model_name),
                "expected_h0": bool(_label_for_model(model_name) == "observable_sufficient"),
                "seed": int(seed),
                "status": "failed",
                "error": str(evidence.get("reason", "surrogate_evidence_unavailable")),
            }, {
                "pd_model": model_name,
                "seed": int(seed),
                "evidence": evidence,
            }
        row = _flatten_case(model_name, seed, evidence)
        return row, {
            "pd_model": model_name,
            "seed": int(seed),
            "evidence": evidence,
        }
    except Exception as exc:
        return {
            "pd_model": model_name,
            "label": _label_for_model(model_name),
            "expected_h0": bool(_label_for_model(model_name) == "observable_sufficient"),
            "seed": int(seed),
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, None


def _format_confusion_table(labels, cm):
    lines = ["|true\\pred|" + "|".join(str(x) for x in labels) + "|", "|" + "|".join(["---"] + ["---:"] * len(labels)) + "|"]
    for label, row in zip(labels, cm):
        lines.append("|" + str(label) + "|" + "|".join(str(int(x)) for x in row) + "|")
    return lines


def _evaluate_classifier(df, target_col, row_filter=None):
    ok = df[df["status"].eq("success")].copy()
    if row_filter is not None:
        ok = ok[row_filter(ok)].copy()
    cols = [c for c in RESIDUAL_FEATURE_COLUMNS if c in ok.columns]
    ok = ok.dropna(subset=cols + [target_col, "seed"])
    if len(ok) < 6 or ok[target_col].nunique() < 2 or ok["seed"].nunique() < 2:
        return None
    x_all = ok[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = ok[target_col].astype(str)
    preds = []
    truth = []
    seeds = []
    importances = pd.Series(0.0, index=cols)
    n_fits = 0
    for seed in sorted(ok["seed"].unique()):
        train = ok["seed"] != seed
        test = ok["seed"] == seed
        if y_all[train].nunique() < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                random_state=int(seed),
                class_weight="balanced",
            ),
        )
        clf.fit(x_all[train], y_all[train])
        pred = clf.predict(x_all[test])
        preds.extend(pred.tolist())
        truth.extend(y_all[test].tolist())
        seeds.extend([int(seed)] * int(np.sum(test)))
        rf = clf.named_steps["randomforestclassifier"]
        importances = importances.add(pd.Series(rf.feature_importances_, index=cols), fill_value=0.0)
        n_fits += 1
    if not preds:
        return None
    labels = sorted(set(truth) | set(preds))
    cm = confusion_matrix(truth, preds, labels=labels)
    by_seed = pd.DataFrame({"seed": seeds, "truth": truth, "pred": preds})
    by_seed["correct"] = by_seed["truth"].eq(by_seed["pred"])
    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "labels": labels,
        "confusion": cm,
        "by_seed": by_seed.groupby("seed")["correct"].mean().to_dict(),
        "top_features": (importances / max(n_fits, 1)).sort_values(ascending=False).head(8),
    }


def _write_classifier_sections(lines, df):
    ok = df[df["status"].eq("success")].copy()
    if ok.empty:
        return
    tmp = ok.copy()
    tmp["binary_target"] = np.where(tmp["expected_h0"].astype(bool), "observable", "hidden")
    binary = _evaluate_classifier(tmp, "binary_target")
    hidden = _evaluate_classifier(tmp, "label", row_filter=lambda x: ~x["expected_h0"].astype(bool))
    lines.extend(["## Classifier diagnostics", ""])
    if binary is None:
        lines.extend(["- binary leave-one-seed-out: unavailable", ""])
    else:
        lines.extend([
            f"- binary leave-one-seed-out accuracy: {binary['accuracy']:.3f}",
            "",
            "### Binary confusion",
            "",
        ])
        lines.extend(_format_confusion_table(binary["labels"], binary["confusion"]))
        lines.extend(["", "### Binary top features", ""])
        for name, val in binary["top_features"].items():
            lines.append(f"- `{name}`: {float(val):.4f}")
        lines.append("")
    if hidden is None:
        lines.extend(["- hidden-type leave-one-seed-out: unavailable", ""])
    else:
        lines.extend([
            f"- hidden-type leave-one-seed-out accuracy: {hidden['accuracy']:.3f}",
            "",
            "### Hidden-type confusion",
            "",
        ])
        lines.extend(_format_confusion_table(hidden["labels"], hidden["confusion"]))
        lines.extend(["", "### Hidden-type top features", ""])
        for name, val in hidden["top_features"].items():
            lines.append(f"- `{name}`: {float(val):.4f}")
        lines.append("")


def _write_summary(df, out_md):
    lines = [
        "# Fast surrogate residual feature summary",
        "",
        f"- rows: {len(df)}",
        "",
    ]
    if not df.empty and "status" in df.columns:
        status_counts = df["status"].value_counts(dropna=False)
        lines.append("## Status")
        lines.append("")
        for k, v in status_counts.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")
    if not df.empty and "h0_accepted" in df.columns and "expected_h0" in df.columns:
        ok = df[df["status"].eq("success")].copy()
        if not ok.empty:
            acc = float(ok["h0_expected_correct"].mean())
            lines.extend([
                "## H0 gate",
                "",
                f"- expected-correct rate: {acc:.3f}",
                f"- accepted observable controls: {int(((ok['expected_h0']) & (ok['h0_accepted'])).sum())}/{int(ok['expected_h0'].sum())}",
                f"- rejected hidden cases: {int(((~ok['expected_h0']) & (~ok['h0_accepted'])).sum())}/{int((~ok['expected_h0']).sum())}",
                "",
            ])
            ct = pd.crosstab(ok["expected_h0"].map({True: "observable_control", False: "hidden_case"}), ok["h0_accepted"].map({True: "accepted", False: "rejected"}))
            lines.extend(["|expected|accepted|rejected|", "|---|---:|---:|"])
            for idx, row in ct.iterrows():
                lines.append(f"|{idx}|{int(row.get('accepted', 0))}|{int(row.get('rejected', 0))}|")
            lines.append("")
    _write_classifier_sections(lines, df)
    keep_cols = [
        "pd_model",
        "label",
        "seed",
        "status",
        "h0_accepted",
        "h0_verdict",
        "mechanistic_mse",
        "empirical_mse",
        "minimal_to_mechanistic_gain",
        "mechanistic_to_empirical_gain",
        "poor_fit_score",
        "amplitude_weighted_structure_score",
        "residual_peak_lag_score",
        "residual_loop_area_score",
        "residual_autocorr_score",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    if cols:
        lines.extend(["## Cases", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"])
        for _, row in df[cols].iterrows():
            vals = []
            for c in cols:
                val = row[c]
                if isinstance(val, float):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append(str(val))
            lines.append("|" + "|".join(vals) + "|")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Fast fixed-surrogate residual feature extraction for PKPD model families.")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_fast_surrogate_residuals"))
    parser.add_argument("--pd-models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    parser.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dose-levels", default="0,0.2,1,5")
    parser.add_argument("--disable-iiv", action="store_true")
    parser.add_argument("--disable-quality-guard", action="store_true", help="Do not retune simulated PD parameters per subject; useful for fixed H0 protocol calibration.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-jsonl", action="store_true", help="Write per-case nested evidence JSONL for debugging.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = [x.strip() for x in args.pd_models.split(",") if x.strip()]
    dose_levels = [float(x.strip()) for x in args.dose_levels.split(",") if x.strip()]
    cases = []
    for rep in range(max(1, int(args.n_replicates))):
        seed = int(args.seed) + rep
        for model_name in models:
            cases.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "n_subjects": int(args.n_subjects),
                    "pk_route": args.pk_route,
                    "pk_compartments": int(args.pk_compartments),
                    "dose_levels": dose_levels,
                    "disable_iiv": bool(args.disable_iiv),
                    "disable_quality_guard": bool(args.disable_quality_guard),
                }
            )

    rows = []
    details = []
    if int(args.workers) <= 1 or len(cases) <= 1:
        for case in cases:
            row, detail = _run_one_case(case)
            rows.append(row)
            if detail is not None:
                details.append(detail)
    else:
        with cf.ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            for row, detail in ex.map(_run_one_case, cases):
                rows.append(row)
                if detail is not None:
                    details.append(detail)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "surrogate_residual_features.csv")
    md_path = os.path.join(args.out_dir, "surrogate_residual_summary.md")
    df.to_csv(csv_path, index=False)
    _write_summary(df, md_path)
    if args.write_jsonl:
        jsonl_path = os.path.join(args.out_dir, "surrogate_residual_details.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in details:
                f.write(json.dumps(_to_jsonable(item), ensure_ascii=False) + "\n")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
