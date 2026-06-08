import argparse
import concurrent.futures as cf
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.defaults import DEFAULTS
from src.data.simulate_pkpd import generate_population_data
from src.pipeline.cli_compare_raw_residual_baselines import (
    _format_confusion,
    _plot_confusion,
    _plot_raw_curves,
    _prepare_feature_table,
    _raw_pd_features,
)
from src.pipeline.cli_fast_surrogate_residuals import (
    DEFAULT_MODELS,
    RESIDUAL_FEATURE_COLUMNS,
    _flatten_case,
    _label_for_model,
)
from src.pipeline.mechanism_hints import build_fixed_surrogate_protocol_evidence


SCENARIOS = [
    "clean_id",
    "baseline_shift",
    "dose_shift",
    "sparse_sampling",
    "noise_pk_error",
    "iiv_shift",
    "combined_shift",
]


def _make_classifier(seed):
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            random_state=int(seed),
            class_weight="balanced",
        ),
    )


def _parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _scenario_options(name, train_dose_levels, shifted_dose_levels):
    opts = {
        "dose_levels": list(train_dose_levels),
        "observation_times": None,
        "disable_iiv": True,
        "extra_pk_iiv_sigma": 0.0,
        "baseline_shift": False,
        "pd_noise_sigma": 0.0,
        "pk_error_sigma": 0.0,
    }
    if name == "clean_id":
        return opts
    if name == "baseline_shift":
        opts["baseline_shift"] = True
        return opts
    if name == "dose_shift":
        opts["dose_levels"] = list(shifted_dose_levels)
        return opts
    if name == "sparse_sampling":
        opts["observation_times"] = [0, 1, 2, 4, 8, 12, 24, 36, 48]
        return opts
    if name == "noise_pk_error":
        opts["pd_noise_sigma"] = 0.10
        opts["pk_error_sigma"] = 0.25
        opts["extra_pk_iiv_sigma"] = 0.25
        return opts
    if name == "iiv_shift":
        opts["disable_iiv"] = False
        opts["extra_pk_iiv_sigma"] = 0.20
        return opts
    if name == "combined_shift":
        opts["dose_levels"] = list(shifted_dose_levels)
        opts["observation_times"] = [0, 1, 2, 4, 8, 12, 24, 36, 48]
        opts["disable_iiv"] = False
        opts["extra_pk_iiv_sigma"] = 0.25
        opts["baseline_shift"] = True
        opts["pd_noise_sigma"] = 0.12
        opts["pk_error_sigma"] = 0.30
        return opts
    raise ValueError(f"Unknown scenario: {name}")


def _apply_label_preserving_perturbations(df, scenario_opts, seed):
    out = df.copy()
    rng = np.random.default_rng(int(seed) + 7919)
    if bool(scenario_opts.get("baseline_shift", False)):
        for sid, idx in out.groupby("sid").groups.items():
            idx = list(idx)
            vals = out.loc[idx, "R_obs"].to_numpy(dtype=float)
            baseline = float(vals[0])
            spread = float(np.std(vals) + 1e-8)
            amp_scale = float(np.exp(rng.normal(0.0, 0.35)))
            offset = float(rng.normal(0.0, 0.15) * max(abs(baseline), spread, 1e-8))
            out.loc[idx, "R_obs"] = np.clip(baseline + offset + amp_scale * (vals - baseline), 1e-8, None)
    pd_noise_sigma = float(scenario_opts.get("pd_noise_sigma", 0.0))
    if pd_noise_sigma > 0:
        for sid, idx in out.groupby("sid").groups.items():
            idx = list(idx)
            vals = out.loc[idx, "R_obs"].to_numpy(dtype=float)
            sigma = pd_noise_sigma * float(np.std(vals) + 1e-8)
            out.loc[idx, "R_obs"] = np.clip(vals + rng.normal(0.0, sigma, size=vals.size), 1e-8, None)
    pk_error_sigma = float(scenario_opts.get("pk_error_sigma", 0.0))
    if pk_error_sigma > 0:
        noise = np.exp(rng.normal(0.0, pk_error_sigma, size=len(out)))
        out["C_obs"] = np.clip(out["C_obs"].to_numpy(dtype=float) * noise, 0.0, None)
    return out


def _run_one_case(args):
    model_name = args["model_name"]
    seed = int(args["seed"])
    split = str(args["split"])
    scenario = str(args["scenario"])
    label = _label_for_model(model_name)
    case_id = f"{split}__{scenario}__{model_name}__seed{seed}"
    try:
        pop_data, _, _ = generate_population_data(
            model_name=model_name,
            seed=seed,
            n_subjects=int(args["n_subjects"]),
            extra_pk_iiv_sigma=float(args["extra_pk_iiv_sigma"]),
            return_pk_scale=False,
            pk_route=args["pk_route"],
            pk_compartments=int(args["pk_compartments"]),
            dose_design_enabled=True,
            dose_levels=list(args["dose_levels"]),
            observation_times=args["observation_times"],
            disable_iiv=bool(args["disable_iiv"]),
            disable_quality_guard=bool(args["disable_quality_guard"]),
        )
        df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
        df = _apply_label_preserving_perturbations(df, args, seed)
        df["pd_model"] = model_name
        df["label"] = label
        df["expected_h0"] = bool(label == "observable_sufficient")
        df["seed"] = seed
        df["split"] = split
        df["scenario"] = scenario
        df["case_id"] = case_id
        dose_levels = list(args["dose_levels"])
        df["dose_level"] = df["sid"].astype(int).map(lambda sid: float(dose_levels[int(sid) % len(dose_levels)]))

        evidence = build_fixed_surrogate_protocol_evidence(
            df[["sid", "time", "C_obs", "R_obs"]],
            config=dict(DEFAULTS),
            interaction_evidence="none",
        )
        if not evidence.get("available", False):
            residual_row = {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "expected_h0": bool(label == "observable_sufficient"),
                "seed": seed,
                "split": split,
                "scenario": scenario,
                "status": "failed",
                "error": str(evidence.get("reason", "surrogate_evidence_unavailable")),
            }
        else:
            residual_row = _flatten_case(model_name, seed, evidence)
            residual_row.update({"case_id": case_id, "split": split, "scenario": scenario})
        raw_row = _raw_pd_features(df)
        raw_row.update({"split": split, "scenario": scenario})
        return {"raw_long": df, "raw_features": raw_row, "residual_features": residual_row, "error": None}
    except Exception as exc:
        base = {
            "case_id": case_id,
            "pd_model": model_name,
            "label": label,
            "expected_h0": bool(label == "observable_sufficient"),
            "seed": seed,
            "split": split,
            "scenario": scenario,
            "status": "failed",
        }
        err = {"error": f"{exc.__class__.__name__}: {exc}", "traceback": traceback.format_exc()}
        return {
            "raw_long": pd.DataFrame(),
            "raw_features": dict(base),
            "residual_features": {**base, **err},
            "error": err["error"],
        }


def _prepare_domain_feature_table(raw_features, residual_features):
    raw = raw_features.copy()
    res = residual_features.copy()
    for col in ["split", "scenario"]:
        if col not in raw.columns:
            raw[col] = ""
        if col not in res.columns:
            res[col] = ""
    table, raw_cols, residual_cols = _prepare_feature_table(raw, res)
    meta = raw[["case_id", "split", "scenario"]].drop_duplicates()
    table = table.merge(meta, on="case_id", how="left", suffixes=("", "_meta"))
    for col in ["split", "scenario"]:
        if f"{col}_meta" in table.columns:
            table[col] = table[col].fillna(table[f"{col}_meta"])
            table = table.drop(columns=[f"{col}_meta"])
    return table, raw_cols, residual_cols


def _predict_frame(df, clf, feature_cols, target_col):
    pred = clf.predict(df[feature_cols])
    proba = clf.predict_proba(df[feature_cols]) if hasattr(clf, "predict_proba") else None
    classes = list(clf.classes_)
    rows = []
    for j, idx in enumerate(df.index):
        row = {
            "row_index": int(idx),
            "case_id": str(df.loc[idx, "case_id"]),
            "scenario": str(df.loc[idx, "scenario"]),
            "seed": int(df.loc[idx, "seed"]),
            "pd_model": str(df.loc[idx, "pd_model"]),
            "label": str(df.loc[idx, "label"]),
            "true": str(df.loc[idx, target_col]),
            "pred": str(pred[j]),
        }
        if proba is not None:
            for cls, val in zip(classes, proba[j]):
                row[f"p_{cls}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def _classification_metrics(pred_df, labels=None, auroc_positive=None):
    if pred_df.empty:
        return None
    if labels is None:
        labels = sorted(set(pred_df["true"]) | set(pred_df["pred"]))
    cm = confusion_matrix(pred_df["true"], pred_df["pred"], labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(pred_df["true"], pred_df["pred"])),
        "macro_f1": float(f1_score(pred_df["true"], pred_df["pred"], average="macro", zero_division=0)),
        "labels": labels,
        "confusion": cm,
        "predictions": pred_df,
    }
    if auroc_positive is not None and f"p_{auroc_positive}" in pred_df.columns and pred_df["true"].nunique() == 2:
        y = pred_df["true"].eq(auroc_positive).astype(int)
        metrics["auroc"] = float(roc_auc_score(y, pred_df[f"p_{auroc_positive}"]))
    else:
        metrics["auroc"] = None
    return metrics


def _evaluate_method(train_df, test_df, feature_cols, seed):
    train = train_df.copy()
    test = test_df.copy()
    train_hidden = train[~train["expected_h0"].astype(bool)].copy()
    test_hidden = test[~test["expected_h0"].astype(bool)].copy()
    if not feature_cols or train["stage1_target"].nunique() < 2 or train_hidden["label"].nunique() < 2:
        return None

    stage1_clf = _make_classifier(seed)
    stage1_clf.fit(train[feature_cols], train["stage1_target"].astype(str))
    stage1_pred = _predict_frame(test, stage1_clf, feature_cols, "stage1_target")
    stage1 = _classification_metrics(stage1_pred, labels=["hidden", "observable"], auroc_positive="hidden")

    stage2_clf = _make_classifier(seed + 1)
    stage2_clf.fit(train_hidden[feature_cols], train_hidden["label"].astype(str))
    stage2_pred = _predict_frame(test_hidden, stage2_clf, feature_cols, "label")
    stage2 = _classification_metrics(stage2_pred)

    cascade = test[["case_id", "scenario", "seed", "pd_model", "label", "stage1_target"]].copy()
    cascade["row_index"] = cascade.index.astype(int)
    s1 = stage1_pred[["row_index", "pred"]].rename(columns={"pred": "stage1_pred"})
    cascade = cascade.merge(s1, on="row_index", how="left")
    hidden_mask = cascade["stage1_pred"].eq("hidden")
    cascade["cascade_pred"] = "observable_sufficient"
    if hidden_mask.any():
        hidden_rows = test.loc[cascade.loc[hidden_mask, "row_index"].to_numpy()]
        hidden_pred = stage2_clf.predict(hidden_rows[feature_cols])
        cascade.loc[hidden_mask, "cascade_pred"] = hidden_pred
    cascade_pred = cascade.rename(columns={"label": "true", "cascade_pred": "pred"})
    cascade_metrics = _classification_metrics(cascade_pred[["row_index", "case_id", "scenario", "seed", "pd_model", "true", "pred"]])

    return {"stage1": stage1, "stage2": stage2, "cascade": cascade_metrics}


def _plot_shift_accuracy(path, summary):
    methods = list(summary["method"].drop_duplicates())
    scenarios = list(summary["scenario"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.2), 4.8))
    for i, method in enumerate(methods):
        vals = []
        for scenario in scenarios:
            sub = summary[(summary["scenario"].eq(scenario)) & (summary["method"].eq(method))]
            vals.append(float(sub["cascade_acc"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cascade accuracy")
    ax.set_title("Clean-trained classifier under domain shifts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_shift_drop(path, summary):
    clean = summary[summary["scenario"].eq("clean_id")][["method", "cascade_acc"]].rename(columns={"cascade_acc": "clean_acc"})
    df = summary.merge(clean, on="method", how="left")
    df["cascade_drop"] = df["clean_acc"] - df["cascade_acc"]
    methods = list(df["method"].drop_duplicates())
    scenarios = [s for s in df["scenario"].drop_duplicates() if s != "clean_id"]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.2), 4.8))
    for i, method in enumerate(methods):
        vals = []
        for scenario in scenarios:
            sub = df[(df["scenario"].eq(scenario)) & (df["method"].eq(method))]
            vals.append(float(sub["cascade_drop"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("accuracy drop vs clean_id")
    ax.set_title("Performance degradation under nuisance shifts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path, manifest, summary, results):
    lines = [
        "# Domain-shift raw PD vs residual baseline comparison",
        "",
        "## Design",
        "",
        "- Train all classifiers only on clean generated cases.",
        "- Evaluate on independent clean seeds and shifted test domains.",
        "- Labels are targets/evaluation columns only; feature columns are raw PD grid summaries, residual signatures, or both.",
        "- Stage 1: observable vs hidden H0 rejection.",
        "- Stage 2: hidden-type classification among true hidden rows.",
        "- Cascade: Stage 1 observable calls map to observable_sufficient; hidden calls are typed by Stage 2.",
        "",
        "## Dataset",
        "",
        f"- train rows: {manifest['n_train_rows']}",
        f"- test rows: {manifest['n_test_rows']}",
        f"- models: {manifest['n_models']}",
        f"- train seeds: {manifest['n_train_seeds']}",
        f"- test seeds: {manifest['n_test_seeds']}",
        f"- raw features: {manifest['n_raw_features']}",
        f"- residual features: {manifest['n_residual_features']}",
        "",
        "## Summary",
        "",
        "|scenario|method|stage1_acc|stage1_f1|stage1_auroc|stage2_acc|stage2_f1|cascade_acc|cascade_f1|cascade_drop_vs_clean|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        auroc = row["stage1_auroc"]
        lines.append(
            f"|{row['scenario']}|{row['method']}|"
            f"{row['stage1_acc']:.4f}|{row['stage1_f1']:.4f}|"
            f"{'' if pd.isna(auroc) else f'{auroc:.4f}'}|"
            f"{row['stage2_acc']:.4f}|{row['stage2_f1']:.4f}|"
            f"{row['cascade_acc']:.4f}|{row['cascade_f1']:.4f}|"
            f"{row['cascade_drop_vs_clean']:.4f}|"
        )
    for scenario, method_results in results.items():
        lines.extend(["", f"## {scenario}", ""])
        for method, result in method_results.items():
            lines.extend(["", f"### {method}", ""])
            for stage_name in ["stage1", "stage2", "cascade"]:
                obj = result[stage_name]
                lines.extend(["", f"#### {stage_name}", ""])
                lines.append(f"- accuracy: {obj['accuracy']:.4f}")
                lines.append(f"- macro F1: {obj['macro_f1']:.4f}")
                if obj.get("auroc") is not None:
                    lines.append(f"- AUROC: {obj['auroc']:.4f}")
                lines.append("")
                lines.extend(_format_confusion(obj["labels"], obj["confusion"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Stress-test raw PD, residual, and raw+residual classifiers under nuisance/domain shifts.")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_raw_vs_residual_domain_shift_v1"))
    parser.add_argument("--pd-models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=1042)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    parser.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--train-dose-levels", default="0,0.2,1,5")
    parser.add_argument("--shifted-dose-levels", default="0,0.1,0.7,3,8")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--disable-quality-guard", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    models = [x.strip() for x in args.pd_models.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {SCENARIOS}")
    train_dose_levels = _parse_float_list(args.train_dose_levels)
    shifted_dose_levels = _parse_float_list(args.shifted_dose_levels)

    cases = []
    for rep in range(max(1, int(args.n_replicates))):
        train_seed = int(args.seed) + rep
        test_seed = int(args.test_seed) + rep
        train_opts = _scenario_options("clean_id", train_dose_levels, shifted_dose_levels)
        for model_name in models:
            cases.append(
                {
                    **train_opts,
                    "model_name": model_name,
                    "seed": train_seed,
                    "split": "train",
                    "scenario": "train_clean",
                    "n_subjects": int(args.n_subjects),
                    "pk_route": args.pk_route,
                    "pk_compartments": int(args.pk_compartments),
                    "disable_quality_guard": bool(args.disable_quality_guard),
                }
            )
            for scenario in scenarios:
                opts = _scenario_options(scenario, train_dose_levels, shifted_dose_levels)
                cases.append(
                    {
                        **opts,
                        "model_name": model_name,
                        "seed": test_seed,
                        "split": "test",
                        "scenario": scenario,
                        "n_subjects": int(args.n_subjects),
                        "pk_route": args.pk_route,
                        "pk_compartments": int(args.pk_compartments),
                        "disable_quality_guard": bool(args.disable_quality_guard),
                    }
                )

    outputs = []
    if int(args.workers) <= 1:
        for case in cases:
            outputs.append(_run_one_case(case))
    else:
        with cf.ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            for out in ex.map(_run_one_case, cases):
                outputs.append(out)

    raw_long = pd.concat([o["raw_long"] for o in outputs if not o["raw_long"].empty], ignore_index=True)
    raw_features = pd.DataFrame([o["raw_features"] for o in outputs])
    residual_features = pd.DataFrame([o["residual_features"] for o in outputs])
    feature_table, raw_cols, residual_cols = _prepare_domain_feature_table(raw_features, residual_features)
    feature_table["split"] = feature_table["split"].fillna("")
    feature_table["scenario"] = feature_table["scenario"].fillna("")

    raw_long.to_csv(os.path.join(args.out_dir, "raw_observations_long.csv"), index=False)
    raw_features.to_csv(os.path.join(args.out_dir, "raw_pd_curve_features.csv"), index=False)
    residual_features.to_csv(os.path.join(args.out_dir, "residual_features.csv"), index=False)
    feature_table.to_csv(os.path.join(args.out_dir, "domain_shift_feature_table.csv"), index=False)

    train_df = feature_table[feature_table["split"].eq("train")].copy()
    test_all = feature_table[feature_table["split"].eq("test")].copy()
    methods = {
        "raw_pd": raw_cols,
        "residual": residual_cols,
        "raw_plus_residual": raw_cols + residual_cols,
    }
    results = {}
    summary_rows = []
    prediction_rows = []
    for scenario in scenarios:
        scenario_df = test_all[test_all["scenario"].eq(scenario)].copy()
        results[scenario] = {}
        for i, (method, cols) in enumerate(methods.items()):
            result = _evaluate_method(train_df, scenario_df, cols, int(args.seed) + 100 * (i + 1))
            if result is None:
                continue
            results[scenario][method] = result
            for stage_name, obj in result.items():
                pred = obj["predictions"].copy()
                pred["method"] = method
                pred["stage"] = stage_name
                prediction_rows.append(pred)
                _plot_confusion(
                    os.path.join(plot_dir, f"{scenario}_{method}_{stage_name}_confusion.png"),
                    obj["labels"],
                    obj["confusion"],
                    f"{scenario} {method} {stage_name}",
                )
            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "stage1_acc": result["stage1"]["accuracy"],
                    "stage1_f1": result["stage1"]["macro_f1"],
                    "stage1_auroc": result["stage1"]["auroc"],
                    "stage2_acc": result["stage2"]["accuracy"],
                    "stage2_f1": result["stage2"]["macro_f1"],
                    "cascade_acc": result["cascade"]["accuracy"],
                    "cascade_f1": result["cascade"]["macro_f1"],
                }
            )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        clean = summary[summary["scenario"].eq("clean_id")][["method", "cascade_acc"]].rename(columns={"cascade_acc": "clean_cascade_acc"})
        summary = summary.merge(clean, on="method", how="left")
        summary["cascade_drop_vs_clean"] = summary["clean_cascade_acc"] - summary["cascade_acc"]
        summary = summary.drop(columns=["clean_cascade_acc"])
    summary.to_csv(os.path.join(args.out_dir, "domain_shift_summary.csv"), index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(os.path.join(args.out_dir, "domain_shift_predictions.csv"), index=False)

    manifest = {
        "scenarios": scenarios,
        "models": models,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_all)),
        "n_models": int(feature_table["pd_model"].nunique()),
        "n_train_seeds": int(train_df["seed"].nunique()),
        "n_test_seeds": int(test_all["seed"].nunique()),
        "n_raw_features": int(len(raw_cols)),
        "n_residual_features": int(len(residual_cols)),
        "train_dose_levels": train_dose_levels,
        "shifted_dose_levels": shifted_dose_levels,
        "scenario_definitions": {name: _scenario_options(name, train_dose_levels, shifted_dose_levels) for name in scenarios},
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _write_report(os.path.join(args.out_dir, "domain_shift_report.md"), manifest, summary, results)

    if not summary.empty:
        _plot_shift_accuracy(os.path.join(plot_dir, "cascade_accuracy_by_shift.png"), summary)
        _plot_shift_drop(os.path.join(plot_dir, "cascade_accuracy_drop_vs_clean.png"), summary)
    if not raw_long.empty:
        for scenario in ["train_clean", "clean_id", "combined_shift"]:
            sub = raw_long[raw_long["scenario"].eq(scenario)]
            if not sub.empty:
                _plot_raw_curves(os.path.join(plot_dir, f"raw_pd_mean_curves_{scenario}.png"), sub)

    print(os.path.join(args.out_dir, "domain_shift_report.md"))
    print(os.path.join(args.out_dir, "domain_shift_summary.csv"))
    print(plot_dir)


if __name__ == "__main__":
    main()
