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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.defaults import DEFAULTS
from src.data.simulate_pkpd import generate_population_data
from src.pipeline.cli_fast_surrogate_residuals import (
    DEFAULT_MODELS,
    RESIDUAL_FEATURE_COLUMNS,
    _flatten_case,
    _label_for_model,
)
from src.pipeline.mechanism_hints import build_fixed_surrogate_protocol_evidence


def _make_classifier(seed):
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=2,
            random_state=int(seed),
            class_weight="balanced",
        ),
    )


def _raw_pd_features(df_case):
    meta = df_case[["case_id", "pd_model", "label", "expected_h0", "seed"]].iloc[0].to_dict()
    features = dict(meta)
    for col in ["R_obs"]:
        vals = df_case[col].to_numpy(dtype=float)
        scale = float(np.std(vals) + 1e-8)
        center = float(np.mean(vals))
        for (dose, time), grp in df_case.groupby(["dose_level", "time"], sort=True):
            key = f"d{float(dose):g}_t{float(time):g}".replace(".", "p").replace("-", "m")
            raw = grp[col].to_numpy(dtype=float)
            features[f"raw_{key}_mean"] = float(np.mean(raw))
            features[f"raw_{key}_std"] = float(np.std(raw))
            features[f"raw_{key}_zmean"] = float((np.mean(raw) - center) / scale)
    return features


def _run_one_case(args):
    model_name = args["model_name"]
    seed = int(args["seed"])
    rep = int(args["rep"])
    dose_levels = list(args["dose_levels"])
    label = _label_for_model(model_name)
    case_id = f"{model_name}__seed{seed}"
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
            dose_levels=dose_levels,
            disable_iiv=bool(args["disable_iiv"]),
            disable_quality_guard=bool(args["disable_quality_guard"]),
        )
        df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
        df["pd_model"] = model_name
        df["label"] = label
        df["expected_h0"] = bool(label == "observable_sufficient")
        df["seed"] = seed
        df["replicate"] = rep
        df["case_id"] = case_id
        df["dose_level"] = df["sid"].astype(int).map(lambda sid: float(dose_levels[int(sid) % len(dose_levels)]))

        evidence = build_fixed_surrogate_protocol_evidence(df[["sid", "time", "C_obs", "R_obs"]], config=dict(DEFAULTS), interaction_evidence="none")
        if not evidence.get("available", False):
            residual_row = {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "expected_h0": bool(label == "observable_sufficient"),
                "seed": seed,
                "status": "failed",
                "error": str(evidence.get("reason", "surrogate_evidence_unavailable")),
            }
        else:
            residual_row = _flatten_case(model_name, seed, evidence)
            residual_row["case_id"] = case_id
        raw_row = _raw_pd_features(df)
        return {
            "raw_long": df,
            "raw_features": raw_row,
            "residual_features": residual_row,
            "error": None,
        }
    except Exception as exc:
        return {
            "raw_long": pd.DataFrame(),
            "raw_features": {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "expected_h0": bool(label == "observable_sufficient"),
                "seed": seed,
                "status": "failed",
            },
            "residual_features": {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "expected_h0": bool(label == "observable_sufficient"),
                "seed": seed,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _prepare_feature_table(raw_features, residual_features):
    raw = raw_features.copy()
    res = residual_features.copy()
    meta = ["case_id", "pd_model", "label", "expected_h0", "seed"]
    for col in meta:
        if col not in raw.columns and col in res.columns:
            raw[col] = res[col]
        if col not in res.columns and col in raw.columns:
            res[col] = raw[col]
    raw_cols = [c for c in raw.columns if c.startswith("raw_")]
    res_cols = [c for c in RESIDUAL_FEATURE_COLUMNS if c in res.columns]
    df = raw[meta + raw_cols].merge(res[meta + res_cols + [c for c in ["status", "error"] if c in res.columns]], on=meta, how="inner")
    for col in raw_cols + res_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[raw_cols + res_cols] = df[raw_cols + res_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["stage1_target"] = np.where(df["expected_h0"].astype(bool), "observable", "hidden")
    return df, raw_cols, res_cols


def _evaluate(df, feature_cols, target_col, filter_hidden=False):
    data = df.copy()
    if filter_hidden:
        data = data[~data["expected_h0"].astype(bool)].copy()
    if len(data) < 6 or data[target_col].nunique() < 2 or data["seed"].nunique() < 2:
        return None
    rows = []
    importances = pd.Series(0.0, index=feature_cols)
    n_fits = 0
    for seed in sorted(data["seed"].unique()):
        train = data["seed"].ne(seed)
        test = data["seed"].eq(seed)
        if data.loc[train, target_col].nunique() < 2:
            continue
        clf = _make_classifier(seed)
        clf.fit(data.loc[train, feature_cols], data.loc[train, target_col].astype(str))
        pred = clf.predict(data.loc[test, feature_cols])
        proba = clf.predict_proba(data.loc[test, feature_cols])
        classes = list(clf.classes_)
        rf = clf.named_steps["randomforestclassifier"]
        importances = importances.add(pd.Series(rf.feature_importances_, index=feature_cols), fill_value=0.0)
        n_fits += 1
        for j, (idx, pp) in enumerate(zip(data.index[test], pred)):
            row = {
                "row_index": int(idx),
                "case_id": str(data.loc[idx, "case_id"]),
                "seed": int(seed),
                "pd_model": str(data.loc[idx, "pd_model"]),
                "label": str(data.loc[idx, "label"]),
                "true": str(data.loc[idx, target_col]),
                "pred": str(pp),
            }
            for cls, val in zip(classes, proba[j]):
                row[f"p_{cls}"] = float(val)
            rows.append(row)
    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return None
    labels = sorted(set(pred_df["true"]) | set(pred_df["pred"]))
    cm = confusion_matrix(pred_df["true"], pred_df["pred"], labels=labels)
    return {
        "accuracy": float(accuracy_score(pred_df["true"], pred_df["pred"])),
        "labels": labels,
        "confusion": cm,
        "predictions": pred_df,
        "feature_importance": (importances / max(n_fits, 1)).sort_values(ascending=False),
    }


def _cascade(df, stage1, stage2):
    if stage1 is None or stage2 is None:
        return None
    s1 = stage1["predictions"][["row_index", "pred"]].rename(columns={"pred": "stage1_pred"})
    s2 = stage2["predictions"][["row_index", "pred"]].rename(columns={"pred": "stage2_pred"})
    merged = df[["case_id", "pd_model", "label", "stage1_target"]].copy()
    merged["row_index"] = merged.index.astype(int)
    merged = merged.merge(s1, on="row_index", how="left").merge(s2, on="row_index", how="left")
    merged["cascade_pred"] = np.where(merged["stage1_pred"].eq("observable"), "observable_sufficient", merged["stage2_pred"])
    merged["cascade_pred"] = merged["cascade_pred"].fillna("hidden_untyped")
    labels = sorted(set(merged["label"]) | set(merged["cascade_pred"]))
    cm = confusion_matrix(merged["label"], merged["cascade_pred"], labels=labels)
    return {
        "accuracy": float(accuracy_score(merged["label"], merged["cascade_pred"])),
        "labels": labels,
        "confusion": cm,
        "predictions": merged,
    }


def _format_confusion(labels, cm):
    lines = ["|true\\pred|" + "|".join(str(x) for x in labels) + "|", "|" + "|".join(["---"] + ["---:"] * len(labels)) + "|"]
    for label, row in zip(labels, cm):
        lines.append("|" + str(label) + "|" + "|".join(str(int(x)) for x in row) + "|")
    return lines


def _write_report(path, manifest, results):
    lines = [
        "# Raw PD vs residual classifier controls",
        "",
        "## Design",
        "",
        "- Same generated cases are used for all three controls.",
        "- Labels are used only as supervised targets/evaluation columns, never as feature columns.",
        "- Split is leave-one-seed-out for all methods.",
        "- Stage 1 target: observable vs hidden.",
        "- Stage 2 target: hidden type only among true hidden rows.",
        "",
        "## Dataset",
        "",
        f"- rows: {manifest['n_rows']}",
        f"- seeds: {manifest['n_seeds']}",
        f"- models: {manifest['n_models']}",
        f"- raw feature columns: {manifest['n_raw_features']}",
        f"- residual feature columns: {manifest['n_residual_features']}",
        "",
        "## Summary",
        "",
        "|method|stage1_acc|stage2_acc|cascade_acc|",
        "|---|---:|---:|---:|",
    ]
    for method, result in results.items():
        stage1 = result["stage1"]
        stage2 = result["stage2"]
        cascade = result["cascade"]
        lines.append(
            f"|{method}|"
            f"{stage1['accuracy'] if stage1 else float('nan'):.4f}|"
            f"{stage2['accuracy'] if stage2 else float('nan'):.4f}|"
            f"{cascade['accuracy'] if cascade else float('nan'):.4f}|"
        )
    for method, result in results.items():
        lines.extend(["", f"## {method}", ""])
        for stage_name in ["stage1", "stage2", "cascade"]:
            obj = result[stage_name]
            lines.extend(["", f"### {stage_name}", ""])
            if obj is None:
                lines.append("Unavailable.")
                continue
            lines.append(f"- accuracy: {obj['accuracy']:.4f}")
            lines.append("")
            lines.extend(_format_confusion(obj["labels"], obj["confusion"]))
            if stage_name != "cascade":
                lines.extend(["", "Top features:", ""])
                for name, val in obj["feature_importance"].head(10).items():
                    lines.append(f"- `{name}`: {float(val):.4f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_confusion(path, labels, cm, title):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.75), max(4, len(labels) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_accuracy_bar(path, results):
    methods = list(results.keys())
    stages = ["stage1", "stage2", "cascade"]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, stage in enumerate(stages):
        vals = [results[m][stage]["accuracy"] if results[m][stage] is not None else np.nan for m in methods]
        ax.bar(x + (k - 1) * width, vals, width, label=stage)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("leave-one-seed-out accuracy")
    ax.legend()
    ax.set_title("Raw PD vs residual controls")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_raw_curves(path, raw_long):
    labels = sorted(raw_long["label"].unique())
    dose_levels = sorted(raw_long["dose_level"].unique())
    n = len(dose_levels)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 4)), squeeze=False)
    for ax, dose in zip(axes.ravel(), dose_levels):
        sub = raw_long[np.isclose(raw_long["dose_level"], dose)]
        for label in labels:
            agg = sub[sub["label"].eq(label)].groupby("time", as_index=False).agg(R=("R_obs", "mean"))
            if not agg.empty:
                ax.plot(agg["time"], agg["R"], label=label, linewidth=1.4)
        ax.set_title(f"dose={dose:g}")
        ax.set_xlabel("time")
        ax.set_ylabel("mean R_obs")
    for ax in axes.ravel()[len(dose_levels):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pca(path, df, feature_cols, title):
    if len(feature_cols) < 2:
        return
    x = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    x = StandardScaler().fit_transform(x)
    xy = PCA(n_components=2, random_state=0).fit_transform(x)
    labels = sorted(df["label"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, label in enumerate(labels):
        m = df["label"].eq(label).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], label=label, s=28, alpha=0.85, color=cmap(i % 10))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare raw PD curve RF, residual RF, and raw+residual RF on identical generated cases.")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_raw_vs_residual_controls_v1"))
    parser.add_argument("--pd-models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    parser.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dose-levels", default="0,0.2,1,5")
    parser.add_argument("--disable-iiv", action="store_true", default=True)
    parser.add_argument("--disable-quality-guard", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
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
                    "rep": rep,
                    "n_subjects": int(args.n_subjects),
                    "pk_route": args.pk_route,
                    "pk_compartments": int(args.pk_compartments),
                    "dose_levels": dose_levels,
                    "disable_iiv": bool(args.disable_iiv),
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
    feature_table, raw_cols, residual_cols = _prepare_feature_table(raw_features, residual_features)

    raw_long.to_csv(os.path.join(args.out_dir, "raw_observations_long.csv"), index=False)
    raw_features.to_csv(os.path.join(args.out_dir, "raw_pd_curve_features.csv"), index=False)
    residual_features.to_csv(os.path.join(args.out_dir, "residual_features.csv"), index=False)
    feature_table.to_csv(os.path.join(args.out_dir, "classifier_feature_table.csv"), index=False)

    methods = {
        "raw_pd": raw_cols,
        "residual": residual_cols,
        "raw_plus_residual": raw_cols + residual_cols,
    }
    results = {}
    for method, cols in methods.items():
        stage1 = _evaluate(feature_table, cols, "stage1_target", filter_hidden=False)
        stage2 = _evaluate(feature_table, cols, "label", filter_hidden=True)
        cascade = _cascade(feature_table, stage1, stage2)
        results[method] = {"stage1": stage1, "stage2": stage2, "cascade": cascade}
        for stage_name, result in results[method].items():
            if result is not None and "predictions" in result:
                result["predictions"].to_csv(os.path.join(args.out_dir, f"{method}_{stage_name}_predictions.csv"), index=False)
            if result is not None:
                _plot_confusion(
                    os.path.join(plot_dir, f"{method}_{stage_name}_confusion.png"),
                    result["labels"],
                    result["confusion"],
                    f"{method} {stage_name}",
                )

    manifest = {
        "n_rows": int(len(feature_table)),
        "n_seeds": int(feature_table["seed"].nunique()),
        "n_models": int(feature_table["pd_model"].nunique()),
        "n_raw_features": int(len(raw_cols)),
        "n_residual_features": int(len(residual_cols)),
        "methods": {
            m: {
                s: (None if results[m][s] is None else float(results[m][s]["accuracy"]))
                for s in ["stage1", "stage2", "cascade"]
            }
            for m in results
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _write_report(os.path.join(args.out_dir, "raw_vs_residual_report.md"), manifest, results)
    _plot_accuracy_bar(os.path.join(plot_dir, "accuracy_comparison.png"), results)
    if not raw_long.empty:
        _plot_raw_curves(os.path.join(plot_dir, "raw_pd_mean_curves_by_label.png"), raw_long)
    _plot_pca(os.path.join(plot_dir, "pca_raw_pd_features.png"), feature_table, raw_cols, "Raw PD curve features PCA")
    _plot_pca(os.path.join(plot_dir, "pca_residual_features.png"), feature_table, residual_cols, "Residual features PCA")
    _plot_pca(os.path.join(plot_dir, "pca_raw_plus_residual_features.png"), feature_table, raw_cols + residual_cols, "Raw + residual features PCA")

    print(os.path.join(args.out_dir, "raw_vs_residual_report.md"))
    print(os.path.join(args.out_dir, "raw_observations_long.csv"))
    print(plot_dir)


if __name__ == "__main__":
    main()
