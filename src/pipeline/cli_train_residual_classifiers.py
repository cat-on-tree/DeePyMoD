import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.cli_fast_surrogate_residuals import RESIDUAL_FEATURE_COLUMNS


def _load_features(path):
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].eq("success")].copy()
    required = {"seed", "label", "expected_h0"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in feature CSV: {missing}")
    feature_cols = [c for c in RESIDUAL_FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        raise ValueError("No residual feature columns found.")
    df = df.dropna(subset=feature_cols + ["seed", "label", "expected_h0"]).copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["stage1_target"] = np.where(df["expected_h0"].astype(bool), "observable", "hidden")
    return df, feature_cols


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


def _fit_full_classifier(df, feature_cols, target_col, seed):
    clf = _make_classifier(seed)
    clf.fit(df[feature_cols], df[target_col].astype(str))
    return clf


def _feature_importance(clf, feature_cols):
    rf = clf.named_steps["randomforestclassifier"]
    return pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)


def _leave_one_seed_eval(df, feature_cols, target_col, seed_col="seed"):
    rows = []
    importances = pd.Series(0.0, index=feature_cols)
    n_fits = 0
    for seed in sorted(df[seed_col].unique()):
        train_mask = df[seed_col].ne(seed)
        test_mask = df[seed_col].eq(seed)
        y_train = df.loc[train_mask, target_col].astype(str)
        if y_train.nunique() < 2:
            continue
        clf = _make_classifier(seed)
        clf.fit(df.loc[train_mask, feature_cols], y_train)
        pred = clf.predict(df.loc[test_mask, feature_cols])
        proba = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(df.loc[test_mask, feature_cols])
        classes = list(clf.classes_)
        for j, (idx, p) in enumerate(zip(df.index[test_mask], pred)):
            row = {
                "row_index": int(idx),
                "seed": int(seed),
                "true": str(df.loc[idx, target_col]),
                "pred": str(p),
            }
            if proba is not None:
                for cls, val in zip(classes, proba[j]):
                    row[f"p_{cls}"] = float(val)
            rows.append(row)
        importances = importances.add(_feature_importance(clf, feature_cols), fill_value=0.0)
        n_fits += 1
    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return None
    labels = sorted(set(pred_df["true"]) | set(pred_df["pred"]))
    cm = confusion_matrix(pred_df["true"], pred_df["pred"], labels=labels)
    by_seed = pred_df.assign(correct=pred_df["true"].eq(pred_df["pred"])).groupby("seed")["correct"].mean()
    return {
        "accuracy": float(accuracy_score(pred_df["true"], pred_df["pred"])),
        "labels": labels,
        "confusion": cm,
        "by_seed": by_seed,
        "predictions": pred_df,
        "feature_importance": (importances / max(n_fits, 1)).sort_values(ascending=False),
    }


def _format_confusion(labels, cm):
    lines = [
        "|true\\pred|" + "|".join(str(x) for x in labels) + "|",
        "|" + "|".join(["---"] + ["---:"] * len(labels)) + "|",
    ]
    for label, row in zip(labels, cm):
        lines.append("|" + str(label) + "|" + "|".join(str(int(x)) for x in row) + "|")
    return lines


def _append_eval_section(lines, title, result):
    lines.extend([f"## {title}", ""])
    if result is None:
        lines.extend(["Evaluation unavailable: not enough seeds/classes.", ""])
        return
    lines.extend([f"- leave-one-seed-out accuracy: {result['accuracy']:.4f}", ""])
    lines.extend(["### Confusion", ""])
    lines.extend(_format_confusion(result["labels"], result["confusion"]))
    lines.extend(["", "### Accuracy by seed", "", "|seed|accuracy|", "|---:|---:|"])
    for seed, acc in result["by_seed"].items():
        lines.append(f"|{int(seed)}|{float(acc):.4f}|")
    lines.extend(["", "### Top features", ""])
    for name, val in result["feature_importance"].head(12).items():
        lines.append(f"- `{name}`: {float(val):.4f}")
    lines.append("")


def _write_report(path, df, feature_cols, stage1_eval, stage2_eval, cascade):
    lines = [
        "# Two-stage residual classifier report",
        "",
        f"- rows used: {len(df)}",
        f"- seeds: {df['seed'].nunique()}",
        f"- feature columns: {len(feature_cols)}",
        "",
        "## Stage design",
        "",
        "- Stage 1: observable vs hidden",
        "- Stage 2: hidden type classification, trained only on hidden-labeled rows",
        "",
    ]
    counts = df.groupby(["stage1_target", "label"]).size().reset_index(name="n")
    lines.extend(["## Dataset", "", "|stage1|label|n|", "|---|---|---:|"])
    for _, row in counts.iterrows():
        lines.append(f"|{row['stage1_target']}|{row['label']}|{int(row['n'])}|")
    lines.append("")
    _append_eval_section(lines, "Stage 1 H0 classifier", stage1_eval)
    _append_eval_section(lines, "Stage 2 hidden-type classifier", stage2_eval)
    lines.extend(["## Cascaded two-stage result", ""])
    if cascade is None:
        lines.extend(["Cascaded evaluation unavailable.", ""])
    else:
        lines.extend([f"- cascaded accuracy over all rows: {cascade['accuracy']:.4f}", ""])
        lines.extend(_format_confusion(cascade["labels"], cascade["confusion"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_cascade(stage1_eval, stage2_eval, df):
    if stage1_eval is None or stage2_eval is None:
        return None
    s1 = stage1_eval["predictions"][["row_index", "pred"]].rename(columns={"pred": "stage1_pred"})
    s2 = stage2_eval["predictions"][["row_index", "pred"]].rename(columns={"pred": "stage2_pred"})
    merged = df[["label", "stage1_target"]].copy()
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


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate two-stage residual classifiers.")
    parser.add_argument("--features-csv", default=os.path.join("artifacts", "pkpd_fast_surrogate_residuals_v3", "surrogate_residual_features.csv"))
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_residual_classifiers_v1"))
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, feature_cols = _load_features(args.features_csv)
    hidden_df = df[~df["expected_h0"].astype(bool)].copy()

    stage1_eval = _leave_one_seed_eval(df, feature_cols, "stage1_target")
    stage2_eval = _leave_one_seed_eval(hidden_df, feature_cols, "label")
    cascade = _build_cascade(stage1_eval, stage2_eval, df)

    stage1_model = _fit_full_classifier(df, feature_cols, "stage1_target", args.seed)
    stage2_model = _fit_full_classifier(hidden_df, feature_cols, "label", args.seed + 1)
    joblib.dump({"model": stage1_model, "feature_cols": feature_cols}, os.path.join(args.out_dir, "stage1_h0_classifier.joblib"))
    joblib.dump({"model": stage2_model, "feature_cols": feature_cols}, os.path.join(args.out_dir, "stage2_hidden_type_classifier.joblib"))

    if stage1_eval is not None:
        stage1_eval["predictions"].to_csv(os.path.join(args.out_dir, "stage1_leave_seed_predictions.csv"), index=False)
    if stage2_eval is not None:
        stage2_eval["predictions"].to_csv(os.path.join(args.out_dir, "stage2_leave_seed_predictions.csv"), index=False)
    if cascade is not None:
        cascade["predictions"].to_csv(os.path.join(args.out_dir, "cascade_leave_seed_predictions.csv"), index=False)

    manifest = {
        "features_csv": args.features_csv,
        "n_rows": int(len(df)),
        "n_seeds": int(df["seed"].nunique()),
        "feature_cols": feature_cols,
        "stage1_accuracy": None if stage1_eval is None else float(stage1_eval["accuracy"]),
        "stage2_accuracy": None if stage2_eval is None else float(stage2_eval["accuracy"]),
        "cascade_accuracy": None if cascade is None else float(cascade["accuracy"]),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(args.out_dir, "residual_classifier_report.md")
    _write_report(report_path, df, feature_cols, stage1_eval, stage2_eval, cascade)
    print(report_path)
    print(os.path.join(args.out_dir, "stage1_h0_classifier.joblib"))
    print(os.path.join(args.out_dir, "stage2_hidden_type_classifier.joblib"))


if __name__ == "__main__":
    main()
