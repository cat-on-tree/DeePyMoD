import argparse
import concurrent.futures as cf
import json
import os
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from src.data.simulate_pkpd import generate_population_data
from src.configs.pd_module_registry import resolve_discovery_profile
from src.pipeline.auto_workflow import run_discovery_workflow


PD_MODELS_13 = [
    "TMDD_BASE",
    "TRANSDUCTION_DELAY",
    "IDR_INHIB_KIN_SIG",
    "FEEDBACK_REGULATION",
    "CIRCADIAN_REGULATION",
    "TGI_BASIC",
    "DIRECT_SIGEMAX",
    "BIOPHASE_SIGEMAX",
    "DISEASE_PROGRESSION",
    "TOLERANCE_ADAPTATION",
    "DRUG_INTERACTION",
    "PRECURSOR_POOL",
    "ANTIBODY_PKPD",
]

PK_VARIANTS_6 = [
    {"route": "bolus", "compartments": 1},
    {"route": "oral", "compartments": 1},
    {"route": "bolus", "compartments": 2},
    {"route": "oral", "compartments": 2},
    {"route": "bolus", "compartments": 3},
    {"route": "oral", "compartments": 3},
]

TMDD_FAMILY = {"TMDD_BASE", "ANTIBODY_PKPD"}
OBSERVABLE_ONLY_PD = {"DIRECT_SIGEMAX", "IDR_INHIB_KIN_SIG", "TGI_BASIC"}
DEFAULT_DOSE_LEVELS = [0.0, 0.2, 1.0, 5.0]
OBSERVABLE_TIME_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]


def _pk_tag(route, compartments):
    return f"{route}_{compartments}c"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _plot_pk_pd_lines(df, out_dir):
    _ensure_dir(out_dir)
    pk_png = os.path.join(out_dir, "pk_line.png")
    pd_png = os.path.join(out_dir, "pd_line.png")

    dsort = df.sort_values(["sid", "time"])
    agg = dsort.groupby("time", as_index=False).agg(C=("C_obs", "mean"), R=("R_obs", "mean"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, grp in dsort.groupby("sid"):
        ax.plot(grp["time"], grp["C_obs"], alpha=0.25, linewidth=1.0)
    ax.plot(agg["time"], agg["C"], color="black", linewidth=2.0, label="mean")
    ax.set_xlabel("time")
    ax.set_ylabel("C_obs")
    ax.set_title("PK concentration-time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pk_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, grp in dsort.groupby("sid"):
        ax.plot(grp["time"], grp["R_obs"], alpha=0.25, linewidth=1.0)
    ax.plot(agg["time"], agg["R"], color="black", linewidth=2.0, label="mean")
    ax.set_xlabel("time")
    ax.set_ylabel("R_obs")
    ax.set_title("PD response-time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pd_png, dpi=160)
    plt.close(fig)
    return pk_png, pd_png


def _write_markdown_summary(df_sum, out_md):
    lines = [
        "# PK×PD 支持矩阵结果",
        "",
        f"- 生成时间: {datetime.now().isoformat()}",
        f"- 组合数: {len(df_sum)}",
        "",
        "| PK | PD模型 | 状态 | 隐空间现象判定 | 最强现象 | 机制Top1(概率) | Discovery秒 | NLME秒 | 总秒 | 报告 | NLME结果 |",
        "|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for _, row in df_sum.iterrows():
        mech_disp = ""
        if row.get("top_mechanism"):
            mech_disp = f"{row.get('top_mechanism')} ({float(row.get('top_mechanism_prob', 0.0)):.3f})"
        lines.append(
            f"| {row['pk_variant']} | {row['pd_model']} | {row['status']} | "
            f"{row.get('hidden_state_verdict', '')} | {row.get('strongest_hidden_phenomenon', '')} | {mech_disp} | {float(row.get('discovery_seconds', 0.0)):.3f} | "
            f"{float(row.get('nlme_seconds', 0.0)):.3f} | "
            f"{float(row.get('total_workflow_seconds', 0.0)):.3f} | "
            f"{row.get('report_md','')} | {row.get('nlme_csv','')} |"
        )
    lines.extend(
        [
            "",
            "## 失败详情",
            "",
            "| PK | PD模型 | 错误 |",
            "|---|---|---|",
        ]
    )
    failed = df_sum[df_sum["status"] != "success"]
    for _, row in failed.iterrows():
        lines.append(f"| {row['pk_variant']} | {row['pd_model']} | {str(row.get('error','')).replace('|', '/')} |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _row_key(pk_variant, pd_model):
    return f"{pk_variant}::{pd_model}"


def _persist_summary(rows, csv_path, md_path):
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(csv_path, index=False)
    _write_markdown_summary(df_sum, md_path)


def _run_single_case(case):
    route = str(case["route"])
    n_comp = int(case["n_comp"])
    pk_variant = str(case["pk_variant"])
    pd_model = str(case["pd_model"])
    combo_dir = str(case["combo_dir"])
    scope = str(case["scope"])
    hint_scope = str(case["hint_scope"])
    seed = int(case["seed"])
    n_subjects = int(case["n_subjects"])
    skip_bootstrap = bool(case["skip_bootstrap"])
    discovery_topk = int(case["discovery_topk"])
    disable_iiv = bool(case["disable_iiv"])
    resolved_dose_levels = list(case["dose_levels"])
    matlab_engine_reuse = bool(case.get("matlab_engine_reuse", False))

    case_t0 = time.perf_counter()
    try:
        pop_data, _, _ = generate_population_data(
            model_name=pd_model,
            seed=seed,
            n_subjects=n_subjects,
            extra_pk_iiv_sigma=0.0,
            return_pk_scale=False,
            pk_route=route,
            pk_compartments=n_comp,
            dose_design_enabled=(scope == "all" or (scope == "observable" and pd_model in OBSERVABLE_ONLY_PD)),
            dose_levels=resolved_dose_levels,
            observation_times=(OBSERVABLE_TIME_GRID if pd_model in OBSERVABLE_ONLY_PD else None),
            disable_iiv=disable_iiv,
        )
        df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
        df["sid"] = df["sid"].astype(int)

        data_csv = os.path.join(combo_dir, "pkpd_long.csv")
        df.to_csv(data_csv, index=False)
        pk_png, pd_png = _plot_pk_pd_lines(df, combo_dir)

        is_tmdd = pd_model in TMDD_FAMILY
        is_observable = pd_model in OBSERVABLE_ONLY_PD
        discovery_overrides = None
        discovery_profile = "tmdd_antibody" if is_tmdd else ("observable_blind" if is_observable else "observable_surrogate")
        if is_observable:
            discovery_overrides = resolve_discovery_profile(
                "observable_blind",
                available_columns=df.columns,
                pk_compartments=n_comp,
                topk=int(discovery_topk),
            )
        elif not is_tmdd:
            discovery_overrides = resolve_discovery_profile(
                "observable_surrogate",
                available_columns=df.columns,
                pk_compartments=n_comp,
                topk=int(discovery_topk),
            )
        elif discovery_topk is not None:
            discovery_overrides = {"topk": int(discovery_topk)}

        enable_hints = (
            (not is_tmdd)
            and (
                hint_scope == "all"
                or (hint_scope == "standard" and not is_observable)
                or (hint_scope == "observable" and is_observable)
            )
        )

        wf = run_discovery_workflow(
            input_csv=data_csv,
            active_model=(pd_model if is_tmdd else "SMALL_MOLECULE_BLIND"),
            pk_model_name=pk_variant,
            pd_data_label=pd_model,
            out_root=combo_dir,
            pk_route=route,
            pk_compartments=n_comp,
            run_name=f"wf_{pk_variant}_{pd_model}",
            run_initial_confirmation=(not is_tmdd),
            skip_bootstrap=skip_bootstrap,
            enable_mechanism_hints=enable_hints,
            discovery_overrides=discovery_overrides,
            matlab_engine_reuse=matlab_engine_reuse,
        )
        timing = wf.get("timing", {}) or {}
        top_mechanism = ""
        top_mechanism_prob = 0.0
        hidden_state_verdict = ""
        strongest_hidden_phenomenon = ""
        mechanism_hint_json = wf.get("mechanism_hint_json", "")
        mechanism_confirmation_json = wf.get("mechanism_confirmation_json", "")
        if mechanism_hint_json and os.path.exists(mechanism_hint_json):
            with open(mechanism_hint_json, "r", encoding="utf-8") as f:
                mh = json.load(f)
            top_mechanism = str(mh.get("top_mechanism") or "")
            top_mechanism_prob = float(mh.get("top_probability") or 0.0)
            assessment = mh.get("hidden_phenomena_assessment", {}) or {}
            hidden_state_verdict = str(assessment.get("overall_verdict") or "")
            strongest_hidden_phenomenon = str(assessment.get("strongest_phenomenon") or "")
        if mechanism_confirmation_json and os.path.exists(mechanism_confirmation_json):
            with open(mechanism_confirmation_json, "r", encoding="utf-8") as f:
                mc = json.load(f)
            hidden_state_verdict = str(mc.get("hidden_state_verdict") or hidden_state_verdict)
        return {
            "pk_variant": pk_variant,
            "pd_model": pd_model,
            "active_model": (pd_model if is_tmdd else "SMALL_MOLECULE_BLIND"),
            "discovery_profile": discovery_profile,
            "dose_design_scope": scope,
            "status": "success",
            "data_csv": data_csv,
            "pk_png": pk_png,
            "pd_png": pd_png,
            "report_md": wf.get("report_md", ""),
            "nlme_csv": wf.get("nlme_csv", ""),
            "mechanism_hint_json": mechanism_hint_json,
            "mechanism_confirmation_json": mechanism_confirmation_json,
            "top_mechanism": top_mechanism,
            "top_mechanism_prob": top_mechanism_prob,
            "hidden_state_verdict": hidden_state_verdict,
            "strongest_hidden_phenomenon": strongest_hidden_phenomenon,
            "discovery_seconds": float(timing.get("discovery_seconds", 0.0)),
            "nlme_seconds": float(timing.get("nlme_seconds", 0.0)),
            "total_workflow_seconds": float(timing.get("total_workflow_seconds", 0.0)),
            "case_elapsed_seconds": float(time.perf_counter() - case_t0),
            "error": "",
        }
    except Exception as exc:
        err = f"{exc.__class__.__name__}: {exc}"
        tb_path = os.path.join(combo_dir, "error_traceback.txt")
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return {
            "pk_variant": pk_variant,
            "pd_model": pd_model,
            "active_model": (pd_model if pd_model in TMDD_FAMILY else "SMALL_MOLECULE_BLIND"),
            "discovery_profile": (
                "tmdd_antibody" if pd_model in TMDD_FAMILY
                else ("observable_blind" if pd_model in OBSERVABLE_ONLY_PD else "observable_surrogate")
            ),
            "dose_design_scope": scope,
            "status": "failed",
            "data_csv": os.path.join(combo_dir, "pkpd_long.csv"),
            "pk_png": os.path.join(combo_dir, "pk_line.png"),
            "pd_png": os.path.join(combo_dir, "pd_line.png"),
            "report_md": "",
            "nlme_csv": "",
            "mechanism_hint_json": "",
            "mechanism_confirmation_json": "",
            "top_mechanism": "",
            "top_mechanism_prob": 0.0,
            "hidden_state_verdict": "",
            "strongest_hidden_phenomenon": "",
            "discovery_seconds": 0.0,
            "nlme_seconds": 0.0,
            "total_workflow_seconds": 0.0,
            "case_elapsed_seconds": float(time.perf_counter() - case_t0),
            "error": err,
        }


def run_matrix(
    out_dir,
    pd_models=None,
    seed=42,
    n_subjects=12,
    skip_bootstrap=True,
    resume=True,
    rerun_failed=False,
    pk_routes=None,
    pk_compartments=None,
    dose_design_scope="observable",
    dose_levels=None,
    discovery_topk=5,
    disable_iiv=False,
    mechanism_hints_scope="standard",
    workers=1,
    matlab_engine_reuse=False,
):
    _ensure_dir(out_dir)
    pd_list = pd_models or list(PD_MODELS_13)
    routes_filter = {str(x).strip().lower() for x in (pk_routes or []) if str(x).strip()}
    comps_filter = {int(x) for x in (pk_compartments or [])}
    pk_list = [
        pk for pk in PK_VARIANTS_6
        if (not routes_filter or str(pk["route"]).lower() in routes_filter)
        and (not comps_filter or int(pk["compartments"]) in comps_filter)
    ]
    if not pk_list:
        raise ValueError("No PK variants matched filters. Check --pk-routes/--pk-compartments.")
    scope = str(dose_design_scope or "observable").strip().lower()
    if scope not in {"none", "observable", "all"}:
        raise ValueError("dose_design_scope must be one of: none, observable, all")
    resolved_dose_levels = list(dose_levels or DEFAULT_DOSE_LEVELS)
    hint_scope = str(mechanism_hints_scope or "standard").strip().lower()
    if hint_scope not in {"none", "standard", "observable", "all"}:
        raise ValueError("mechanism_hints_scope must be one of: none, standard, observable, all")
    csv_path = os.path.join(out_dir, "matrix_summary.csv")
    md_path = os.path.join(out_dir, "matrix_summary.md")

    existing_map = {}
    rows = []
    if resume and os.path.exists(csv_path):
        try:
            prev_df = pd.read_csv(csv_path)
            for _, r in prev_df.iterrows():
                rec = r.to_dict()
                key = _row_key(str(rec.get("pk_variant", "")), str(rec.get("pd_model", "")))
                existing_map[key] = rec
            rows = list(existing_map.values())
        except Exception:
            existing_map = {}
            rows = []

    todo_cases = []
    for pk in pk_list:
        route = pk["route"]
        n_comp = int(pk["compartments"])
        pk_variant = _pk_tag(route, n_comp)
        for pd_model in pd_list:
            key = _row_key(pk_variant, pd_model)
            if key in existing_map:
                old_status = str(existing_map[key].get("status", ""))
                if old_status == "success" or (old_status == "failed" and not rerun_failed):
                    continue
            combo_dir = _ensure_dir(os.path.join(out_dir, pk_variant, pd_model))
            todo_cases.append(
                {
                    "route": route,
                    "n_comp": n_comp,
                    "pk_variant": pk_variant,
                    "pd_model": pd_model,
                    "combo_dir": combo_dir,
                    "scope": scope,
                    "hint_scope": hint_scope,
                    "seed": seed,
                    "n_subjects": n_subjects,
                    "skip_bootstrap": bool(skip_bootstrap),
                    "discovery_topk": int(discovery_topk),
                    "disable_iiv": bool(disable_iiv),
                    "dose_levels": resolved_dose_levels,
                    "matlab_engine_reuse": bool(matlab_engine_reuse),
                }
            )

    if int(workers) <= 1 or len(todo_cases) <= 1:
        for case in todo_cases:
            row = _run_single_case(case)
            key = _row_key(str(row.get("pk_variant", "")), str(row.get("pd_model", "")))
            existing_map[key] = row
            rows = list(existing_map.values())
            _persist_summary(rows, csv_path, md_path)
    else:
        with cf.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            fut_map = {ex.submit(_run_single_case, case): case for case in todo_cases}
            for fut in cf.as_completed(fut_map):
                row = fut.result()
                key = _row_key(str(row.get("pk_variant", "")), str(row.get("pd_model", "")))
                existing_map[key] = row
                rows = list(existing_map.values())
                _persist_summary(rows, csv_path, md_path)

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run PK(6) x PD(13) full benchmark matrix.")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_matrix"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-subjects", type=int, default=12)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--pd-models", default="", help="Optional comma-separated PD model names")
    parser.add_argument(
        "--pk-routes",
        default="",
        help="Optional comma-separated PK routes (oral,bolus)",
    )
    parser.add_argument(
        "--pk-compartments",
        default="",
        help="Optional comma-separated PK compartment counts (1,2,3)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint-resume mode")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed entries when resuming")
    parser.add_argument(
        "--dose-design-scope",
        default="observable",
        choices=["none", "observable", "all"],
        help="Enable multi-dose (including 0-dose baseline) for none/observable-only/all PD models.",
    )
    parser.add_argument(
        "--dose-levels",
        default="0,0.2,1,5",
        help="Comma-separated dose multipliers used when dose design is enabled.",
    )
    parser.add_argument(
        "--disable-iiv",
        action="store_true",
        help="Disable inter-individual variability in data generation (omega set to zero).",
    )
    parser.add_argument(
        "--discovery-topk",
        type=int,
        default=5,
        help="Top-K structures kept by discovery and forwarded to downstream fitting.",
    )
    parser.add_argument(
        "--mechanism-hints-scope",
        default="standard",
        choices=["none", "standard", "observable", "all"],
        help="Run residual/module mechanism hinting for no models, standard non-observable models, observable models, or all small-molecule models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of case workers. workers>1 enables process-level parallel execution across PK×PD cases.",
    )
    parser.add_argument(
        "--reuse-matlab-engine",
        action="store_true",
        help="Reuse one MATLAB engine per worker process (reduces engine startup overhead).",
    )
    args = parser.parse_args()

    pd_models = [x.strip() for x in args.pd_models.split(",") if x.strip()] if args.pd_models else None
    pk_routes = [x.strip() for x in args.pk_routes.split(",") if x.strip()] if args.pk_routes else None
    pk_compartments = [int(x.strip()) for x in args.pk_compartments.split(",") if x.strip()] if args.pk_compartments else None
    dose_levels = [float(x.strip()) for x in args.dose_levels.split(",") if x.strip()]
    csv_path, md_path = run_matrix(
        out_dir=args.out_dir,
        pd_models=pd_models,
        seed=args.seed,
        n_subjects=args.n_subjects,
        skip_bootstrap=bool(args.skip_bootstrap),
        resume=(not bool(args.no_resume)),
        rerun_failed=bool(args.rerun_failed),
        pk_routes=pk_routes,
        pk_compartments=pk_compartments,
        dose_design_scope=args.dose_design_scope,
        dose_levels=dose_levels,
        discovery_topk=int(args.discovery_topk),
        disable_iiv=bool(args.disable_iiv),
        mechanism_hints_scope=args.mechanism_hints_scope,
        workers=max(1, int(args.workers)),
        matlab_engine_reuse=bool(args.reuse_matlab_engine),
    )
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
