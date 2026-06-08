import argparse
import importlib
import sys


DEFAULT_PD_MODELS = (
    "DIRECT_SIGEMAX,IDR_INHIB_KIN_SIG,TGI_BASIC,"
    "BIOPHASE_EMAX,BIOPHASE_SIGEMAX,TRANSDUCTION_DELAY,"
    "FEEDBACK_REGULATION,CIRCADIAN_REGULATION,DISEASE_PROGRESSION,"
    "TOLERANCE_ADAPTATION,PRECURSOR_POOL"
)


def _run_module_main(module_name, argv):
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        module.main()
    finally:
        sys.argv = old_argv


def _add_common_generated_case_args(parser, default_reps=10, default_workers=4):
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pd-models", default=DEFAULT_PD_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicates", type=int, default=default_reps)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    parser.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dose-levels", default="0,0.2,1,5")
    parser.add_argument("--workers", type=int, default=default_workers)


def _bool_flag(enabled, flag):
    return [flag] if bool(enabled) else []


def _cmd_observable(args):
    argv = [
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
        "--n-subjects", str(args.n_subjects),
        "--skip-bootstrap",
        "--no-resume",
        "--disable-iiv",
        "--pk-routes", args.pk_route,
        "--pk-compartments", str(args.pk_compartments),
        "--pd-models", args.pd_models,
        "--discovery-topk", str(args.discovery_topk),
        "--dose-design-scope", "observable",
        "--dose-levels", args.dose_levels,
        "--mechanism-hints-scope", "observable",
    ]
    _run_module_main("src.pipeline.run_pkpd_matrix", argv)


def _cmd_residuals(args):
    argv = [
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
        "--n-replicates", str(args.n_replicates),
        "--n-subjects", str(args.n_subjects),
        "--pk-route", args.pk_route,
        "--pk-compartments", str(args.pk_compartments),
        "--pd-models", args.pd_models,
        "--dose-levels", args.dose_levels,
        "--workers", str(args.workers),
        *_bool_flag(args.disable_iiv, "--disable-iiv"),
        *_bool_flag(args.disable_quality_guard, "--disable-quality-guard"),
        *_bool_flag(args.write_jsonl, "--write-jsonl"),
    ]
    _run_module_main("src.pipeline.cli_fast_surrogate_residuals", argv)


def _cmd_train_classifiers(args):
    argv = [
        "--features-csv", args.features_csv,
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
    ]
    _run_module_main("src.pipeline.cli_train_residual_classifiers", argv)


def _cmd_raw_controls(args):
    argv = [
        "--out-dir", args.out_dir,
        "--pd-models", args.pd_models,
        "--seed", str(args.seed),
        "--n-replicates", str(args.n_replicates),
        "--n-subjects", str(args.n_subjects),
        "--pk-route", args.pk_route,
        "--pk-compartments", str(args.pk_compartments),
        "--dose-levels", args.dose_levels,
        "--workers", str(args.workers),
        *_bool_flag(args.disable_iiv, "--disable-iiv"),
        *_bool_flag(args.disable_quality_guard, "--disable-quality-guard"),
    ]
    _run_module_main("src.pipeline.cli_compare_raw_residual_baselines", argv)


def _cmd_domain_shift(args):
    argv = [
        "--out-dir", args.out_dir,
        "--pd-models", args.pd_models,
        "--seed", str(args.seed),
        "--test-seed", str(args.test_seed),
        "--n-replicates", str(args.n_replicates),
        "--n-subjects", str(args.n_subjects),
        "--pk-route", args.pk_route,
        "--pk-compartments", str(args.pk_compartments),
        "--train-dose-levels", args.dose_levels,
        "--shifted-dose-levels", args.shifted_dose_levels,
        "--workers", str(args.workers),
        "--scenarios", args.scenarios,
        *_bool_flag(args.disable_quality_guard, "--disable-quality-guard"),
    ]
    _run_module_main("src.pipeline.cli_compare_domain_shift_baselines", argv)


def _cmd_repair_benchmark(args):
    argv = [
        "--out-dir", args.out_dir,
        "--pd-models", args.pd_models,
        "--seed", str(args.seed),
        "--n-replicates", str(args.n_replicates),
        "--n-subjects", str(args.n_subjects),
        "--pk-route", args.pk_route,
        "--pk-compartments", str(args.pk_compartments),
        "--dose-levels", args.dose_levels,
        "--workers", str(args.workers),
        "--guide-mode", args.guide_mode,
        "--classifier-dir", args.classifier_dir,
        "--observable-fp-penalty", str(args.observable_fp_penalty),
        "--gate-threshold", str(args.gate_threshold),
        "--gate-penalty", str(args.gate_penalty),
        *_bool_flag(args.disable_iiv, "--disable-iiv"),
        *_bool_flag(args.disable_quality_guard, "--disable-quality-guard"),
    ]
    _run_module_main("src.pipeline.cli_mechanism_repair_benchmark", argv)


def _cmd_analyze_repair(args):
    argv = [
        "--result-dir", args.result_dir,
        "--top-n", str(args.top_n),
    ]
    _run_module_main("src.pipeline.cli_analyze_repair_benchmark", argv)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified CLI for residual-guided PD mechanism refinement experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("observable", help="Run observable small-molecule discovery baseline.")
    p.add_argument("--out-dir", default="artifacts/pkpd_matrix_smallmol_observable_baseline_v6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-subjects", type=int, default=8)
    p.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    p.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--pd-models", default="DIRECT_SIGEMAX,IDR_INHIB_KIN_SIG,TGI_BASIC")
    p.add_argument("--discovery-topk", type=int, default=5)
    p.add_argument("--dose-levels", default="0,0.2,1,5")
    p.set_defaults(func=_cmd_observable)

    p = sub.add_parser("residuals", help="Generate fixed observable-surrogate residual features.")
    _add_common_generated_case_args(p, default_reps=1, default_workers=8)
    p.add_argument("--disable-iiv", action="store_true", default=True)
    p.add_argument("--disable-quality-guard", action="store_true", default=True)
    p.add_argument("--write-jsonl", action="store_true", default=True)
    p.set_defaults(func=_cmd_residuals)

    p = sub.add_parser("train-classifiers", help="Train/evaluate two-stage residual classifiers.")
    p.add_argument("--features-csv", default="artifacts/pkpd_fast_surrogate_residuals_v3/surrogate_residual_features.csv")
    p.add_argument("--out-dir", default="artifacts/pkpd_residual_classifiers_v1")
    p.add_argument("--seed", type=int, default=20260607)
    p.set_defaults(func=_cmd_train_classifiers)

    p = sub.add_parser("raw-controls", help="Compare Raw PD, Residual, and Raw+Residual same-distribution controls.")
    _add_common_generated_case_args(p, default_reps=10, default_workers=4)
    p.add_argument("--disable-iiv", action="store_true", default=True)
    p.add_argument("--disable-quality-guard", action="store_true", default=True)
    p.set_defaults(func=_cmd_raw_controls)

    p = sub.add_parser("domain-shift", help="Evaluate clean-trained classifiers under nuisance/domain shifts.")
    _add_common_generated_case_args(p, default_reps=10, default_workers=4)
    p.add_argument("--test-seed", type=int, default=1042)
    p.add_argument("--shifted-dose-levels", default="0,0.1,0.7,3,8")
    p.add_argument("--scenarios", default="clean_id,baseline_shift,dose_shift,sparse_sampling,noise_pk_error,iiv_shift,combined_shift")
    p.add_argument("--disable-quality-guard", action="store_true", default=True)
    p.set_defaults(func=_cmd_domain_shift)

    p = sub.add_parser("repair-benchmark", help="Run residual-guided mechanism repair/refinement benchmark.")
    _add_common_generated_case_args(p, default_reps=10, default_workers=4)
    p.add_argument("--disable-iiv", action="store_true", default=True)
    p.add_argument("--disable-quality-guard", action="store_true", default=True)
    p.add_argument("--guide-mode", default="classifier", choices=["classifier", "frozen-classifier", "frozen_classifier", "rule", "oracle"])
    p.add_argument("--classifier-dir", default="artifacts/pkpd_residual_classifiers_v1")
    p.add_argument("--observable-fp-penalty", type=float, default=50.0)
    p.add_argument("--gate-threshold", type=float, default=0.35)
    p.add_argument("--gate-penalty", type=float, default=200.0)
    p.set_defaults(func=_cmd_repair_benchmark)

    p = sub.add_parser("analyze-repair", help="Analyze repair benchmark artifacts and suggest model-atom refinements.")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.set_defaults(func=_cmd_analyze_repair)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
