import argparse
import json

import pandas as pd

from src.configs.pd_module_registry import resolve_discovery_profile
from src.pipeline.auto_workflow import run_discovery_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Small-molecule PD discovery: known PK model -> PK imputation -> discovery -> NLME -> report."
    )
    parser.add_argument("--input-csv", required=True, help="CSV with sid,time,R_obs and optional C_obs")
    parser.add_argument(
        "--active-model",
        default="SMALL_MOLECULE_BLIND",
        help="Discovery label for reporting/pruning context (default: SMALL_MOLECULE_BLIND)",
    )
    parser.add_argument("--pk-model-name", required=True, help="Known PK model used for PK imputation")
    parser.add_argument("--pk-route", choices=["oral", "bolus"], default=None)
    parser.add_argument("--pk-compartments", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--out-root", default=".")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-population-mean-for-nlme", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--nlme-mode", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--nlme-multistart-on-fail", action="store_true", default=True)
    parser.add_argument("--disable-nlme-fallback", action="store_true")
    parser.add_argument("--disable-mechanism-hints", action="store_true")
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Use observable-only blind regression (no mechanism combo trial search).",
    )
    parser.add_argument(
        "--discovery-profile",
        choices=["auto", "observable_blind", "observable_surrogate", "standard_blind"],
        default="auto",
        help="Discovery profile. auto uses observable_blind when --observable-only is set, otherwise observable_surrogate.",
    )
    parser.add_argument(
        "--specified-mechanism",
        choices=["biophase", "delay", "feedback", "tolerance", "circadian", "disease", "interaction"],
        default=None,
        help="Optional explicit mechanism to judge yes/no by screening evidence",
    )
    parser.add_argument("--mechanism-confirm-only", action="store_true", help="Only output mechanism yes/no decision")
    parser.add_argument("--config-json", default=None, help="Optional JSON file to override discovery config")
    args = parser.parse_args()

    overrides = None
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    profile = args.discovery_profile
    if profile == "auto":
        profile = "observable_blind" if args.observable_only else "observable_surrogate"
    if profile in {"observable_blind", "observable_surrogate", "standard_blind"}:
        df_cols = pd.read_csv(args.input_csv, nrows=1).columns
        profile_overrides = resolve_discovery_profile(
            profile,
            available_columns=df_cols,
            pk_compartments=(int(args.pk_compartments) if args.pk_compartments is not None else None),
        )
        overrides = dict(overrides or {})
        overrides.update(profile_overrides)

    result = run_discovery_workflow(
        input_csv=args.input_csv,
        active_model=args.active_model,
        pk_model_name=args.pk_model_name,
        out_root=args.out_root,
        pk_route=args.pk_route,
        pk_compartments=(int(args.pk_compartments) if args.pk_compartments is not None else None),
        run_name=args.run_name,
        device=args.device,
        use_population_mean_for_nlme=bool(args.use_population_mean_for_nlme),
        run_initial_confirmation=True,
        nlme_mode=args.nlme_mode,
        nlme_multistart_on_fail=(False if args.disable_nlme_fallback else bool(args.nlme_multistart_on_fail)),
        skip_bootstrap=bool(args.skip_bootstrap),
        discovery_overrides=overrides,
        enable_mechanism_hints=(not bool(args.disable_mechanism_hints)),
        specified_mechanism=args.specified_mechanism,
        mechanism_confirm_only=bool(args.mechanism_confirm_only),
    )
    print(result["report_md"])


if __name__ == "__main__":
    main()
