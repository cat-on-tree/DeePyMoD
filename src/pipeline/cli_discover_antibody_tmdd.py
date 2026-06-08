import argparse
import json

from src.pipeline.auto_workflow import run_discovery_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Antibody/TMDD PD discovery: known PK model -> PK imputation -> discovery -> NLME -> report."
    )
    parser.add_argument("--input-csv", required=True, help="CSV with sid,time,R_obs and optional C_obs")
    parser.add_argument(
        "--active-model",
        default="ANTIBODY_PKPD",
        choices=["ANTIBODY_PKPD", "TMDD_BASE"],
        help="TMDD-family discovery model",
    )
    parser.add_argument("--pk-model-name", required=True, help="Known PK model used for PK imputation")
    parser.add_argument("--pk-route", choices=["oral", "bolus"], default=None)
    parser.add_argument("--pk-compartments", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--out-root", default=".")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-population-mean-for-nlme", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--nlme-mode", choices=["screen", "confirm"], default=None)
    parser.add_argument("--nlme-multistart-on-fail", action="store_true", default=True)
    parser.add_argument("--disable-nlme-fallback", action="store_true")
    parser.add_argument("--config-json", default=None, help="Optional JSON file to override discovery config")
    args = parser.parse_args()

    overrides = None
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            overrides = json.load(f)

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
        run_initial_confirmation=False,
        nlme_mode=args.nlme_mode,
        nlme_multistart_on_fail=(False if args.disable_nlme_fallback else bool(args.nlme_multistart_on_fail)),
        skip_bootstrap=bool(args.skip_bootstrap),
        discovery_overrides=overrides,
    )
    print(result["report_md"])


if __name__ == "__main__":
    main()
