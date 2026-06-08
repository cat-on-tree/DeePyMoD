import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from src.data.simulate_pkpd import generate_population_data


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    try:
        import numpy as np

        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:
        pass
    return x


def main():
    parser = argparse.ArgumentParser(description="Generate PKPD simulation data by model args.")
    parser.add_argument("--model-name", required=True, help="Model name in MODEL_REGISTRY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-subjects", type=int, default=12)
    parser.add_argument("--extra-pk-iiv-sigma", type=float, default=0.0)
    parser.add_argument("--pk-route", choices=["oral", "bolus"], default="oral")
    parser.add_argument("--pk-compartments", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--disable-iiv", action="store_true", help="Disable inter-individual variability.")
    parser.add_argument("--dose-design-enabled", action="store_true")
    parser.add_argument(
        "--dose-levels",
        default="0,0.2,1,5",
        help="Comma-separated dose multipliers when --dose-design-enabled is set.",
    )
    parser.add_argument("--observable-enhanced-design", action="store_true")
    parser.add_argument(
        "--observable-dose-levels",
        default="0,0.2,1,5",
        help="Deprecated alias for --dose-levels; kept for compatibility.",
    )
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "generated"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_id = args.run_name or f"{args.model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    dose_levels_primary = [float(x.strip()) for x in args.dose_levels.split(",") if x.strip()]
    dose_levels_alias = [float(x.strip()) for x in args.observable_dose_levels.split(",") if x.strip()]
    if "--dose-levels" in sys.argv:
        dose_levels = dose_levels_primary
    elif "--observable-dose-levels" in sys.argv:
        dose_levels = dose_levels_alias
    else:
        dose_levels = dose_levels_primary
    pop_data, subject_params, cfg, pk_scale = generate_population_data(
        model_name=args.model_name,
        seed=args.seed,
        n_subjects=args.n_subjects,
        extra_pk_iiv_sigma=args.extra_pk_iiv_sigma,
        return_pk_scale=True,
        pk_route=args.pk_route,
        pk_compartments=int(args.pk_compartments),
        dose_design_enabled=bool(args.dose_design_enabled) or bool(args.observable_enhanced_design),
        dose_levels=dose_levels,
        observable_enhanced_design=bool(args.observable_enhanced_design),
        observable_dose_levels=dose_levels,
        disable_iiv=bool(args.disable_iiv),
    )

    df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
    df["sid"] = df["sid"].astype(int)
    data_csv = os.path.join(run_dir, "pkpd_long.csv")
    df.to_csv(data_csv, index=False)

    subject_json = os.path.join(run_dir, "subject_params.json")
    cfg_json = os.path.join(run_dir, "model_cfg.json")
    meta_json = os.path.join(run_dir, "meta.json")

    with open(subject_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(subject_params), f, ensure_ascii=False, indent=2)
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(cfg), f, ensure_ascii=False, indent=2)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "seed": args.seed,
                "n_subjects": args.n_subjects,
                "extra_pk_iiv_sigma": args.extra_pk_iiv_sigma,
                "pk_route": args.pk_route,
                "pk_compartments": int(args.pk_compartments),
                "dose_design_enabled": bool(args.dose_design_enabled) or bool(args.observable_enhanced_design),
                "dose_levels": dose_levels,
                "observable_enhanced_design": bool(args.observable_enhanced_design),
                "observable_dose_levels": dose_levels,
                "disable_iiv": bool(args.disable_iiv),
                "pk_scale_by_sid": _json_safe(pk_scale),
                "data_csv": data_csv,
                "subject_params_json": subject_json,
                "model_cfg_json": cfg_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(data_csv)


if __name__ == "__main__":
    main()
