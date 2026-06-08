#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash run_pkpd_pd_refinement.sh <command> [args]

Commands:
  observable [out_dir]
      Run small-molecule observable-only discovery baseline.

  residuals [out_dir] [reps] [seed]
  residuals [reps] [seed]
      Generate fixed observable-surrogate residual features.

  train-classifiers [features_csv] [out_dir]
      Train/evaluate two-stage residual classifiers.

  raw-controls [out_dir] [reps] [seed]
  raw-controls [reps] [seed]
      Compare Raw PD RF, Residual RF, and Raw+Residual RF on identical cases.

  domain-shift [out_dir] [reps] [seed]
  domain-shift [reps] [seed]
      Train on clean cases and evaluate raw/residual classifiers under nuisance shifts.

  repair-benchmark [out_dir] [reps] [seed]
  repair-benchmark [reps] [seed]
      Test whether residual-proposed mechanism modules repair H0 surrogate errors.

  analyze-repair [result_dir]
      Read repair benchmark artifacts and write diagnostic tables/recommendations.

Environment overrides:
  PYTHON_BIN, N_SUBJECTS, WORKERS, PK_ROUTE, PK_COMPARTMENTS, PD_MODELS,
  DOSE_LEVELS, SEED, TEST_SEED, SCENARIOS, DISCOVERY_TOPK
EOF
}

is_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_out_reps_seed() {
  local default_out="$1"
  shift || true
  OUT_DIR="$default_out"
  N_REPLICATES="${N_REPLICATES:-10}"
  SEED="${SEED:-42}"

  if [[ $# -ge 1 ]] && is_int "$1"; then
    N_REPLICATES="$1"
    if [[ $# -ge 2 ]]; then
      SEED="$2"
    fi
  else
    if [[ $# -ge 1 ]]; then
      OUT_DIR="$1"
    fi
    if [[ $# -ge 2 ]]; then
      N_REPLICATES="$2"
    fi
    if [[ $# -ge 3 ]]; then
      SEED="$3"
    fi
  fi
}

run_observable() {
  local out_dir="${1:-artifacts/pkpd_matrix_smallmol_observable_baseline_v6}"
  local seed="${SEED:-42}"
  local n_subjects="${N_SUBJECTS:-8}"
  local pk_route="${PK_ROUTE:-oral}"
  local pk_compartments="${PK_COMPARTMENTS:-1}"
  local pd_models="${PD_MODELS:-DIRECT_SIGEMAX,IDR_INHIB_KIN_SIG,TGI_BASIC}"
  local topk="${DISCOVERY_TOPK:-5}"
  local dose_levels="${DOSE_LEVELS:-0,0.2,1,5}"

  echo "[PKPD-OBSERVABLE] out_dir=${out_dir} seed=${seed} n_subjects=${n_subjects} pk=${pk_route}_${pk_compartments}c topk=${topk} doses=${dose_levels}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement observable \
    --out-dir "${out_dir}" \
    --seed "${seed}" \
    --n-subjects "${n_subjects}" \
    --pk-route "${pk_route}" \
    --pk-compartments "${pk_compartments}" \
    --pd-models "${pd_models}" \
    --discovery-topk "${topk}" \
    --dose-levels "${dose_levels}"
  echo "[PKPD-OBSERVABLE] done"
}

run_residuals() {
  parse_out_reps_seed "artifacts/pkpd_fast_surrogate_residuals_v3" "$@"
  local n_subjects="${N_SUBJECTS:-8}"
  local pk_route="${PK_ROUTE:-oral}"
  local pk_compartments="${PK_COMPARTMENTS:-1}"
  local pd_models="${PD_MODELS:-DIRECT_SIGEMAX,IDR_INHIB_KIN_SIG,TGI_BASIC,BIOPHASE_EMAX,BIOPHASE_SIGEMAX,TRANSDUCTION_DELAY,FEEDBACK_REGULATION,CIRCADIAN_REGULATION,DISEASE_PROGRESSION,TOLERANCE_ADAPTATION,PRECURSOR_POOL}"
  local dose_levels="${DOSE_LEVELS:-0,0.2,1,5}"
  local workers="${WORKERS:-8}"

  echo "[PKPD-RESIDUALS] out_dir=${OUT_DIR} seed=${SEED} reps=${N_REPLICATES} n_subjects=${n_subjects} pk=${pk_route}_${pk_compartments}c workers=${workers}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement residuals \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --n-replicates "${N_REPLICATES}" \
    --n-subjects "${n_subjects}" \
    --pk-route "${pk_route}" \
    --pk-compartments "${pk_compartments}" \
    --pd-models "${pd_models}" \
    --dose-levels "${dose_levels}" \
    --workers "${workers}" \
    --disable-iiv \
    --disable-quality-guard \
    --write-jsonl
  echo "[PKPD-RESIDUALS] done"
}

run_train_classifiers() {
  local features_csv="${1:-artifacts/pkpd_fast_surrogate_residuals_v3/surrogate_residual_features.csv}"
  local out_dir="${2:-artifacts/pkpd_residual_classifiers_v1}"
  local seed="${SEED:-20260607}"

  echo "[PKPD-RESIDUAL-CLASSIFIERS] features=${features_csv} out_dir=${out_dir} seed=${seed}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement train-classifiers \
    --features-csv "${features_csv}" \
    --out-dir "${out_dir}" \
    --seed "${seed}"
  echo "[PKPD-RESIDUAL-CLASSIFIERS] done"
}

run_raw_controls() {
  parse_out_reps_seed "artifacts/pkpd_raw_vs_residual_controls_v1" "$@"
  local n_subjects="${N_SUBJECTS:-8}"
  local workers="${WORKERS:-4}"

  echo "[PKPD-RAW-CONTROLS] out_dir=${OUT_DIR} seed=${SEED} reps=${N_REPLICATES} n_subjects=${n_subjects} workers=${workers}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement raw-controls \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --n-replicates "${N_REPLICATES}" \
    --n-subjects "${n_subjects}" \
    --workers "${workers}" \
    --disable-iiv \
    --disable-quality-guard
  echo "[PKPD-RAW-CONTROLS] done"
}

run_domain_shift() {
  parse_out_reps_seed "artifacts/pkpd_raw_vs_residual_domain_shift_v1" "$@"
  local test_seed="${TEST_SEED:-1042}"
  local n_subjects="${N_SUBJECTS:-8}"
  local workers="${WORKERS:-4}"
  local scenarios="${SCENARIOS:-clean_id,baseline_shift,dose_shift,sparse_sampling,noise_pk_error,iiv_shift,combined_shift}"

  echo "[PKPD-DOMAIN-SHIFT] out_dir=${OUT_DIR} seed=${SEED} test_seed=${test_seed} reps=${N_REPLICATES} n_subjects=${n_subjects} workers=${workers}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement domain-shift \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --test-seed "${test_seed}" \
    --n-replicates "${N_REPLICATES}" \
    --n-subjects "${n_subjects}" \
    --workers "${workers}" \
    --scenarios "${scenarios}" \
    --disable-quality-guard
  echo "[PKPD-DOMAIN-SHIFT] done"
}

run_repair_benchmark() {
  parse_out_reps_seed "artifacts/pkpd_mechanism_repair_v1" "$@"
  local n_subjects="${N_SUBJECTS:-8}"
  local pk_route="${PK_ROUTE:-oral}"
  local pk_compartments="${PK_COMPARTMENTS:-1}"
  local pd_models="${PD_MODELS:-DIRECT_SIGEMAX,IDR_INHIB_KIN_SIG,TGI_BASIC,BIOPHASE_EMAX,BIOPHASE_SIGEMAX,TRANSDUCTION_DELAY,FEEDBACK_REGULATION,CIRCADIAN_REGULATION,DISEASE_PROGRESSION,TOLERANCE_ADAPTATION,PRECURSOR_POOL}"
  local dose_levels="${DOSE_LEVELS:-0,0.2,1,5}"
  local workers="${WORKERS:-4}"
  local guide_mode="${GUIDE_MODE:-classifier}"
  local classifier_dir="${CLASSIFIER_DIR:-artifacts/pkpd_residual_classifiers_v1}"
  local observable_fp_penalty="${OBSERVABLE_FP_PENALTY:-50.0}"
  local gate_threshold="${GATE_THRESHOLD:-0.35}"
  local gate_penalty="${GATE_PENALTY:-200.0}"

  echo "[PKPD-REPAIR-BENCHMARK] out_dir=${OUT_DIR} seed=${SEED} reps=${N_REPLICATES} n_subjects=${n_subjects} workers=${workers} guide=${guide_mode} gate=${gate_threshold}/${gate_penalty}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement repair-benchmark \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --n-replicates "${N_REPLICATES}" \
    --n-subjects "${n_subjects}" \
    --pk-route "${pk_route}" \
    --pk-compartments "${pk_compartments}" \
    --pd-models "${pd_models}" \
    --dose-levels "${dose_levels}" \
    --workers "${workers}" \
    --guide-mode "${guide_mode}" \
    --classifier-dir "${classifier_dir}" \
    --observable-fp-penalty "${observable_fp_penalty}" \
    --gate-threshold "${gate_threshold}" \
    --gate-penalty "${gate_penalty}" \
    --disable-iiv \
    --disable-quality-guard
  echo "[PKPD-REPAIR-BENCHMARK] done"
}

run_analyze_repair() {
  local result_dir="${1:-artifacts/pkpd_mechanism_repair_specificity_v1}"
  local top_n="${TOP_N:-20}"

  echo "[PKPD-REPAIR-ANALYSIS] result_dir=${result_dir} top_n=${top_n}"
  "${PYTHON_BIN}" -m src.pipeline.cli_pd_refinement analyze-repair \
    --result-dir "${result_dir}" \
    --top-n "${top_n}"
  echo "[PKPD-REPAIR-ANALYSIS] done"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  observable)
    run_observable "$@"
    ;;
  residuals)
    run_residuals "$@"
    ;;
  train-classifiers)
    run_train_classifiers "$@"
    ;;
  raw-controls)
    run_raw_controls "$@"
    ;;
  domain-shift)
    run_domain_shift "$@"
    ;;
  repair-benchmark)
    run_repair_benchmark "$@"
    ;;
  analyze-repair)
    run_analyze_repair "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
