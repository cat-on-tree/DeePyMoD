# CLAUDE.md

This file provides guidance for coding agents working on this repository.

## 1. Project status and scope

DeePyMoD is used here as a **residual-guided PD mechanism refinement engine**:

1. Core engine: `src/deepymod/` (sparse equation discovery).
2. PD application layer: `src/` (simulation, discovery workflow, residual diagnostics, mechanism hinting, classifier controls).
3. PK is treated as a known/external exposure driver (`C_obs`), not as the current fitting target.

Current strategy is:

- **Small molecule**: strict blind discovery/refinement by model components (no true PD form leakage).
- **TMDD / antibody**: dedicated known-mechanism path.
- **Residual-guided refinement**: observable surrogate H0 rejection, hidden-mechanism hinting, and mechanism confirmation.
- **Mechanism hinting**: screening-level probability ranking (biophase/delay/feedback/tolerance/circadian/disease/precursor/interaction).

## 2. Modeling principles (important)

### 2.1 Small-molecule strict blind mode

- Default should not leak true model family labels into pruning/selection.
- Discovery should be driven by model components (e.g., `1, R, C, C*R, Hill(C)` etc.) and data evidence.
- H0 observable surrogate terms and hidden mechanism terms must stay separated.

### 2.2 PD atom policy

- Central atom definitions live in `src/configs/pd_atom_registry.py`.
- H0 observable atoms are allowed to explain observable-sufficient PD:
  - core: `1`, `R`, `R^2`, `C`, `Emax(C)`, `Hill(C)`, `C*R`, `Emax(C)*R`, `Hill(C)*R`
  - empirical/compatibility: `C^2`, time-modulated terms, `exp(-t)`, `exp(-t)*R`
- Hidden modules are not allowed in H0:
  - `biophase`, `delay`, `feedback`, `tolerance`, `circadian`, `disease`, `precursor`, `interaction`
- Residual classifier output should be treated as a mechanism proposal, then checked by confirmation/refinement.

### 2.3 TMDD / antibody as separate track

- TMDD and antibody use dedicated library (`src/models/library_antibody.py`) and are not mixed into generic small-molecule blind flow.

### 2.4 DDI hinting policy

- DDI is always a **hintable** mechanism.
- Priority is evidence-dependent:
  - `none` (no co-med evidence): low priority
  - `coadmin` (co-med fields detected): medium priority
  - `cint` (`C_int_obs` with signal): high priority

## 3. Current pipeline architecture

### 3.1 Core workflow entry

- `src/pipeline/auto_workflow.py` → `run_discovery_workflow(...)`

This handles:

1. PK imputation from known PK model
2. blind/mechanism-aware discovery
3. optional multistart initial confirmation
4. NLME validation + diagnostics
5. mechanism hinting + optional mechanism yes/no confirmation
6. LLM handoff payload generation

### 3.2 Discovery engine

- `src/pipeline/discovery.py` → `run_single_discovery(...)`
- `src/training/pruning.py` handles hierarchy + protected terms.
- strict blind behavior is controlled by `DEFAULTS` (`strict_blind_discovery=True`).

### 3.3 Input preprocessing

- `src/data/preprocess_pkpd.py` supports:
  - required: `sid,time,C_obs,R_obs`
  - optional: `Ct_obs`, `C_int_obs`

### 3.4 Mechanism hinting and confirmation

- `src/pipeline/mechanism_hints.py`
  - `score_hidden_mechanisms(...)`
  - `confirm_mechanism_presence(...)`
  - `build_fixed_surrogate_protocol_evidence(...)`

### 3.5 Residual classifier and controls

- Public Python entry: `src/pipeline/cli_pd_refinement.py`.
- Implementation/compatibility modules:
  - `src/pipeline/cli_fast_surrogate_residuals.py` - generate fixed-surrogate residual features.
  - `src/pipeline/cli_train_residual_classifiers.py` - two-stage residual classifier:
    - Stage 1: observable vs hidden H0 rejection.
    - Stage 2: hidden-type classification among true/predicted hidden cases.
  - `src/pipeline/cli_compare_raw_residual_baselines.py` - Raw PD vs Residual vs Raw+Residual same-distribution controls.
  - `src/pipeline/cli_compare_domain_shift_baselines.py` - clean-trained domain-shift controls.
  - `src/pipeline/cli_mechanism_repair_benchmark.py` - residual-guided mechanism repair/refinement benchmark.

### 3.6 LLM interface (orchestration layer, not numerical solver)

- `src/agent/llm_interface.py`
  - `build_llm_agent_payload(...)`

LLM is used for planning/orchestration/hypothesis management, not direct numerical fitting.

## 4. Key scripts and commands

### 4.1 Install

```bash
pip install -e .
```

### 4.2 Single-run small molecule discovery

```bash
python -m src.pipeline.cli_discover_small_molecule \
  --input-csv <csv> \
  --pk-model-name <known_pk_model> \
  --active-model SMALL_MOLECULE_BLIND
```

Optional mechanism confirmation mode:

```bash
python -m src.pipeline.cli_discover_small_molecule \
  --input-csv <csv> \
  --pk-model-name <known_pk_model> \
  --specified-mechanism biophase \
  --mechanism-confirm-only
```

### 4.3 Unified experiment wrapper

Use the single root wrapper instead of multiple historical `run_*.sh` files:

```bash
bash run_pkpd_pd_refinement.sh <command> [args]
```

Commands:

```bash
# Observable small-molecule baseline
bash run_pkpd_pd_refinement.sh observable artifacts/pkpd_matrix_smallmol_observable_baseline_v6

# Generate residual features; shorthand form means reps=10, seed=42
bash run_pkpd_pd_refinement.sh residuals 10 42

# Train two-stage residual classifiers
bash run_pkpd_pd_refinement.sh train-classifiers \
  artifacts/pkpd_fast_surrogate_residuals_v3/surrogate_residual_features.csv \
  artifacts/pkpd_residual_classifiers_v1

# Same-distribution Raw PD vs Residual vs Raw+Residual controls
bash run_pkpd_pd_refinement.sh raw-controls artifacts/pkpd_raw_vs_residual_controls_v1 10 42

# Domain-shift controls
bash run_pkpd_pd_refinement.sh domain-shift artifacts/pkpd_raw_vs_residual_domain_shift_v1 10 42

# Residual-guided mechanism repair/refinement benchmark
bash run_pkpd_pd_refinement.sh repair-benchmark artifacts/pkpd_mechanism_repair_v1 10 42

# Analyze repair benchmark specificity diagnostics
bash run_pkpd_pd_refinement.sh analyze-repair artifacts/pkpd_mechanism_repair_v1
```

Set `PYTHON_BIN=/path/to/python` when the shell default Python is not the `deepymod` environment.
Set `GUIDE_MODE=classifier|frozen-classifier|rule|oracle` for repair benchmark guide ablations; default is `classifier`.
Set `CLASSIFIER_DIR=artifacts/pkpd_residual_classifiers_v1` when using `GUIDE_MODE=frozen-classifier`.
Set `GATE_THRESHOLD` and `GATE_PENALTY` to tune mechanism-specific residual evidence gates in repair specificity scoring; default penalty is intentionally strong for development screening.

Direct Python equivalent:

```bash
python -m src.pipeline.cli_pd_refinement <command> [options]
```

Use `python -m src.pipeline.cli_pd_refinement <command> --help` for command-specific flags.

### 4.4 Matrix scans

- Direct CLI: `python -m src.pipeline.run_pkpd_matrix ...`
- Prefer the unified wrapper for maintained PD refinement experiments.

## 5. Important files

- `src/pipeline/auto_workflow.py` - end-to-end orchestration.
- `src/pipeline/discovery.py` - DeepMoD-based structure discovery loop.
- `src/pipeline/mechanism_hints.py` - mechanism probability ranking and out-of-library hints.
- `src/pipeline/cli_pd_refinement.py` - maintained Python CLI for PD refinement experiments.
- `src/pipeline/cli_mechanism_repair_benchmark.py` - H0 residual repair and mechanism specificity benchmark.
- `src/configs/pd_atom_registry.py` - central PD atom/module definitions.
- `src/configs/pd_module_registry.py` - module combinations and compatibility profile resolver.
- `src/training/pruning.py` - pruning rules and blind-safe protected terms.
- `src/models/library.py` - generic and modular library terms.
- `src/models/library_antibody.py` - TMDD/antibody dedicated terms.
- `src/pipeline/pk_imputation.py` - known-PK-driven exposure imputation.
- `src/agent/llm_interface.py` - LLM handoff payload schema.
- `run_pkpd_pd_refinement.sh` - maintained root wrapper for current experiments.

## 6. Editing guidelines for agents in this repo

1. Keep small-molecule flow blind by default.
2. Do not merge TMDD/antibody logic into generic library.
3. Add/modify PD atoms through `src/configs/pd_atom_registry.py` first.
4. Keep numerical engine deterministic and reproducible.
5. LLM-related additions should be schema-driven and side-effect free.
6. Avoid re-introducing heavy report skeleton abstractions; keep workflow outputs lean.
