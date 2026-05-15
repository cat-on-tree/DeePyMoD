# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeePyMoD is a PyTorch-based implementation of the DeepMoD (Deep Model Discovery) algorithm for discovering governing PDEs from observational data. The repository extends this core capability with a **PK/PD (pharmacokinetic/pharmacodynamic) modeling application** layer.

The codebase has **two layers**:
1. **Core DeepMoD library** (`src/deepymod/`) - the algorithmic framework for sparse model discovery
2. **PK/PD application layer** (`src/` except `deepymod/`) - domain-specific models for drug response discovery

## Architecture

### Core DeepMoD (`src/deepymod/model/deepmod.py`)

DeepMoD integrates four components into a single pipeline:

```
DeepMoD = FunctionApproximator (NN) + Library + SparsityEstimator + Constraint
```

- **FunctionApproximator**: Neural network that approximates the dynamical field and computes derivatives via autodiff
- **Library**: Builds the feature matrix (theta) from network predictions - candidate terms for the discovered equation
- **SparsityEstimator**: Applies sparse regression (e.g., thresholding) to identify active terms
- **Constraint**: Constrains NN output by fitting coefficients via least-squares against the library

The training loop iteratively prunes terms and refits coefficients until convergence.

### PK/PD Application Layer

Located in `src/`, this layer applies DeepMoD to drug response modeling:

| Module | Purpose |
|--------|---------|
| `models/network.py` | `TimeOnlyNet` - NN that predicts response R(t) from time |
| `models/library.py` | `PDLibraryExpanded` - PK/PD term library (Emax, Hill, etc.) |
| `models/constraint.py` | `RidgeConstraint` - Ridge regression for coefficient fitting |
| `models/estimator.py` | `FixedMaskEstimator` - Fixed sparsity mask during training |
| `pipeline/discovery.py` | `run_single_discovery()` - main workflow: warmup → pruning → top-k ranking |
| `training/ranking.py` | Top-k candidate generation with validation BIC scoring |
| `configs/pkpd_registry.py` | Model configurations (DIRECT_LINEAR, IDR_INHIB_KIN_SIG, etc.) |
| `data/simulate_pkpd.py` | Population data generation with IIV (inter-individual variability) |

### Library Terms

`PDLibraryExpanded` creates these candidate terms for PK/PD model discovery:
```
["1", "R", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"]
```
Where R = response, C = concentration.

## Commands

### Installation
```bash
pip install -e .
```

### Run Tests
```bash
# All tests
pytest tests/

# Single test file
pytest tests/burgers.py
```

### Run Discovery Pipeline
```python
from src.pipeline.discovery import run_single_discovery
from src.configs.defaults import DEFAULTS
from src.data.simulate_pkpd import generate_population_data

pop_data, subject_params, cfg = generate_population_data("IDR_INHIB_KIN_SIG")
results = run_single_discovery(pop_data, "IDR_INHIB_KIN_SIG", DEFAULTS)
```

### Core Library Tests
Tests in `tests/burgers.py` demonstrate the base DeepMoD algorithm on Burgers equation using:
- `deepymod.model.func_approx.NN`
- `deepymod.model.library.Library1D`
- `deepymod.model.constraint.LeastSquares`
- `deepymod.model.sparse_estimators.Threshold`

## Key Files

- `src/deepymod/model/deepmod.py` - Core DeepMoD class (Constraint, Estimator, Library, DeepMoD)
- `src/pipeline/discovery.py` - Main PK/PD discovery entry point
- `src/training/ranking.py` - Top-k candidate ranking with validation BIC
- `src/configs/defaults.py` - Hyperparameter defaults for discovery
- `src/models/library.py` - PK/PD term library definition

## Hyperparameters

Discovery pipeline defaults (`src/configs/defaults.py`):
- `n_epochs_warmup=1800` - Initial NN training
- `n_epochs_prune=700` - Retraining after each prune round
- `max_prune_rounds=10` - Maximum pruning iterations
- `rel_thr_main=0.10`, `rel_thr_interaction=0.15` - Pruning thresholds
- `topk=5` - Number of top candidates to return