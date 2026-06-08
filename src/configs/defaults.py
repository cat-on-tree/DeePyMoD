from src.configs.pd_atom_registry import (
    H0_EMPIRICAL_TERMS,
    H0_MECHANISTIC_TERMS,
    H0_MINIMAL_TERMS,
)


DEFAULTS = dict(
    # train
    n_epochs_warmup=1800,
    n_epochs_prune=700,
    max_prune_rounds=10,
    stable_rounds_required=2,
    lr_nn=1e-3,
    lr_lib=3e-4,
    lambda_reg=0.2,
    lambda_gamma_pen=1e-4,
    ridge_lam=1e-4,

    # prune
    rel_thr_main=0.10,
    rel_thr_interaction=0.15,
    min_terms_keep=2,

    # ranking
    topk=5,
    candidate_extra_terms_max=2,
    candidate_refit_epochs=400,
    ranking_mode="candidate_search",
    required_any_terms=None,
    interaction_keep_ratio=0.25,
    scoring_mode="bic",
    scoring_prior_weight=1.0,
    force_single_exposure_basis=False,
    mandatory_any_terms=None,
    train_frac=0.7,
    module_combo=None,
    auto_module_for_model=False,
    strict_blind_discovery=True,
    baseline_candidate_terms=["1", "cos24", "sin24"],
    baseline_max_correction_ratio=0.25,
    train_bias_penalty=3.0,
    train_temporal_penalty=2.0,
    multistart_lambda_stability=1.0,
    multistart_lambda_temporal=3.0,
    multistart_lambda_late_overpredict=10.0,
    multistart_lambda_identifiability=3.0,
    multistart_lambda_boundary=6.0,
    multistart_lambda_nonconverged=120.0,
    multistart_direct_theta_abs_bound=400.0,
    nlme_struct_theta_abs_bound=400.0,
    hill_gamma_grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    hidden_hinting_mode="residual_only",
    use_fixed_surrogate_protocol=True,
    surrogate_minimal_terms=list(H0_MINIMAL_TERMS),
    surrogate_mechanistic_terms=list(H0_MECHANISTIC_TERMS),
    surrogate_empirical_terms=list(H0_EMPIRICAL_TERMS),
    surrogate_full_terms=list(H0_EMPIRICAL_TERMS),
    surrogate_h0_poor_fit_threshold=0.08,
    surrogate_h0_structure_threshold=0.20,
    surrogate_h0_empirical_gain_max=0.25,
    surrogate_poor_fit_threshold=0.08,
    surrogate_residual_structure_threshold=0.35,
    enable_hidden_state_confirmation=False,
    use_fast_hidden_confirmation=True,
    fast_confirm_max_groups=4,
    fast_confirm_restarts=4,
    hidden_confirmation_topn=2,
    hidden_confirmation_min_probability=0.0,
    hidden_confirmation_max_combos=2,
    hidden_confirmation_delta_bic_threshold=0.03,
    hidden_confirmation_delta_mse_threshold=0.02,
    hidden_confirmation_t_bic_gate=2.0,
    hidden_confirmation_t_bic_supported=2.0,
    enable_hidden_bootstrap_calibration=True,
    hidden_bootstrap_reps=11,
    hidden_bootstrap_alpha=0.10,
    hidden_bootstrap_seed=20260603,
    hidden_bootstrap_warmup_epochs=120,
    hidden_bootstrap_prune_epochs=50,
    hidden_bootstrap_prune_rounds=2,
    hidden_bootstrap_refit_epochs=40,
)

TERM_NAMES = list(H0_EMPIRICAL_TERMS)

EXPECTED_MIN_TERMS = {
    "IDR_BASE": {"1", "R"},
    "IDR_INHIB_KIN_SIG": {"1", "R", "Hill(C)"},
    "IDR_STIM_KIN_SIG": {"1", "R", "Hill(C)"},
    "IDR_INHIB_KOUT_SIG": {"1", "R", "Hill(C)*R"},
    "IDR_STIM_KOUT_SIG": {"1", "R", "Hill(C)*R"},
    "DIRECT_LINEAR": {"1", "C"},
    "DIRECT_EMAX": {"1", "Emax(C)"},
    "DIRECT_SIGEMAX": {"1", "Hill(C)"},
    "BIOPHASE_EMAX": {"1", "Emax(C)"},
    "BIOPHASE_SIGEMAX": {"1", "Hill(C)"},
}

# Explicit TMDD contract defaults for discovery demos/use (no behavior change by themselves).
TMDD_DISCOVERY_DEFAULTS = dict(
    tmdd_include_hill=True,
    tmdd_include_c_term=True,
    tmdd_include_c_r_term=True,
    tmdd_enforce_safe_r_terms=True,
    topk=5,
)

# Explicit TMDD NLME contract defaults for downstream validation/export.
TMDD_NLME_DEFAULTS = dict(
    nlme_mode="screen",
    nlme_multistart_on_fail=True,
)
