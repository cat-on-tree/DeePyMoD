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
    train_frac=0.7,
)

TERM_NAMES = ["1", "R", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"]

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