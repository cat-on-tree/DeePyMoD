import copy
import numpy as np
import torch

from src.training.pruning import enforce_hierarchy
from src.training.trainer import NonFiniteTrainingError, run_train_loop, eval_mse, get_coeff_and_mask


def bic_from_mse(mse, k, n):
    return n * np.log(mse + 1e-12) + k * np.log(n)


def make_candidate_masks(
    base_mask_np,
    coef_np,
    term_names,
    extra_terms_max=2,
    top_inactive_pool=6,
    top_active_drop_pool=4,
):
    term_names = np.array(term_names)
    name_to_idx = {n: i for i, n in enumerate(term_names)}

    inactive_idx = np.where(~base_mask_np)[0]
    inactive_sorted = inactive_idx[np.argsort(-np.abs(coef_np[inactive_idx]))]

    candidates = [base_mask_np.copy()]

    for i in range(min(len(inactive_sorted), top_inactive_pool)):
        m = base_mask_np.copy()
        m[inactive_sorted[i]] = True
        m = enforce_hierarchy(m, name_to_idx)
        candidates.append(m)

    if extra_terms_max >= 2:
        pool = inactive_sorted[:top_inactive_pool]
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                m = base_mask_np.copy()
                m[pool[i]] = True
                m[pool[j]] = True
                m = enforce_hierarchy(m, name_to_idx)
                candidates.append(m)

    active_idx = np.where(base_mask_np)[0]
    active_sorted = active_idx[np.argsort(np.abs(coef_np[active_idx]))]
    active_pool = active_sorted[:top_active_drop_pool]

    for i in range(len(active_pool)):
        m = base_mask_np.copy()
        m[active_pool[i]] = False
        m = enforce_hierarchy(m, name_to_idx)
        if m.any():
            candidates.append(m)

    if extra_terms_max >= 2:
        for i in range(len(active_pool)):
            for j in range(i + 1, len(active_pool)):
                m = base_mask_np.copy()
                m[active_pool[i]] = False
                m[active_pool[j]] = False
                m = enforce_hierarchy(m, name_to_idx)
                if m.any():
                    candidates.append(m)

    uniq, seen = [], set()
    for m in candidates:
        key = tuple(m.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


def _is_multi_term_names(term_names):
    return isinstance(term_names, (list, tuple)) and len(term_names) > 0 and isinstance(term_names[0], (list, tuple, np.ndarray))


def _term_index_map(term_names):
    return {str(n): i for i, n in enumerate(np.array(term_names).tolist())}


def _stage_select_observable_mask(base_mask_np, coef_np, term_names, interaction_keep_ratio=0.25):
    name_to_idx = _term_index_map(term_names)
    abs_coef = np.abs(np.asarray(coef_np).reshape(-1))
    selected = np.zeros_like(base_mask_np, dtype=bool)

    def _activate(term):
        if term in name_to_idx:
            selected[name_to_idx[term]] = True

    # Structural anchors.
    _activate("1")
    _activate("R")

    # Stage 1: pick one dominant exposure basis term.
    basis_terms = ["C", "C^2", "Emax(C)", "Hill(C)"]
    basis_candidates = [t for t in basis_terms if t in name_to_idx]
    if basis_candidates:
        basis_scores = {}
        # Prefer currently active terms while still being coefficient-driven.
        for t in basis_candidates:
            i = name_to_idx[t]
            active_bonus = 0.20 if bool(base_mask_np[i]) else 0.0
            basis_scores[t] = float(abs_coef[i]) + active_bonus
        chosen_basis = max(basis_scores, key=basis_scores.get)
        _activate(chosen_basis)
    else:
        chosen_basis = None

    # Stage 2: attach matched interaction if signal is strong enough.
    interaction_map = {
        "C": "C*R",
        "C^2": None,
        "Emax(C)": "Emax(C)*R",
        "Hill(C)": "Hill(C)*R",
    }
    interaction_term = interaction_map.get(chosen_basis)
    if interaction_term in name_to_idx and chosen_basis in name_to_idx:
        i_base = name_to_idx[chosen_basis]
        i_int = name_to_idx[interaction_term]
        base_mag = float(abs_coef[i_base]) + 1e-12
        int_mag = float(abs_coef[i_int])
        if int_mag >= float(interaction_keep_ratio) * base_mag:
            _activate(interaction_term)

    # Keep strong R-only dynamics if they are clearly present.
    if "R" in name_to_idx:
        i_r = name_to_idx["R"]
        if float(abs_coef[i_r]) >= 0.08 * float(abs_coef.max() + 1e-12):
            _activate("R")

    # Optional universal time-decay basis for nonstationary growth/kill behavior.
    if "exp(-t)" in name_to_idx and "exp(-t)*R" in name_to_idx:
        i_t = name_to_idx["exp(-t)"]
        i_tr = name_to_idx["exp(-t)*R"]
        t_mag = float(abs_coef[i_t])
        tr_mag = float(abs_coef[i_tr])
        if max(t_mag, tr_mag) >= 0.22 * float(abs_coef.max() + 1e-12):
            _activate("exp(-t)")
            if tr_mag >= 0.9 * max(t_mag, 1e-12):
                _activate("exp(-t)*R")

    return selected


def _drop_forbidden_terms(mask_np, term_names, forbidden_terms=None):
    if not forbidden_terms:
        return np.asarray(mask_np, dtype=bool)
    out = np.asarray(mask_np, dtype=bool).copy()
    name_to_idx = _term_index_map(term_names)
    for t in set(forbidden_terms):
        if t in name_to_idx:
            out[name_to_idx[t]] = False
    return out


def _generate_observable_candidates(
    base_mask_np,
    coef_np,
    term_names,
    interaction_keep_ratio=0.25,
    extra_terms_max=2,
    forbidden_terms=None,
):
    term_names = np.array(term_names)
    name_to_idx = _term_index_map(term_names)
    abs_coef = np.abs(np.asarray(coef_np).reshape(-1))

    seed = _stage_select_observable_mask(
        base_mask_np=base_mask_np,
        coef_np=coef_np,
        term_names=term_names,
        interaction_keep_ratio=interaction_keep_ratio,
    )
    candidates = make_candidate_masks(
        base_mask_np=seed,
        coef_np=coef_np,
        term_names=term_names,
        extra_terms_max=extra_terms_max,
    )

    basis_terms = ["C", "C^2", "Emax(C)", "Hill(C)"]
    interaction_map = {
        "C": "C*R",
        "C^2": None,
        "Emax(C)": "Emax(C)*R",
        "Hill(C)": "Hill(C)*R",
    }
    basis_idx = [name_to_idx[t] for t in basis_terms if t in name_to_idx]
    interaction_idx = [
        name_to_idx[t]
        for t in ["C*R", "Emax(C)*R", "Hill(C)*R"]
        if t in name_to_idx
    ]

    for basis in basis_terms:
        if basis not in name_to_idx:
            continue
        m = seed.copy()
        for i in basis_idx + interaction_idx:
            m[i] = False
        m[name_to_idx[basis]] = True

        int_term = interaction_map.get(basis)
        if int_term in name_to_idx:
            i_base = name_to_idx[basis]
            i_int = name_to_idx[int_term]
            base_mag = float(abs_coef[i_base]) + 1e-12
            int_mag = float(abs_coef[i_int])
            if int_mag >= float(interaction_keep_ratio) * base_mag:
                m[i_int] = True

        m = enforce_hierarchy(m, name_to_idx)
        if m.any():
            candidates.append(m)

    if "exp(-t)" in name_to_idx:
        m_drop_time = seed.copy()
        m_drop_time[name_to_idx["exp(-t)"]] = False
        if "exp(-t)*R" in name_to_idx:
            m_drop_time[name_to_idx["exp(-t)*R"]] = False
        m_drop_time = enforce_hierarchy(m_drop_time, name_to_idx)
        if m_drop_time.any():
            candidates.append(m_drop_time)

    candidates = [_drop_forbidden_terms(m, term_names, forbidden_terms=forbidden_terms) for m in candidates]

    uniq, seen = [], set()
    for m in candidates:
        key = tuple(np.asarray(m, dtype=bool).tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(np.asarray(m, dtype=bool))
    return uniq


def _structure_prior_penalty_single(terms, required_any_terms=None):
    required_any_terms = set(required_any_terms or [])
    tset = set(terms)
    penalty = 0.0

    if required_any_terms and not (tset & required_any_terms):
        penalty += 100.0

    basis = {"C", "C^2", "Emax(C)", "Hill(C)"}
    interaction = {"C*R", "Emax(C)*R", "Hill(C)*R"}
    n_basis = len(tset & basis)
    n_inter = len(tset & interaction)

    if n_basis == 0:
        penalty += 20.0
    elif n_basis > 1:
        penalty += 6.0 * float(n_basis - 1)

    if n_inter > 1:
        penalty += 8.0 * float(n_inter - 1)

    time_terms = {"exp(-t)", "exp(-t)*R"}
    n_time = len(tset & time_terms)
    if n_time > 0:
        penalty += 2.0 * float(n_time)
        if "exp(-t)" in tset and n_inter == 0:
            penalty += 4.0

    parent_map = {"C*R": "C", "Emax(C)*R": "Emax(C)", "Hill(C)*R": "Hill(C)"}
    parent_map["exp(-t)*R"] = "exp(-t)"
    for child, parent in parent_map.items():
        if child in tset and parent not in tset:
            penalty += 12.0

    if len(tset) <= 2:
        penalty += 10.0
    return penalty


def _score_candidate(
    mse_val,
    k,
    n_samples,
    terms,
    scoring_mode="bic",
    scoring_prior_weight=1.0,
    required_any_terms=None,
):
    bic_val = bic_from_mse(mse_val, k, n_samples)
    if str(scoring_mode).lower() != "bic_prior":
        return bic_val
    if isinstance(terms, list) and terms and isinstance(terms[0], list):
        return bic_val
    penalty = _structure_prior_penalty_single(
        terms=terms if isinstance(terms, list) else [str(terms)],
        required_any_terms=required_any_terms,
    )
    return float(bic_val + float(scoring_prior_weight) * penalty)


def rank_topk_validation_bic(
    model,
    estimator,
    constraint,
    network,
    library,
    opt_nn,
    opt_lib,
    X_train,
    Y_train,
    X_val,
    Y_val,
    train_loader,
    term_names,
    topk=5,
    candidate_extra_terms_max=2,
    candidate_refit_epochs=400,
    lambda_reg=0.2,
    lambda_gamma_pen=1e-4,
    ranking_mode="candidate_search",
    required_any_terms=None,
    interaction_keep_ratio=0.25,
    scoring_mode="bic",
    scoring_prior_weight=1.0,
    forbidden_terms=None,
):
    is_multi = _is_multi_term_names(term_names)
    required_any_terms = set(required_any_terms or [])
    forbidden_terms = set(forbidden_terms or [])

    if not is_multi:
        base_coeff, base_mask = get_coeff_and_mask(model)
        base_mask_np = base_mask.detach().cpu().numpy().astype(bool)
        coef_np = base_coeff.detach().cpu().numpy()

        mode = str(ranking_mode).lower()
        if mode == "base_mask_only":
            candidates = [base_mask_np.copy()]
        elif mode == "two_stage_observable":
            candidates = _generate_observable_candidates(
                base_mask_np=base_mask_np,
                coef_np=coef_np,
                term_names=term_names,
                interaction_keep_ratio=interaction_keep_ratio,
                extra_terms_max=candidate_extra_terms_max,
                forbidden_terms=forbidden_terms,
            )
        else:
            candidates = make_candidate_masks(
                base_mask_np, coef_np, term_names,
                extra_terms_max=candidate_extra_terms_max
            )
            if forbidden_terms:
                candidates = [
                    _drop_forbidden_terms(m, term_names, forbidden_terms=forbidden_terms)
                    for m in candidates
                ]
    else:
        base_coeffs = model.constraint_coeffs(sparse=True, scaled=False)
        base_masks = constraint.sparsity_masks
        base_masks_np = [m.detach().cpu().numpy().astype(bool) for m in base_masks]
        candidates = []
        # generate candidates per-equation while keeping others fixed
        for eq_idx, (coef, mask_np, names) in enumerate(zip(base_coeffs, base_masks_np, term_names)):
            coef_np = coef.detach().cpu().numpy().flatten()
            eq_candidates = make_candidate_masks(
                mask_np, coef_np, names,
                extra_terms_max=candidate_extra_terms_max
            )
            for cand in eq_candidates:
                full = [m.copy() for m in base_masks_np]
                full[eq_idx] = cand
                candidates.append(full)

        # de-duplicate
        uniq, seen = [], set()
        for mlist in candidates:
            key = tuple(tuple(m.tolist()) for m in mlist)
            if key not in seen:
                seen.add(key)
                uniq.append(mlist)
        candidates = uniq

    state_snapshot = {
        "net": copy.deepcopy(network.state_dict()),
        "lib": copy.deepcopy(library.state_dict()),
        "opt_nn": copy.deepcopy(opt_nn.state_dict()),
        "opt_lib": copy.deepcopy(opt_lib.state_dict()),
    }

    def restore():
        network.load_state_dict(copy.deepcopy(state_snapshot["net"]))
        library.load_state_dict(copy.deepcopy(state_snapshot["lib"]))
        opt_nn.load_state_dict(copy.deepcopy(state_snapshot["opt_nn"]))
        opt_lib.load_state_dict(copy.deepcopy(state_snapshot["opt_lib"]))

    results = []
    for mask_np in candidates:
        restore()

        if not is_multi:
            m = torch.tensor(mask_np, dtype=torch.bool, device=next(model.parameters()).device)
            estimator.set_mask(m)
            constraint.sparsity_masks = [m]
        else:
            m = [torch.tensor(mm, dtype=torch.bool, device=next(model.parameters()).device) for mm in mask_np]
            estimator.set_mask(m)
            constraint.sparsity_masks = m

        try:
            run_train_loop(
                model, train_loader, opt_nn, opt_lib, n_epochs=candidate_refit_epochs,
                lambda_reg=lambda_reg, lambda_gamma_pen=lambda_gamma_pen
            )
        except NonFiniteTrainingError:
            continue

        mse_tr = eval_mse(model, X_train, Y_train)
        mse_va = eval_mse(model, X_val, Y_val)
        if not np.isfinite(mse_tr) or not np.isfinite(mse_va):
            continue

        if not is_multi:
            k = int(mask_np.sum())
            terms = np.array(term_names)[mask_np].tolist()
            if forbidden_terms and (set(terms) & forbidden_terms):
                continue
            if required_any_terms and not (set(terms) & required_any_terms):
                continue
        else:
            k = int(sum(int(mm.sum()) for mm in mask_np))
            terms = [np.array(tn)[mm].tolist() for tn, mm in zip(term_names, mask_np)]

        score_val = _score_candidate(
            mse_val=mse_va,
            k=k,
            n_samples=X_val.shape[0],
            terms=terms,
            scoring_mode=scoring_mode,
            scoring_prior_weight=scoring_prior_weight,
            required_any_terms=required_any_terms,
        )
        if not np.isfinite(score_val):
            continue

        results.append({
            "terms": terms,
            "k": k,
            "mse_train": mse_tr,
            "mse_val": mse_va,
            "score": score_val,  # lower is better
            "mask": mask_np.copy() if not is_multi else [m.copy() for m in mask_np],
        })

    if not results:
        raise RuntimeError("No finite candidate remained after ranking.")

    results = sorted(results, key=lambda x: x["score"])
    top_results = results[:topk]
    best = top_results[0]

    if not is_multi:
        best_mask_t = torch.tensor(best["mask"], dtype=torch.bool, device=next(model.parameters()).device)
        estimator.set_mask(best_mask_t)
        constraint.sparsity_masks = [best_mask_t]
    else:
        best_mask_t = [torch.tensor(m, dtype=torch.bool, device=next(model.parameters()).device) for m in best["mask"]]
        estimator.set_mask(best_mask_t)
        constraint.sparsity_masks = best_mask_t

    return best, top_results
