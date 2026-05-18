import copy
import numpy as np
import torch

from src.training.pruning import enforce_hierarchy
from src.training.trainer import run_train_loop, eval_mse, get_coeff_and_mask


def bic_from_mse(mse, k, n):
    return n * np.log(mse + 1e-12) + k * np.log(n)


def make_candidate_masks(base_mask_np, coef_np, term_names, extra_terms_max=2, top_inactive_pool=6):
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

    uniq, seen = [], set()
    for m in candidates:
        key = tuple(m.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


def _is_multi_term_names(term_names):
    return isinstance(term_names, (list, tuple)) and len(term_names) > 0 and isinstance(term_names[0], (list, tuple, np.ndarray))


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
):
    is_multi = _is_multi_term_names(term_names)

    if not is_multi:
        base_coeff, base_mask = get_coeff_and_mask(model)
        base_mask_np = base_mask.detach().cpu().numpy().astype(bool)
        coef_np = base_coeff.detach().cpu().numpy()

        candidates = make_candidate_masks(
            base_mask_np, coef_np, term_names,
            extra_terms_max=candidate_extra_terms_max
        )
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

        run_train_loop(
            model, train_loader, opt_nn, opt_lib, n_epochs=candidate_refit_epochs,
            lambda_reg=lambda_reg, lambda_gamma_pen=lambda_gamma_pen
        )

        mse_tr = eval_mse(model, X_train, Y_train)
        mse_va = eval_mse(model, X_val, Y_val)

        if not is_multi:
            k = int(mask_np.sum())
            terms = np.array(term_names)[mask_np].tolist()
        else:
            k = int(sum(int(mm.sum()) for mm in mask_np))
            terms = [np.array(tn)[mm].tolist() for tn, mm in zip(term_names, mask_np)]

        bic_val = bic_from_mse(mse_va, k, X_val.shape[0])

        results.append({
            "terms": terms,
            "k": k,
            "mse_train": mse_tr,
            "mse_val": mse_va,
            "score": bic_val,  # lower is better
            "mask": mask_np.copy() if not is_multi else [m.copy() for m in mask_np],
        })

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
