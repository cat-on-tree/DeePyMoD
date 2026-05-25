import numpy as np
import torch

from src.configs.pd_module_registry import HIERARCHY_RULES as MODULE_HIERARCHY_RULES


PROTECTED_TERMS_MAP = {
    "IDR_BASE": {"1", "R"},
    "IDR_INHIB_KIN_SIG": {"1", "R"},
    "IDR_STIM_KIN_SIG": {"1", "R"},
    "IDR_INHIB_KOUT_SIG": {"1", "R"},
    "IDR_STIM_KOUT_SIG": {"1", "R"},
    "DIRECT_LINEAR": {"1"},
    "DIRECT_EMAX": {"1"},
    "DIRECT_SIGEMAX": {"1"},
    "BIOPHASE_EMAX": {"1"},
    "BIOPHASE_SIGEMAX": {"1"},
    "TMDD_BASE": {"1", "R"},
    "ANTIBODY_PKPD": {"1", "R"},
}

EXCLUSIVE_GROUPS = [
    ["C", "C^2", "Emax(C)", "Hill(C)"],
    ["C*R", "Emax(C)*R", "Hill(C)*R"],
]

INTERACTION_TERMS = {"C*R", "Emax(C)*R", "Hill(C)*R"}

HIERARCHY_RULES = {
    "C*R": "C",
    "Emax(C)*R": "Emax(C)",
    "Hill(C)*R": "Hill(C)",
}


def enforce_hierarchy(mask_np, name_to_idx, extra_rules=None):
    m = mask_np.copy()
    all_rules = {}
    all_rules.update(HIERARCHY_RULES)
    if extra_rules:
        for child, parents in extra_rules.items():
            if isinstance(parents, (list, tuple)):
                all_rules[child] = list(parents)
            else:
                all_rules[child] = [parents]
    for child, parents in all_rules.items():
        if child not in name_to_idx:
            continue
        ci = name_to_idx[child]
        if not m[ci]:
            continue
        ok = False
        for parent in parents:
            if parent in name_to_idx and m[name_to_idx[parent]]:
                ok = True
                break
        if not ok:
            m[ci] = False
    return m


def _prune_single(
    coeff: torch.Tensor,
    old_mask: torch.Tensor,
    term_names,
    protected_terms,
    rel_thr_main=0.10,
    rel_thr_interaction=0.15,
    min_terms_keep=2,
):
    term_names = np.array(term_names)
    name_to_idx = {n: i for i, n in enumerate(term_names)}

    c = coeff.detach().cpu().numpy()
    m = old_mask.detach().cpu().numpy().astype(bool)

    active_idx = np.where(m)[0]
    if len(active_idx) <= min_terms_keep:
        return old_mask.clone(), False

    abs_active = np.abs(c[active_idx])
    cmax = np.max(abs_active) if len(abs_active) else 0.0
    if cmax <= 0:
        return old_mask.clone(), False

    new_m = np.zeros_like(m, dtype=bool)
    for idx in active_idx:
        term = term_names[idx]
        thr = rel_thr_interaction if term in INTERACTION_TERMS else rel_thr_main
        new_m[idx] = (np.abs(c[idx]) >= thr * cmax)

    for t in protected_terms:
        if t in name_to_idx:
            new_m[name_to_idx[t]] = True

    for g in EXCLUSIVE_GROUPS:
        gidx = [name_to_idx[t] for t in g if t in name_to_idx]
        act = [i for i in gidx if new_m[i]]
        if len(act) > 1:
            best = act[np.argmax(np.abs(c[act]))]
            for i in act:
                new_m[i] = (i == best)

    new_m = enforce_hierarchy(new_m, name_to_idx, extra_rules=MODULE_HIERARCHY_RULES)

    if new_m.sum() < min_terms_keep:
        order = np.argsort(-np.abs(c))
        for i in order:
            new_m[i] = True
            new_m = enforce_hierarchy(new_m, name_to_idx, extra_rules=MODULE_HIERARCHY_RULES)
            if new_m.sum() >= min_terms_keep:
                break
        for t in protected_terms:
            if t in name_to_idx:
                new_m[name_to_idx[t]] = True

    changed = not np.array_equal(new_m, m)
    return torch.tensor(new_m, dtype=torch.bool, device=old_mask.device), changed


def prune_mask_general(
    coeff,
    old_mask,
    term_names,
    active_model: str,
    rel_thr_main=0.10,
    rel_thr_interaction=0.15,
    min_terms_keep=2,
):
    """
    Supports both single-equation and multi-equation pruning.

    - coeff: torch.Tensor or list[torch.Tensor]
    - old_mask: torch.Tensor or list[torch.Tensor]
    - term_names: list[str] or list[list[str]]
    """
    protected_terms = PROTECTED_TERMS_MAP.get(active_model, {"1"})

    if isinstance(coeff, (list, tuple)):
        new_masks, changed_any = [], False
        for c, m, names in zip(coeff, old_mask, term_names):
            new_m, changed = _prune_single(
                coeff=c,
                old_mask=m,
                term_names=names,
                protected_terms=protected_terms,
                rel_thr_main=rel_thr_main,
                rel_thr_interaction=rel_thr_interaction,
                min_terms_keep=min_terms_keep,
            )
            new_masks.append(new_m)
            changed_any = changed_any or changed
        return new_masks, changed_any

    return _prune_single(
        coeff=coeff,
        old_mask=old_mask,
        term_names=term_names,
        protected_terms=protected_terms,
        rel_thr_main=rel_thr_main,
        rel_thr_interaction=rel_thr_interaction,
        min_terms_keep=min_terms_keep,
    )
