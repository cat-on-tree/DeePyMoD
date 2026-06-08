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
    "BIOPHASE_EMAX": {"1", "R", "Ce"},
    "BIOPHASE_SIGEMAX": {"1", "R", "Ce"},
    "TOLERANCE_ADAPTATION": {"1", "R", "Tol"},
    "FEEDBACK_REGULATION": {"1", "R", "T6"},
    "CIRCADIAN_REGULATION": {"1", "R", "C1", "C2"},
    "DISEASE_PROGRESSION": {"R", "PD1"},
    "PRECURSOR_POOL": {"1", "R", "P"},
    "TMDD_BASE": {"1", "R", "CpR"},
    "ANTIBODY_PKPD": {"1", "R", "CpR"},
}

EXCLUSIVE_GROUPS = [
    ["C", "C^2", "Emax(C)", "Hill(C)"],
    ["C*R", "Emax(C)*R", "Hill(C)*R"],
]

INTERACTION_TERMS = {"C*R", "Emax(C)*R", "Hill(C)*R", "exp(-t)*R"}

HIERARCHY_RULES = {
    "C*R": "C",
    "Emax(C)*R": "Emax(C)",
    "Hill(C)*R": "Hill(C)",
    "exp(-t)*R": "exp(-t)",
}

TMDD_MODELS = {"TMDD_BASE", "ANTIBODY_PKPD"}
TMDD_HIERARCHY_RULES = {
    "CpR*R": {"parents": ["CpR", "R"], "require_all": True},
    "C*R": {"parents": ["C", "R"], "require_all": True},
    "C-Ct": {"parents": ["C", "Ct"], "require_all": True},
}


def _flatten_term_names(term_names):
    if not isinstance(term_names, (list, tuple)):
        return []
    if len(term_names) == 0:
        return []
    if isinstance(term_names[0], (list, tuple, np.ndarray)):
        flat = []
        for names in term_names:
            flat.extend(list(names))
        return flat
    return list(term_names)


def _infer_protected_terms_from_structure(term_names):
    flat_terms = set(_flatten_term_names(term_names))
    protected = {"1", "R"}
    structural_states = {"Ce", "Tol", "T6", "C1", "C2", "PD1", "P", "CpR", "Ct"}
    for s in structural_states:
        if s in flat_terms:
            protected.add(s)
    return protected


def enforce_hierarchy(mask_np, name_to_idx, extra_rules=None):
    m = mask_np.copy()
    all_rules = {}
    all_rules.update(HIERARCHY_RULES)
    if extra_rules:
        for child, parents in extra_rules.items():
            if isinstance(parents, dict):
                all_rules[child] = parents
                continue
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
        require_all = False
        if isinstance(parents, dict):
            require_all = bool(parents.get("require_all", False))
            parents = parents.get("parents", [])
        if isinstance(parents, str):
            parents = [parents]

        hits = [
            (parent in name_to_idx and m[name_to_idx[parent]])
            for parent in parents
        ]
        ok = all(hits) if require_all else any(hits)
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
    hierarchy_rules=None,
    force_single_exposure_basis=False,
    mandatory_any_terms=None,
    forbidden_terms=None,
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

    forbidden_terms = set(forbidden_terms or [])
    for t in forbidden_terms:
        if t in name_to_idx:
            new_m[name_to_idx[t]] = False

    for g in EXCLUSIVE_GROUPS:
        gidx = [name_to_idx[t] for t in g if t in name_to_idx]
        act = [i for i in gidx if new_m[i]]
        if len(act) > 1:
            best = act[np.argmax(np.abs(c[act]))]
            for i in act:
                new_m[i] = (i == best)

    new_m = enforce_hierarchy(new_m, name_to_idx, extra_rules=hierarchy_rules)

    if force_single_exposure_basis:
        exposure_terms = ["C", "C^2", "Emax(C)", "Hill(C)"]
        exp_idx = [name_to_idx[t] for t in exposure_terms if t in name_to_idx and new_m[name_to_idx[t]]]
        if len(exp_idx) > 1:
            best = exp_idx[np.argmax(np.abs(c[exp_idx]))]
            for i in exp_idx:
                new_m[i] = (i == best)

    mandatory_any_terms = set(mandatory_any_terms or [])
    if mandatory_any_terms:
        has_mandatory = any((t in name_to_idx and new_m[name_to_idx[t]]) for t in mandatory_any_terms)
        if not has_mandatory:
            cand = [name_to_idx[t] for t in mandatory_any_terms if t in name_to_idx]
            if cand:
                best = cand[np.argmax(np.abs(c[cand]))]
                new_m[best] = True
                new_m = enforce_hierarchy(new_m, name_to_idx, extra_rules=hierarchy_rules)

    if new_m.sum() < min_terms_keep:
        order = np.argsort(-np.abs(c))
        for i in order:
            new_m[i] = True
            new_m = enforce_hierarchy(new_m, name_to_idx, extra_rules=hierarchy_rules)
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
    active_model: str = None,
    rel_thr_main=0.10,
    rel_thr_interaction=0.15,
    min_terms_keep=2,
    force_single_exposure_basis=False,
    mandatory_any_terms=None,
    forbidden_terms=None,
):
    """
    Supports both single-equation and multi-equation pruning.

    - coeff: torch.Tensor or list[torch.Tensor]
    - old_mask: torch.Tensor or list[torch.Tensor]
    - term_names: list[str] or list[list[str]]
    """
    if active_model in PROTECTED_TERMS_MAP:
        protected_terms = PROTECTED_TERMS_MAP[active_model]
    else:
        protected_terms = _infer_protected_terms_from_structure(term_names)
    hierarchy_rules = {}
    hierarchy_rules.update(MODULE_HIERARCHY_RULES)
    if active_model in TMDD_MODELS:
        hierarchy_rules.update(TMDD_HIERARCHY_RULES)

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
                hierarchy_rules=hierarchy_rules,
                force_single_exposure_basis=force_single_exposure_basis,
                mandatory_any_terms=mandatory_any_terms,
                forbidden_terms=forbidden_terms,
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
        hierarchy_rules=hierarchy_rules,
        force_single_exposure_basis=force_single_exposure_basis,
        mandatory_any_terms=mandatory_any_terms,
        forbidden_terms=forbidden_terms,
    )
