"""
Module-level configuration for PD model discovery.

This file defines:
- The fixed list of module combinations to evaluate (for reproducibility)
- The latent state layout per module (can be manually edited later)
- A compact, module-scoped candidate term list (kept small to avoid explosion)

You can safely modify MODULE_COMBINATIONS and STATE_SPECS to add/remove
models or change latent state counts without touching core code.
"""
import copy

from src.configs.pd_atom_registry import (
    DISCOVERY_PROFILES,
    MECHANISM_MODULES,
    OBSERVABLE_ATOM_LAYERS,
    PD_ATOMS,
)

# ----------------------------
# 1) Fixed module combinations (model families)
# ----------------------------
# Keep this list fixed for now (per your request), but editable in the future.
MODULE_COMBINATIONS = [
    # Base families
    "direct",
    "idr",
    "tgi",

    # Base + delay chain
    "idr+delay",
    "direct+delay",
    "tgi+delay",

    # Base + biophase
    "direct+biophase",
    "idr+biophase",

    # Base + tolerance
    "idr+tolerance",
    "direct+tolerance",

    # Base + feedback
    "idr+feedback",
    "direct+feedback",

    # Circadian variants
    "idr+circadian",
    "direct+circadian",

    # Drug interaction (dual input)
    "direct+interaction",
    "idr+interaction",

    # Mechanism-focused single modules
    "biophase",
    "tolerance",
    "feedback",
    "circadian",
    "disease",
    "precursor",
]


# ----------------------------
# 2) Latent state specs per module
# ----------------------------
# These reflect the PD模型 definitions. You can change counts here later.
STATE_SPECS = {
    "direct":   {"states": ["R"], "latent": []},
    "idr":      {"states": ["R"], "latent": []},
    "tgi":      {"states": ["R"], "latent": []},
    "tmdd":     {"states": ["R", "CpR"], "latent": ["CpR"]},

    "delay":    {"states": ["T1", "T2", "T3"], "latent": ["T1", "T2", "T3"]},
    "feedback": {"states": ["T4", "T5", "T6"], "latent": ["T4", "T5", "T6"]},
    "tolerance":{"states": ["Tol"], "latent": ["Tol"]},
    "biophase": {"states": ["Ce"], "latent": ["Ce"]},
    "circadian":{"states": ["C1", "C2"], "latent": ["C1", "C2"]},
    "disease": {"states": ["PD1"], "latent": ["PD1"]},
    "precursor": {"states": ["P"], "latent": ["P"]},
    "interaction": {"states": ["C_int"], "latent": []},
}


# ----------------------------
# 3) Compact, module-scoped candidate terms
# ----------------------------
# Terms are intentionally small to avoid collinearity explosion.
# These are module-level term templates to be expanded in library code.
MODULE_TERMS = {
    "direct": {
        "R": ["1", "R", "R^2", "C", "Emax(C)", "Hill(C)"]
    },
    "idr": {
        "R": ["1", "R", "R^2", "Hill(C)", "Hill(C)*R"]
    },
    "tgi": {
        "R": ["R", "R^2", "Hill(C)", "Hill(C)*R", "exp(-t/24)*Hill(C)*R", "exp(-t)", "exp(-t)*R"]
    },
    "delay": {
        "T3": ["T3", "Hill(T3)"],
        "R":  ["1", "R", "Hill(T3)", "Hill(T3)*R"]
    },
    "feedback": {
        "T6": ["R", "T6"],
        "R":  ["1", "R", "Hill(C)", "Hill(T6)", "Hill(T6)*R"]
    },
    "tolerance": {
        "Tol": ["1", "Tol", "Hill(C)"],
        "R":   ["1", "R", "Hill(C)", "Tol*R"]
    },
    "biophase": {
        "Ce": ["C", "Ce"],
        "R":  ["1", "R", "Emax(Ce)", "Hill(Ce)", "Hill(Ce)*R"]
    },
    "circadian": {
        "C1": ["C1", "C2"],
        "C2": ["C1", "C2"],
        "R": ["1", "R", "C1", "C2", "Hill(C)"]
    },
    "disease": {
        "PD1": ["1", "PD1", "Hill(C)"],
        "R": ["PD1", "R"]
    },
    "precursor": copy.deepcopy(MECHANISM_MODULES["precursor"]["confirmation_terms"]),
    "interaction": {
        "R": ["1", "R", "Hill(C)", "Hill(C_int)"]
    }
}


# ----------------------------
# 4) Hierarchy rules (for pruning)
# ----------------------------
# If a term is present, required base terms should also be present.
HIERARCHY_RULES = {
    "Hill(C)*R": ["Hill(C)", "R"],
    "Hill(Ce)*R": ["Hill(Ce)", "R"],
    "Hill(T3)*R": ["Hill(T3)", "R"],
    "Hill(T6)*R": ["Hill(T6)", "R"],
    "Tol*R": ["Tol", "R"],
    "P*R": ["P", "R"],
    "CpR*R": ["CpR", "R"],
    "C-Ct": ["C", "Ct"],
    "exp(-t)*R": ["exp(-t)", "R"],
}


ACTIVE_MODEL_TO_MODULE = {
    "BIOPHASE_EMAX": "biophase",
    "BIOPHASE_SIGEMAX": "biophase",
    "TOLERANCE_ADAPTATION": "tolerance",
    "FEEDBACK_REGULATION": "feedback",
    "CIRCADIAN_REGULATION": "circadian",
    "DISEASE_PROGRESSION": "disease",
    "PRECURSOR_POOL": "precursor",
}


# ----------------------------
# 5) Atom metadata and discovery profiles
# ----------------------------
# Atom metadata and discovery profiles are imported from pd_atom_registry.py.


def terms_for_roles(roles, available_columns=None, pk_compartments=None, include_hidden=False):
    available = set(available_columns) if available_columns is not None else set()
    out = []
    for term, meta in PD_ATOMS.items():
        if meta.get("role") not in set(roles or []):
            continue
        if meta.get("state") == "hidden" and not include_hidden:
            continue
        min_comp = meta.get("min_pk_compartments")
        if min_comp is not None and pk_compartments is not None and int(pk_compartments) < int(min_comp):
            continue
        req = set(meta.get("requires", []))
        if req and available and not req.issubset(available):
            continue
        out.append(term)
    return out


def resolve_discovery_profile(
    profile: str,
    available_columns=None,
    pk_compartments=None,
    topk: int = None,
):
    name = str(profile or "observable_blind").strip().lower()
    if name not in DISCOVERY_PROFILES:
        raise ValueError(f"Unknown discovery profile: {profile}")
    spec = DISCOVERY_PROFILES[name]
    cfg = {"discovery_profile": name}

    allowed_roles = set(spec.get("allowed_roles", set()))
    allowed_terms = terms_for_roles(
        roles=allowed_roles,
        available_columns=available_columns,
        pk_compartments=pk_compartments,
        include_hidden=(spec.get("mode") != "discovery"),
    )
    forbidden_terms = set(spec.get("forbidden_terms", set()))

    if name in {"observable_blind", "observable_surrogate"}:
        required_terms = terms_for_roles(
            roles=spec.get("required_any_roles", set()),
            available_columns=available_columns,
            pk_compartments=pk_compartments,
            include_hidden=False,
        )
        cfg.update(
            {
                "ranking_mode": spec["ranking_mode"],
                "required_any_terms": required_terms,
                "mandatory_any_terms": required_terms,
                "forbidden_terms": sorted(forbidden_terms),
                "force_single_exposure_basis": bool(spec.get("force_single_exposure_basis", False)),
                "interaction_keep_ratio": float(spec.get("interaction_keep_ratio", 0.25)),
                "scoring_mode": spec["scoring_mode"],
                "scoring_prior_weight": float(spec["scoring_prior_weight"]),
                "strict_blind_discovery": bool(spec["strict_blind_discovery"]),
                "allowed_terms_by_profile": allowed_terms,
                "observable_atom_layers": copy.deepcopy(spec.get("observable_atom_layers", {})),
            }
        )
        if "candidate_extra_terms_max" in spec:
            cfg["candidate_extra_terms_max"] = int(spec["candidate_extra_terms_max"])
    else:
        cfg.update(
            {
                "forbidden_terms": sorted(forbidden_terms),
                "strict_blind_discovery": bool(spec.get("strict_blind_discovery", True)),
                "allowed_terms_by_profile": allowed_terms,
            }
        )

    if topk is not None:
        cfg["topk"] = int(topk)
    return cfg
