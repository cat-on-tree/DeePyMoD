"""
Module-level configuration for PD model discovery.

This file defines:
- The fixed list of module combinations to evaluate (for reproducibility)
- The latent state layout per module (can be manually edited later)
- A compact, module-scoped candidate term list (kept small to avoid explosion)

You can safely modify MODULE_COMBINATIONS and STATE_SPECS to add/remove
models or change latent state counts without touching core code.
"""

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
    "circadian":{"states": [], "latent": []},
    "interaction": {"states": ["C_int"], "latent": []},
}


# ----------------------------
# 3) Compact, module-scoped candidate terms
# ----------------------------
# Terms are intentionally small to avoid collinearity explosion.
# These are module-level term templates to be expanded in library code.
MODULE_TERMS = {
    "direct": {
        "R": ["1", "R", "C", "Emax(C)", "Hill(C)"]
    },
    "idr": {
        "R": ["1", "R", "Hill(C)", "Hill(C)*R"]
    },
    "tgi": {
        "R": ["R", "Hill(C)", "Hill(C)*R"]
    },
    "delay": {
        "T3": ["T3", "Hill(T3)"],
        "R":  ["1", "R", "Hill(T3)", "Hill(T3)*R"]
    },
    "feedback": {
        "T6": ["T6", "Hill(T6)"],
        "R":  ["1", "R", "Hill(T6)", "Hill(T6)*R"]
    },
    "tolerance": {
        "Tol": ["Tol"],
        "R":   ["1", "R", "Hill(C)", "Hill(C)*R", "Tol*R"]
    },
    "biophase": {
        "Ce": ["Ce"],
        "R":  ["1", "R", "Emax(Ce)", "Hill(Ce)"]
    },
    "circadian": {
        "R": ["1", "R", "cos(2pi*t/24)", "sin(2pi*t/24)"]
    },
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
    "Hill(T3)*R": ["Hill(T3)", "R"],
    "Hill(T6)*R": ["Hill(T6)", "R"],
    "Tol*R": ["Tol", "R"],
    "CpR*R": ["CpR", "R"],
    "C-Ct": ["C", "Ct"],
}
