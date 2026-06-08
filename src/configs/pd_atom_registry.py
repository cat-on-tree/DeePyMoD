"""
Central PD atom and mechanism-module registry.

This file is the source of truth for the residual-guided PD refinement
workflow.  It separates three concerns that should not be mixed:

1. H0 observable surrogate atoms: terms allowed to explain an
   observable-sufficient PD response.
2. Hidden mechanism modules: terms/states used only for mechanism proposal
   and confirmation.
3. Dedicated TMDD/antibody atoms: isolated from the generic small-molecule
   blind flow.
"""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PDAtom:
    name: str
    formula: str
    role: str
    mechanism_family: str = "observable"
    required_inputs: tuple = field(default_factory=tuple)
    requires_latent_state: str | None = None
    allowed_in_h0: bool = False
    allowed_in_confirmation: bool = True
    min_pk_compartments: int | None = None
    notes: str = ""

    def to_legacy_dict(self):
        out = asdict(self)
        out["requires"] = list(self.required_inputs)
        if self.role == "tmdd_complex":
            out["state"] = "hidden_or_observed"
        elif self.requires_latent_state is not None:
            out["state"] = "hidden"
        elif any(x in {"C_int_obs", "Ct_obs"} for x in self.required_inputs):
            out["state"] = "observed_optional"
        else:
            out["state"] = "observed"
        if self.requires_latent_state is not None:
            out["requires_state"] = self.requires_latent_state
        if self.min_pk_compartments is not None:
            out["min_pk_compartments"] = self.min_pk_compartments
        return out


ATOM_DEFINITIONS = {
    # H0 observable surrogate core.
    "1": PDAtom("1", "1", "baseline", required_inputs=("time",), allowed_in_h0=True),
    "R": PDAtom("R", "R", "turnover", required_inputs=("R_obs",), allowed_in_h0=True),
    "R^2": PDAtom("R^2", "R^2", "turnover", required_inputs=("R_obs",), allowed_in_h0=True),
    "C": PDAtom("C", "C", "exposure", required_inputs=("C_obs",), allowed_in_h0=True),
    "Emax(C)": PDAtom("Emax(C)", "C / (EC50 + C)", "nonlinear_exposure", required_inputs=("C_obs",), allowed_in_h0=True),
    "Hill(C)": PDAtom("Hill(C)", "C^gamma / (EC50^gamma + C^gamma)", "nonlinear_exposure", required_inputs=("C_obs",), allowed_in_h0=True),
    "C*R": PDAtom("C*R", "C * R", "modulation", required_inputs=("C_obs", "R_obs"), allowed_in_h0=True),
    "Emax(C)*R": PDAtom("Emax(C)*R", "Emax(C) * R", "modulation", required_inputs=("C_obs", "R_obs"), allowed_in_h0=True),
    "Hill(C)*R": PDAtom("Hill(C)*R", "Hill(C) * R", "modulation", required_inputs=("C_obs", "R_obs"), allowed_in_h0=True),

    # Empirical observable atoms.  These are useful for compatibility and
    # stress testing, but should be reported separately from the minimal
    # mechanistic core.
    "C^2": PDAtom("C^2", "C^2", "empirical_exposure", required_inputs=("C_obs",), allowed_in_h0=True),
    "exp(-t/24)*Emax(C)*R": PDAtom(
        "exp(-t/24)*Emax(C)*R",
        "exp(-t/24) * Emax(C) * R",
        "observable_time_modulation",
        required_inputs=("time", "C_obs", "R_obs"),
        allowed_in_h0=True,
    ),
    "exp(-t/24)*Hill(C)*R": PDAtom(
        "exp(-t/24)*Hill(C)*R",
        "exp(-t/24) * Hill(C) * R",
        "observable_time_modulation",
        required_inputs=("time", "C_obs", "R_obs"),
        allowed_in_h0=True,
    ),
    "exp(-t)": PDAtom("exp(-t)", "exp(-t)", "time_baseline", required_inputs=("time",), allowed_in_h0=True),
    "exp(-t)*R": PDAtom("exp(-t)*R", "exp(-t) * R", "time_baseline", required_inputs=("time", "R_obs"), allowed_in_h0=True),

    # Optional observed inputs.
    "C_int": PDAtom("C_int", "C_int", "interaction", "interaction", required_inputs=("C_int_obs",), allowed_in_h0=False),
    "Hill(C_int)": PDAtom(
        "Hill(C_int)",
        "C_int^gamma / (EC50^gamma + C_int^gamma)",
        "interaction",
        "interaction",
        required_inputs=("C_int_obs",),
        allowed_in_h0=False,
    ),
    "Ct": PDAtom("Ct", "Ct", "tissue", "tissue", required_inputs=("Ct_obs",), allowed_in_h0=False, min_pk_compartments=2),
    "C-Ct": PDAtom("C-Ct", "C - Ct", "tissue", "tissue", required_inputs=("Ct_obs",), allowed_in_h0=False, min_pk_compartments=2),
    "Ct*R": PDAtom("Ct*R", "Ct * R", "tissue", "tissue", required_inputs=("Ct_obs", "R_obs"), allowed_in_h0=False, min_pk_compartments=2),

    # Hidden mechanism atoms.
    "Ce": PDAtom("Ce", "Ce", "biophase", "biophase", requires_latent_state="Ce", allowed_in_h0=False),
    "Emax(Ce)": PDAtom("Emax(Ce)", "Ce / (EC50 + Ce)", "biophase", "biophase", requires_latent_state="Ce", allowed_in_h0=False),
    "Hill(Ce)": PDAtom("Hill(Ce)", "Hill(Ce)", "biophase", "biophase", requires_latent_state="Ce", allowed_in_h0=False),
    "Hill(Ce)*R": PDAtom("Hill(Ce)*R", "Hill(Ce) * R", "biophase", "biophase", requires_latent_state="Ce", allowed_in_h0=False),
    "T1": PDAtom("T1", "T1", "delay", "delay", requires_latent_state="Transit", allowed_in_h0=False),
    "T2": PDAtom("T2", "T2", "delay", "delay", requires_latent_state="Transit", allowed_in_h0=False),
    "T3": PDAtom("T3", "T3", "delay", "delay", requires_latent_state="Transit", allowed_in_h0=False),
    "Hill(T3)": PDAtom("Hill(T3)", "Hill(T3)", "delay", "delay", requires_latent_state="Transit", allowed_in_h0=False),
    "Hill(T3)*R": PDAtom("Hill(T3)*R", "Hill(T3) * R", "delay", "delay", requires_latent_state="Transit", allowed_in_h0=False),
    "T4": PDAtom("T4", "T4", "feedback", "feedback", requires_latent_state="Feedback", allowed_in_h0=False),
    "T5": PDAtom("T5", "T5", "feedback", "feedback", requires_latent_state="Feedback", allowed_in_h0=False),
    "T6": PDAtom("T6", "T6", "feedback", "feedback", requires_latent_state="Feedback", allowed_in_h0=False),
    "Hill(T6)": PDAtom("Hill(T6)", "Hill(T6)", "feedback", "feedback", requires_latent_state="Feedback", allowed_in_h0=False),
    "Hill(T6)*R": PDAtom("Hill(T6)*R", "Hill(T6) * R", "feedback", "feedback", requires_latent_state="Feedback", allowed_in_h0=False),
    "Tol": PDAtom("Tol", "Tol", "tolerance", "tolerance", requires_latent_state="Tol", allowed_in_h0=False),
    "Tol*R": PDAtom("Tol*R", "Tol * R", "tolerance", "tolerance", requires_latent_state="Tol", allowed_in_h0=False),
    "C1": PDAtom("C1", "circadian cos-like state", "circadian", "circadian", requires_latent_state="Circadian", allowed_in_h0=False),
    "C2": PDAtom("C2", "circadian sin-like state", "circadian", "circadian", requires_latent_state="Circadian", allowed_in_h0=False),
    "cos(2pi*t/24)": PDAtom("cos(2pi*t/24)", "cos(2*pi*t/24)", "circadian", "circadian", required_inputs=("time",), allowed_in_h0=False),
    "sin(2pi*t/24)": PDAtom("sin(2pi*t/24)", "sin(2*pi*t/24)", "circadian", "circadian", required_inputs=("time",), allowed_in_h0=False),
    "PD1": PDAtom("PD1", "PD1", "progression", "disease", requires_latent_state="PD1", allowed_in_h0=False),
    "P": PDAtom("P", "precursor pool", "precursor", "precursor", requires_latent_state="P", allowed_in_h0=False),
    "Hill(P)": PDAtom("Hill(P)", "Hill(P)", "precursor", "precursor", requires_latent_state="P", allowed_in_h0=False),
    "P*R": PDAtom("P*R", "P * R", "precursor", "precursor", requires_latent_state="P", allowed_in_h0=False),

    # Dedicated TMDD/antibody atoms.
    "CpR": PDAtom("CpR", "drug-receptor complex", "tmdd_complex", "tmdd", requires_latent_state="CpR", allowed_in_h0=False),
    "CpR*R": PDAtom("CpR*R", "CpR * R", "tmdd_complex", "tmdd", requires_latent_state="CpR", allowed_in_h0=False),
}


PD_ATOMS = {name: atom.to_legacy_dict() for name, atom in ATOM_DEFINITIONS.items()}


OBSERVABLE_CORE_TERMS = ["1", "R", "R^2", "C", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"]
OBSERVABLE_EMPIRICAL_TERMS = ["C^2", "exp(-t/24)*Emax(C)*R", "exp(-t/24)*Hill(C)*R", "exp(-t)", "exp(-t)*R"]
H0_MINIMAL_TERMS = ["1", "R", "Hill(C)"]
H0_MECHANISTIC_TERMS = ["1", "R", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R", "R^2", "exp(-t/24)*Emax(C)*R", "exp(-t/24)*Hill(C)*R"]
H0_EMPIRICAL_TERMS = H0_MECHANISTIC_TERMS + ["exp(-t)", "exp(-t)*R"]
EXPANDED_PD_LIBRARY_TERMS = ["1", "R", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R", "R^2", "exp(-t)", "exp(-t)*R"]


OBSERVABLE_ATOM_LAYERS = {
    "baseline_turnover": ["1", "R", "R^2"],
    "exposure_drive": ["C", "Emax(C)", "Hill(C)"],
    "empirical_exposure": ["C^2"],
    "exposure_response_coupling": ["C*R", "Emax(C)*R", "Hill(C)*R"],
    "observable_time_modulation": ["exp(-t/24)*Emax(C)*R", "exp(-t/24)*Hill(C)*R"],
    "empirical_time_basis": ["exp(-t)", "exp(-t)*R"],
    "optional_tissue_exposure": ["Ct", "C-Ct", "Ct*R"],
}


MECHANISM_MODULES = {
    "biophase": {
        "latent_states": ["Ce"],
        "marker_terms": ["Ce", "Emax(Ce)", "Hill(Ce)", "Hill(Ce)*R"],
        "confirmation_terms": {"Ce": ["C", "Ce"], "R": ["1", "R", "Emax(Ce)", "Hill(Ce)", "Hill(Ce)*R"]},
        "combos": ["biophase", "direct+biophase", "idr+biophase"],
    },
    "delay": {
        "latent_states": ["T1", "T2", "T3"],
        "marker_terms": ["T1", "T2", "T3", "Hill(T3)", "Hill(T3)*R"],
        "confirmation_terms": {"T3": ["T3", "Hill(T3)"], "R": ["1", "R", "Hill(T3)", "Hill(T3)*R"]},
        "combos": ["direct+delay", "idr+delay", "tgi+delay"],
    },
    "feedback": {
        "latent_states": ["T4", "T5", "T6"],
        "marker_terms": ["T4", "T5", "T6", "Hill(T6)", "Hill(T6)*R"],
        "confirmation_terms": {"T6": ["R", "T6"], "R": ["1", "R", "Hill(C)", "Hill(T6)", "Hill(T6)*R"]},
        "combos": ["feedback", "direct+feedback", "idr+feedback"],
    },
    "tolerance": {
        "latent_states": ["Tol"],
        "marker_terms": ["Tol", "Tol*R"],
        "confirmation_terms": {"Tol": ["1", "Tol", "Hill(C)"], "R": ["1", "R", "Hill(C)", "Tol*R"]},
        "combos": ["tolerance", "direct+tolerance", "idr+tolerance"],
    },
    "circadian": {
        "latent_states": ["C1", "C2"],
        "marker_terms": ["C1", "C2", "cos(2pi*t/24)", "sin(2pi*t/24)"],
        "confirmation_terms": {"C1": ["C1", "C2"], "C2": ["C1", "C2"], "R": ["1", "R", "C1", "C2", "Hill(C)"]},
        "combos": ["circadian", "direct+circadian", "idr+circadian"],
    },
    "disease": {
        "latent_states": ["PD1"],
        "marker_terms": ["PD1"],
        "confirmation_terms": {"PD1": ["1", "PD1", "Hill(C)"], "R": ["PD1", "R"]},
        "combos": ["disease"],
    },
    "precursor": {
        "latent_states": ["P"],
        "marker_terms": ["P", "Hill(P)", "P*R"],
        "confirmation_terms": {"P": ["1", "P", "Hill(C)"], "R": ["1", "R", "P", "Hill(P)"]},
        "combos": ["precursor"],
    },
    "interaction": {
        "latent_states": [],
        "marker_terms": ["C_int", "Hill(C_int)"],
        "confirmation_terms": {"R": ["1", "R", "Hill(C)", "Hill(C_int)"]},
        "combos": ["direct+interaction", "idr+interaction"],
    },
}


MECHANISM_CONFIRMATION_COMBOS = {name: list(spec["combos"]) for name, spec in MECHANISM_MODULES.items()}
MECHANISM_MARKER_TERMS = {name: set(spec["marker_terms"]) for name, spec in MECHANISM_MODULES.items()}


DISCOVERY_PROFILES = {
    "standard_blind": {
        "mode": "discovery",
        "allowed_roles": {
            "baseline", "turnover", "exposure", "empirical_exposure", "nonlinear_exposure", "modulation",
            "time_baseline", "observable_time_modulation", "interaction", "tissue",
        },
        "forbidden_terms": set(),
        "strict_blind_discovery": True,
    },
    "observable_surrogate": {
        "mode": "discovery",
        "allowed_roles": {
            "baseline", "turnover", "exposure", "empirical_exposure", "nonlinear_exposure", "modulation",
            "time_baseline", "observable_time_modulation", "tissue",
        },
        "forbidden_terms": {"C_int", "Hill(C_int)", "Ct", "C-Ct", "Ct*R"},
        "required_any_roles": {"exposure", "nonlinear_exposure", "modulation"},
        "force_single_exposure_basis": False,
        "ranking_mode": "candidate_search",
        "scoring_mode": "bic_prior",
        "scoring_prior_weight": 1.25,
        "strict_blind_discovery": True,
        "candidate_extra_terms_max": 3,
        "observable_atom_layers": OBSERVABLE_ATOM_LAYERS,
    },
    "observable_blind": {
        "mode": "discovery",
        "allowed_roles": {"baseline", "turnover", "exposure", "empirical_exposure", "nonlinear_exposure", "modulation"},
        "forbidden_terms": {"exp(-t)", "exp(-t)*R"},
        "required_any_roles": {"exposure", "nonlinear_exposure", "modulation"},
        "force_single_exposure_basis": True,
        "ranking_mode": "two_stage_observable",
        "scoring_mode": "bic_prior",
        "scoring_prior_weight": 2.0,
        "interaction_keep_ratio": 0.22,
        "strict_blind_discovery": True,
    },
    "hidden_hinting": {
        "mode": "hint_only",
        "allowed_roles": {
            "baseline", "turnover", "exposure", "empirical_exposure", "nonlinear_exposure", "modulation",
            "biophase", "delay", "feedback", "tolerance", "circadian", "progression", "precursor", "interaction",
        },
        "forbidden_terms": set(),
        "strict_blind_discovery": True,
    },
    "tmdd_antibody": {
        "mode": "dedicated_library",
        "library": "library_antibody",
        "allowed_roles": {"baseline", "turnover", "exposure", "tmdd_complex", "tissue"},
        "strict_blind_discovery": False,
    },
}


def terms_for_h0_layer(layer: str):
    if layer == "minimal":
        return list(H0_MINIMAL_TERMS)
    if layer == "mechanistic":
        return list(H0_MECHANISTIC_TERMS)
    if layer in {"empirical", "full"}:
        return list(H0_EMPIRICAL_TERMS)
    raise ValueError(f"Unknown H0 layer: {layer}")

