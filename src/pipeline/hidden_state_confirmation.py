import math

import numpy as np
import pandas as pd

from src.pipeline.discovery import run_single_discovery
from src.pipeline.mechanism_hints import MECHANISM_COMBOS


def _safe_rel_gain(base, new, eps=1e-8):
    den = max(abs(float(base)), eps)
    return float((float(base) - float(new)) / den)


def _is_finite(x):
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _pick_top_mechanisms(mechanism_hint, top_n=2, min_probability=0.20):
    rows = list(mechanism_hint.get("mechanisms", []))
    rows.sort(key=lambda r: float(r.get("probability", 0.0) or 0.0), reverse=True)
    picked = []
    for row in rows:
        mech = str(row.get("mechanism") or "")
        p = float(row.get("probability", 0.0) or 0.0)
        if not mech:
            continue
        if p < float(min_probability):
            continue
        picked.append(mech)
        if len(picked) >= int(top_n):
            break
    return picked


def _confirmation_combos_for_mech(mech, row, max_combos=2):
    combos = []
    best_combo = row.get("best_combo")
    if isinstance(best_combo, str) and best_combo:
        combos.append(best_combo)
    combos.extend(list(MECHANISM_COMBOS.get(mech, [])))
    if mech == "interaction":
        combos.extend(["direct+interaction", "idr+interaction"])
    out = []
    seen = set()
    for c in combos:
        c = str(c or "").strip()
        if (not c) or c in seen:
            continue
        out.append(c)
        seen.add(c)
        if len(out) >= int(max_combos):
            break
    return out


HIDDEN_R_MARKERS = {
    "biophase": {"Ce", "Emax(Ce)", "Hill(Ce)", "Hill(Ce)*R"},
    "delay": {"T1", "T2", "T3", "Hill(T3)", "Hill(T3)*R"},
    "feedback": {"T4", "T5", "T6", "Hill(T6)", "Hill(T6)*R"},
    "tolerance": {"Tol", "Tol*R"},
    "circadian": {"C1", "C2"},
    "disease": {"PD1"},
    "interaction": {"C_int", "Hill(C_int)"},
}


def _response_equation_terms(terms):
    if isinstance(terms, (list, tuple)) and terms and isinstance(terms[0], (list, tuple)):
        return [str(x) for x in terms[0]]
    if isinstance(terms, (list, tuple)):
        return [str(x) for x in terms]
    return [str(terms)]


def _has_valid_hidden_r_coupling(mech, terms):
    markers = HIDDEN_R_MARKERS.get(str(mech), set())
    if not markers:
        return True
    response_terms = set(_response_equation_terms(terms))
    return bool(response_terms.intersection(markers))


def _make_quick_cfg(config):
    cfg = dict(config or {})
    cfg["n_epochs_warmup"] = min(int(cfg.get("n_epochs_warmup", 1800)), int(cfg.get("hidden_bootstrap_warmup_epochs", 120)))
    cfg["n_epochs_prune"] = min(int(cfg.get("n_epochs_prune", 700)), int(cfg.get("hidden_bootstrap_prune_epochs", 50)))
    cfg["max_prune_rounds"] = min(int(cfg.get("max_prune_rounds", 10)), int(cfg.get("hidden_bootstrap_prune_rounds", 2)))
    cfg["candidate_refit_epochs"] = min(
        int(cfg.get("candidate_refit_epochs", 400)),
        int(cfg.get("hidden_bootstrap_refit_epochs", 40)),
    )
    cfg["topk"] = 1
    cfg["strict_blind_discovery"] = True
    cfg["force_single_exposure_basis"] = False
    cfg["required_any_terms"] = None
    cfg["mandatory_any_terms"] = None
    cfg["forbidden_terms"] = []
    cfg.pop("allowed_terms_by_profile", None)
    return cfg


def _residual_bootstrap_dataset(pop_data, rng):
    df = pd.DataFrame(pop_data).copy()
    required = ["sid", "time", "C_obs", "R_obs"]
    if not set(required).issubset(df.columns):
        return None
    out = df[required].copy()
    time_mean = out.groupby("time")["R_obs"].transform("mean")
    resid = (out["R_obs"].astype(float) - time_mean.astype(float)).to_numpy(dtype=float)
    if resid.size == 0:
        return None
    sampled = rng.choice(resid, size=resid.size, replace=True)
    out["R_obs"] = time_mean.to_numpy(dtype=float) + sampled
    return out


def _bootstrap_null_calibration(
    pop_data,
    active_model,
    observed_t_bic,
    hidden_combo,
    config,
    device=None,
):
    b_reps = int(config.get("hidden_bootstrap_reps", 6))
    if b_reps <= 0:
        return {"enabled": False, "reason": "reps_disabled"}
    quick_cfg = _make_quick_cfg(config)
    rng = np.random.default_rng(int(config.get("hidden_bootstrap_seed", 20260603)))
    t_boot = []
    errors = []
    for b in range(b_reps):
        pseudo = _residual_bootstrap_dataset(pop_data, rng)
        if pseudo is None:
            errors.append(f"rep{b + 1}:invalid_pseudo_data")
            continue
        try:
            obs = run_single_discovery(
                pop_data=pseudo,
                active_model=active_model,
                config=quick_cfg,
                device=device,
                module_combo=None,
            )
            hid = run_single_discovery(
                pop_data=pseudo,
                active_model=active_model,
                config=quick_cfg,
                device=device,
                module_combo=hidden_combo,
            )
            bic_obs = float(obs.get("best", {}).get("score", math.inf))
            bic_hid = float(hid.get("best", {}).get("score", math.inf))
            if _is_finite(bic_obs) and _is_finite(bic_hid):
                t_boot.append(float(bic_obs - bic_hid))
        except Exception as exc:
            errors.append(f"rep{b + 1}:{exc.__class__.__name__}")
    if not t_boot:
        return {
            "enabled": True,
            "n_reps_requested": b_reps,
            "n_reps_success": 0,
            "p_value": 1.0,
            "t_bootstrap": [],
            "errors": errors,
            "reason": "no_successful_bootstrap_reps",
        }
    arr = np.asarray(t_boot, dtype=float)
    p_value = float((1 + int(np.sum(arr >= float(observed_t_bic)))) / (len(arr) + 1))
    return {
        "enabled": True,
        "n_reps_requested": b_reps,
        "n_reps_success": int(len(arr)),
        "p_value": p_value,
        "t_bootstrap": [float(x) for x in arr.tolist()],
        "t_bootstrap_mean": float(np.mean(arr)),
        "t_bootstrap_p95": float(np.percentile(arr, 95.0)),
        "errors": errors,
    }


def _derive_verdict(confirmations, config):
    if not confirmations:
        return "no_hidden_state_supported", "no_confirmation_candidates"
    alpha = float(config.get("hidden_bootstrap_alpha", 0.10))
    strong_t = float(config.get("hidden_confirmation_t_bic_supported", 6.0))
    weak_t = float(config.get("hidden_confirmation_t_bic_gate", 2.0))
    best = confirmations[0]
    if bool(best.get("supported", False)):
        p_value = best.get("bootstrap_p_value", None)
        if p_value is None or float(p_value) <= alpha:
            return "hidden_state_supported", "bic_gain_and_bootstrap_supported"
        return "hidden_state_suspected_unconfirmed", "bootstrap_not_significant"
    max_t = max(float(r.get("t_bic", -math.inf) or -math.inf) for r in confirmations)
    max_risk = max(float(r.get("screen_score", 0.0) or 0.0) for r in confirmations)
    if max_t < weak_t:
        return "no_hidden_state_supported", "hidden_bic_gain_below_gate"
    if max_t >= strong_t or max_risk >= 65.0:
        return "hidden_state_suspected_unconfirmed", "screen_or_bic_risk_without_valid_confirmation"
    return "no_hidden_state_supported", "confirmation_did_not_validate_hidden_state"


def confirm_top_hidden_mechanisms(
    pop_data,
    active_model,
    baseline_discovery,
    mechanism_hint,
    config,
    device=None,
):
    baseline_best = baseline_discovery.get("best", {})
    baseline_bic = float(baseline_best.get("score", math.inf))
    baseline_mse = float(baseline_best.get("mse_val", math.inf))
    baseline_terms = baseline_best.get("terms", [])

    confirm_cfg = dict(config or {})
    confirm_cfg["n_epochs_warmup"] = min(int(confirm_cfg.get("n_epochs_warmup", 1800)), 900)
    confirm_cfg["n_epochs_prune"] = min(int(confirm_cfg.get("n_epochs_prune", 700)), 350)
    confirm_cfg["max_prune_rounds"] = min(int(confirm_cfg.get("max_prune_rounds", 10)), 6)
    confirm_cfg["candidate_refit_epochs"] = min(int(confirm_cfg.get("candidate_refit_epochs", 400)), 220)
    confirm_cfg["strict_blind_discovery"] = True
    confirm_cfg["force_single_exposure_basis"] = False
    confirm_cfg["required_any_terms"] = None
    confirm_cfg["mandatory_any_terms"] = None
    confirm_cfg["forbidden_terms"] = []
    confirm_cfg.pop("allowed_terms_by_profile", None)

    top_n = int(confirm_cfg.get("hidden_confirmation_topn", 2))
    min_prob = float(confirm_cfg.get("hidden_confirmation_min_probability", 0.20))
    max_combos = int(confirm_cfg.get("hidden_confirmation_max_combos", 2))
    thr_bic = float(confirm_cfg.get("hidden_confirmation_delta_bic_threshold", 0.03))
    thr_mse = float(confirm_cfg.get("hidden_confirmation_delta_mse_threshold", 0.02))
    t_gate = float(confirm_cfg.get("hidden_confirmation_t_bic_gate", 2.0))
    t_supported = float(confirm_cfg.get("hidden_confirmation_t_bic_supported", 6.0))
    alpha = float(confirm_cfg.get("hidden_bootstrap_alpha", 0.10))
    use_bootstrap = bool(confirm_cfg.get("enable_hidden_bootstrap_calibration", True))

    rows_by_mech = {str(r.get("mechanism")): r for r in mechanism_hint.get("mechanisms", [])}
    top_mechs = _pick_top_mechanisms(mechanism_hint, top_n=top_n, min_probability=min_prob)
    confirmations = []

    for mech in top_mechs:
        row = rows_by_mech.get(mech, {})
        combos = _confirmation_combos_for_mech(mech, row, max_combos=max_combos)
        best = None
        errors = []
        for combo in combos:
            try:
                rs = run_single_discovery(
                    pop_data=pop_data,
                    active_model=active_model,
                    config=confirm_cfg,
                    device=device,
                    module_combo=combo,
                )
                b = rs.get("best", {})
                bic = float(b.get("score", math.inf))
                mse_val = float(b.get("mse_val", math.inf))
                if (not _is_finite(bic)) or (not _is_finite(mse_val)):
                    continue
                cand = {
                    "combo": combo,
                    "bic": bic,
                    "mse_val": mse_val,
                    "k": int(b.get("k", 0)),
                    "terms": b.get("terms", []),
                }
                if (best is None) or (cand["bic"] < best["bic"]):
                    best = cand
            except Exception as exc:
                errors.append(f"{combo}: {exc.__class__.__name__}")

        if best is None:
            confirmations.append(
                {
                    "mechanism": mech,
                    "supported": False,
                    "reason": "confirmation_failed",
                    "screen_probability": float(row.get("probability", 0.0) or 0.0),
                    "screen_score": float(row.get("score", 0.0) or 0.0),
                    "tested_combos": combos,
                    "errors": errors,
                }
            )
            continue

        t_bic = float(baseline_bic - best["bic"])
        delta_bic = _safe_rel_gain(baseline_bic, best["bic"])
        delta_mse = _safe_rel_gain(baseline_mse, best["mse_val"])
        valid_r_coupling = _has_valid_hidden_r_coupling(mech, best["terms"])
        bootstrap = {"enabled": False, "reason": "not_requested"}
        if use_bootstrap and valid_r_coupling and t_bic >= t_gate:
            bootstrap = _bootstrap_null_calibration(
                pop_data=pop_data,
                active_model=active_model,
                observed_t_bic=t_bic,
                hidden_combo=best["combo"],
                config=confirm_cfg,
                device=device,
            )
        p_value = bootstrap.get("p_value") if bootstrap.get("enabled", False) else None
        bic_supported = bool(t_bic >= t_supported or ((delta_bic >= thr_bic) and (delta_mse >= thr_mse)))
        bootstrap_supported = bool((p_value is None) or (float(p_value) <= alpha))
        supported = bool(valid_r_coupling and bic_supported and bootstrap_supported)
        if not valid_r_coupling:
            reason = "hidden_state_not_coupled_to_response_equation"
        elif not bic_supported:
            reason = "bic_gain_below_hidden_state_gate"
        elif not bootstrap_supported:
            reason = "bootstrap_null_not_rejected"
        else:
            reason = "supported_by_bootstrap_calibrated_confirmation"
        confirmations.append(
            {
                "mechanism": mech,
                "supported": supported,
                "reason": reason,
                "screen_probability": float(row.get("probability", 0.0) or 0.0),
                "screen_score": float(row.get("score", 0.0) or 0.0),
                "best_combo": best["combo"],
                "best_terms": best["terms"],
                "best_bic": float(best["bic"]),
                "best_mse_val": float(best["mse_val"]),
                "t_bic": float(t_bic),
                "delta_bic": float(delta_bic),
                "delta_mse": float(delta_mse),
                "valid_hidden_r_coupling": bool(valid_r_coupling),
                "bootstrap_p_value": (float(p_value) if p_value is not None else None),
                "bootstrap_calibration": bootstrap,
                "tested_combos": combos,
            }
        )

    confirmations.sort(
        key=lambda r: (
            bool(r.get("supported", False)),
            float(r.get("t_bic", -1.0) or -1.0),
            float(r.get("delta_bic", -1.0) or -1.0),
            float(r.get("delta_mse", -1.0) or -1.0),
        ),
        reverse=True,
    )
    verdict, verdict_reason = _derive_verdict(confirmations, confirm_cfg)
    top_supported = next((r for r in confirmations if r.get("supported", False)), None)
    if top_supported is not None:
        recommended = f"fit {top_supported.get('mechanism')} confirmation model"
    elif verdict == "no_hidden_state_supported":
        recommended = "retain observable ODE; no hidden state supported"
    elif confirmations:
        recommended = f"monitor residuals and retest {confirmations[0].get('mechanism')}"
    else:
        recommended = "no hidden-state confirmation candidate"

    return {
        "type": "top_hidden_confirmation",
        "baseline": {
            "bic": baseline_bic,
            "mse_val": baseline_mse,
            "terms": baseline_terms,
        },
        "thresholds": {
            "delta_bic": thr_bic,
            "delta_mse": thr_mse,
            "t_bic_gate": t_gate,
            "t_bic_supported": t_supported,
            "bootstrap_alpha": alpha,
        },
        "top_candidates": top_mechs,
        "confirmations": confirmations,
        "hidden_state_verdict": verdict,
        "verdict_reason": verdict_reason,
        "recommended_confirmation": recommended,
    }
