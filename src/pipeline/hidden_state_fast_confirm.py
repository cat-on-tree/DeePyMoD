import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


FAST_GROUPS = {
    "biophase_effect": ["biophase"],
    "transduction_delay": ["delay"],
    "adaptive_effect": ["tolerance", "feedback"],
    "baseline_drift": ["disease", "circadian"],
    "precursor_pool": ["precursor"],
}


def _clip_pos(x, eps=1e-10):
    return np.maximum(np.asarray(x, dtype=float), eps)


def _hill(c, ec50, gamma):
    cc = _clip_pos(c)
    e = max(float(ec50), 1e-8)
    g = max(float(gamma), 1e-8)
    return (cc ** g) / (e ** g + cc ** g)


def _safe_bic(sse, n, k):
    if n <= 0 or not np.isfinite(sse):
        return math.inf
    sigma2 = max(float(sse) / max(int(n), 1), 1e-12)
    return float(int(n) * math.log(sigma2) + math.log(max(int(n), 1)) * int(k))


def _prepare_mean_data(pop_data, train_frac=0.7):
    df = pd.DataFrame(pop_data).copy()
    required = {"sid", "time", "C_obs", "R_obs"}
    if not required.issubset(df.columns):
        raise ValueError(f"pop_data must contain {required}, got {list(df.columns)}")
    for col in ["time", "C_obs", "R_obs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = df.groupby("time", as_index=False).agg(C=("C_obs", "mean"), R=("R_obs", "mean")).sort_values("time")
    t = agg["time"].to_numpy(dtype=float)
    c = agg["C"].to_numpy(dtype=float)
    r = agg["R"].to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(c) & np.isfinite(r)
    t, c, r = t[m], c[m], r[m]
    if len(t) < 6:
        raise ValueError("too few timepoints for fast hidden confirmation")
    n_train = int(np.floor(float(train_frac) * len(t)))
    n_train = min(max(n_train, 3), len(t) - 2)
    tr_idx = np.arange(n_train)
    va_idx = np.arange(n_train, len(t))
    r_mean = float(np.mean(r[tr_idx]))
    r_std = float(np.std(r[tr_idx]) + 1e-8)
    return {"t": t, "c": c, "r": r, "tr_idx": tr_idx, "va_idx": va_idx, "r_mean": r_mean, "r_std": r_std}


def _feature(term, r, c, ec50, gamma):
    key = str(term)
    if key == "1":
        return 1.0
    if key == "R":
        return float(r)
    if key == "R^2":
        return float(r * r)
    if key == "C":
        return float(c)
    if key == "C^2":
        return float(c * c)
    if key == "Emax(C)":
        cc = max(float(c), 1e-10)
        return float(cc / (max(float(ec50), 1e-10) + cc))
    if key == "Hill(C)":
        return float(_hill(float(c), ec50, gamma))
    if key == "C*R":
        return float(c * r)
    if key == "Emax(C)*R":
        return float(_feature("Emax(C)", r, c, ec50, gamma) * r)
    if key == "Hill(C)*R":
        return float(_feature("Hill(C)", r, c, ec50, gamma) * r)
    return 0.0


def _clean_h0_terms(terms):
    if isinstance(terms, (list, tuple)) and terms and isinstance(terms[0], (list, tuple)):
        terms = terms[0]
    legal = {"1", "R", "R^2", "C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"}
    out = [str(t) for t in (terms or []) if str(t) in legal]
    if not out:
        out = ["1", "R", "Hill(C)"]
    if not any(t in out for t in ["C", "C^2", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R"]):
        out.append("Hill(C)")
    return out


def _simulate_h0(theta, terms, data, ec50, gamma):
    t, c = data["t"], data["c"]
    r0 = float(data["r"][0])

    def c_of_t(tt):
        return float(np.interp(tt, t, c))

    def rhs(tt, y):
        rr = float(y[0])
        cc = c_of_t(tt)
        return [sum(float(theta[i]) * _feature(term, rr, cc, ec50, gamma) for i, term in enumerate(terms))]

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), [r0], t_eval=t, rtol=1e-5, atol=1e-7)
    if (not sol.success) or sol.y.shape[1] != len(t):
        return np.full_like(t, np.nan, dtype=float)
    return sol.y[0]


def _simulate_hidden(group, theta, data, ec50, gamma):
    t, c, r = data["t"], data["c"], data["r"]
    r0 = float(r[0])

    def c_of_t(tt):
        return float(np.interp(tt, t, c))

    def rhs(tt, y):
        rr = float(y[0])
        cc = c_of_t(tt)
        hc = float(_hill(cc, ec50, gamma))
        if group == "biophase_effect":
            ce = float(y[1])
            rate = float(theta[4])
            h = float(_hill(ce, ec50, gamma))
            d_r = theta[0] + theta[1] * rr + theta[2] * h + theta[3] * h * rr
            return [d_r, rate * (cc - ce)]
        if group == "transduction_delay":
            t1, t2 = float(y[1]), float(y[2])
            rate = float(theta[4])
            h = float(_hill(t2, ec50, gamma))
            d_r = theta[0] + theta[1] * rr + theta[2] * h + theta[3] * h * rr
            return [d_r, rate * (cc - t1), rate * (t1 - t2)]
        if group == "adaptive_effect":
            z = float(y[1])
            kin = float(theta[4])
            kout = float(theta[5])
            d_r = theta[0] + theta[1] * rr + theta[2] * hc + theta[3] * z * rr
            return [d_r, kin * hc - kout * z]
        if group == "baseline_drift":
            z = float(y[1])
            kin = float(theta[4])
            kout = float(theta[5])
            d_r = theta[0] + theta[1] * rr + theta[2] * hc + theta[3] * z
            return [d_r, kin - kout * z]
        if group == "precursor_pool":
            p = float(y[1])
            kin = float(theta[4])
            kout = float(theta[5])
            d_r = theta[0] + theta[1] * rr + theta[2] * hc + theta[3] * p
            return [d_r, kin - kout * p]
        return [0.0]

    if group == "transduction_delay":
        y0 = [r0, float(c[0]), float(c[0])]
    elif group in {"biophase_effect"}:
        y0 = [r0, float(c[0])]
    else:
        y0 = [r0, 0.0]
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), y0, t_eval=t, rtol=1e-5, atol=1e-7)
    if (not sol.success) or sol.y.shape[1] != len(t):
        return np.full_like(t, np.nan, dtype=float)
    return sol.y[0]


def _fit_model(sim_fn, p0, lb, ub, data, n_restarts=4, seed=0):
    rng = np.random.default_rng(int(seed))
    tr_idx, va_idx = data["tr_idx"], data["va_idx"]
    r, r_mean, r_std = data["r"], data["r_mean"], data["r_std"]

    def resid(p):
        yhat = sim_fn(p)
        if np.any(~np.isfinite(yhat)):
            return np.ones(len(tr_idx)) * 1e4
        return (yhat[tr_idx] - r[tr_idx]) / r_std

    best = None
    for i in range(int(n_restarts)):
        x0 = np.asarray(p0, dtype=float)
        if i > 0:
            scale = np.maximum(np.abs(x0), 0.2)
            x0 = x0 + rng.normal(0.0, 0.25, size=x0.size) * scale
        x0 = np.clip(x0, lb, ub)
        try:
            out = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=600, xtol=1e-7, ftol=1e-7, gtol=1e-7)
            yhat = sim_fn(out.x)
            if np.any(~np.isfinite(yhat)):
                continue
            val_resid = (yhat[va_idx] - r[va_idx]) / r_std
            sse = float(np.sum(val_resid * val_resid))
            bic = _safe_bic(sse, len(va_idx), len(out.x))
            row = {"theta": out.x.tolist(), "bic": bic, "sse": sse, "n_val": int(len(va_idx)), "success": bool(out.success)}
            if best is None or row["bic"] < best["bic"]:
                best = row
        except Exception:
            continue
    if best is None:
        return {"theta": [], "bic": math.inf, "sse": math.inf, "n_val": int(len(va_idx)), "success": False}
    return best


def _fit_h0(data, terms, ec50, gamma, cfg, seed=0):
    r = data["r"]
    dyn = float(max(np.max(r) - np.min(r), 1.0))
    b = float(min(max(12.0 * dyn, 30.0), 1000.0))
    p0 = np.zeros(len(terms), dtype=float)
    if "R" in terms:
        p0[terms.index("R")] = -0.2
    return _fit_model(
        sim_fn=lambda p: _simulate_h0(p, terms, data, ec50, gamma),
        p0=p0,
        lb=np.full(len(terms), -b),
        ub=np.full(len(terms), b),
        data=data,
        n_restarts=int(cfg.get("fast_confirm_restarts", 4)),
        seed=seed,
    )


def _fit_hidden_group(group, data, ec50, gamma, cfg, seed=0):
    r = data["r"]
    dyn = float(max(np.max(r) - np.min(r), 1.0))
    b = float(min(max(12.0 * dyn, 30.0), 1000.0))
    if group in {"biophase_effect", "transduction_delay"}:
        p0 = np.array([0.0, -0.2, dyn, 0.0, 0.8], dtype=float)
        lb = np.array([-b, -b, -b, -b, 0.02], dtype=float)
        ub = np.array([b, b, b, b, 5.0], dtype=float)
    else:
        p0 = np.array([0.0, -0.2, dyn, 0.0, 0.2, 0.1], dtype=float)
        lb = np.array([-b, -b, -b, -b, 0.0, 0.01], dtype=float)
        ub = np.array([b, b, b, b, 5.0, 5.0], dtype=float)
    return _fit_model(
        sim_fn=lambda p: _simulate_hidden(group, p, data, ec50, gamma),
        p0=p0,
        lb=lb,
        ub=ub,
        data=data,
        n_restarts=int(cfg.get("fast_confirm_restarts", 4)),
        seed=seed,
    )


def _score_groups_from_hint(mechanism_hint):
    mech_scores = {str(r.get("mechanism")): float(r.get("score", 0.0) or 0.0) for r in mechanism_hint.get("mechanisms", [])}
    residual = mechanism_hint.get("residual_evidence", {}) or {}
    out = {}
    hys = float(residual.get("hysteresis_score", 0.0) or 0.0)
    temporal = float(residual.get("temporal_shape_score", 0.0) or 0.0)
    out["biophase_effect"] = mech_scores.get("biophase", 0.0) + 20.0 * hys
    out["transduction_delay"] = mech_scores.get("delay", 0.0) + 15.0 * hys + 20.0 * temporal
    out["adaptive_effect"] = max(mech_scores.get("tolerance", 0.0), mech_scores.get("feedback", 0.0)) + 25.0 * float(residual.get("early_late_drift_score", 0.0) or 0.0)
    out["baseline_drift"] = max(mech_scores.get("disease", 0.0), mech_scores.get("circadian", 0.0)) + 30.0 * float(residual.get("zero_dose_trend_score", 0.0) or 0.0)
    out["precursor_pool"] = max(mech_scores.get("disease", 0.0), mech_scores.get("feedback", 0.0), mech_scores.get("tolerance", 0.0))
    return out


def _bootstrap_fast(pop_data, group, observed_t, terms, ec50, gamma, cfg, seed=0):
    reps = int(cfg.get("hidden_bootstrap_reps", 11))
    if reps <= 0:
        return {"enabled": False, "reason": "reps_disabled"}
    base_data = _prepare_mean_data(pop_data, train_frac=float(cfg.get("train_frac", 0.7)))
    h0 = _fit_h0(base_data, terms, ec50, gamma, cfg, seed=seed)
    yhat0 = _simulate_h0(np.asarray(h0.get("theta", []), dtype=float), terms, base_data, ec50, gamma)
    if np.any(~np.isfinite(yhat0)):
        return {"enabled": True, "n_reps_requested": reps, "n_reps_success": 0, "p_value": 1.0, "reason": "h0_sim_failed"}
    resid = base_data["r"] - yhat0
    rng = np.random.default_rng(int(seed) + 9001)
    t_boot = []
    for i in range(reps):
        pseudo = dict(base_data)
        pseudo["r"] = yhat0 + rng.choice(resid, size=resid.size, replace=True)
        h0_b = _fit_h0(pseudo, terms, ec50, gamma, cfg, seed=seed + i + 1)
        hid_b = _fit_hidden_group(group, pseudo, ec50, gamma, cfg, seed=seed + i + 101)
        if np.isfinite(h0_b["bic"]) and np.isfinite(hid_b["bic"]):
            t_boot.append(float(h0_b["bic"] - hid_b["bic"]))
    if not t_boot:
        return {"enabled": True, "n_reps_requested": reps, "n_reps_success": 0, "p_value": 1.0, "t_bootstrap": []}
    arr = np.asarray(t_boot, dtype=float)
    p = float((1 + int(np.sum(arr >= float(observed_t)))) / (len(arr) + 1))
    return {
        "enabled": True,
        "n_reps_requested": reps,
        "n_reps_success": int(len(arr)),
        "p_value": p,
        "t_bootstrap": [float(x) for x in arr.tolist()],
        "t_bootstrap_mean": float(np.mean(arr)),
        "t_bootstrap_p95": float(np.percentile(arr, 95.0)),
    }


def confirm_hidden_states_fast(pop_data, baseline_discovery, mechanism_hint, config):
    cfg = dict(config or {})
    ec50 = float(baseline_discovery.get("ec50_hat", cfg.get("fast_confirm_ec50", 4.0)) or 4.0)
    gamma = float(baseline_discovery.get("gamma_hat", cfg.get("fast_confirm_gamma", 2.0)) or 2.0)
    terms = _clean_h0_terms(baseline_discovery.get("best", {}).get("terms", []))
    data = _prepare_mean_data(pop_data, train_frac=float(cfg.get("train_frac", 0.7)))
    seed = int(cfg.get("hidden_bootstrap_seed", 20260603))

    h0 = _fit_h0(data, terms, ec50, gamma, cfg, seed=seed)
    group_scores = _score_groups_from_hint(mechanism_hint or {})
    max_groups = int(cfg.get("fast_confirm_max_groups", 4))
    selected = [g for g, _ in sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)[:max_groups]]

    alpha = float(cfg.get("hidden_bootstrap_alpha", 0.10))
    t_gate = float(cfg.get("hidden_confirmation_t_bic_gate", 2.0))
    t_supported = float(cfg.get("hidden_confirmation_t_bic_supported", 2.0))
    confirmations = []
    for i, group in enumerate(selected):
        fit = _fit_hidden_group(group, data, ec50, gamma, cfg, seed=seed + 31 * (i + 1))
        t_bic = float(h0["bic"] - fit["bic"])
        bootstrap = {"enabled": False, "reason": "below_t_gate"}
        p_value = None
        if t_bic >= t_gate and bool(cfg.get("enable_hidden_bootstrap_calibration", True)):
            bootstrap = _bootstrap_fast(pop_data, group, t_bic, terms, ec50, gamma, cfg, seed=seed + 211 * (i + 1))
            p_value = bootstrap.get("p_value")
        supported = bool(t_bic >= t_supported and (p_value is None or float(p_value) <= alpha))
        if supported:
            reason = "fast_template_bic_and_bootstrap_supported"
        elif t_bic < t_gate:
            reason = "fast_template_bic_gain_below_gate"
        else:
            reason = "fast_template_bootstrap_not_rejected"
        confirmations.append(
            {
                "mechanism": group,
                "supported": supported,
                "reason": reason,
                "screen_score": float(group_scores.get(group, 0.0)),
                "best_combo": f"fast_{group}",
                "best_terms": [f"fixed_template:{group}"],
                "best_bic": float(fit["bic"]),
                "best_mse_val": float(fit["sse"] / max(fit["n_val"], 1)),
                "t_bic": t_bic,
                "delta_bic": float((h0["bic"] - fit["bic"]) / max(abs(h0["bic"]), 1e-8)),
                "delta_mse": 0.0,
                "valid_hidden_r_coupling": True,
                "bootstrap_p_value": (float(p_value) if p_value is not None else None),
                "bootstrap_calibration": bootstrap,
                "theta_hat": fit.get("theta", []),
            }
        )

    confirmations.sort(key=lambda r: (bool(r.get("supported", False)), float(r.get("t_bic", -math.inf))), reverse=True)
    top_supported = next((r for r in confirmations if r.get("supported", False)), None)
    max_t = max([float(r.get("t_bic", -math.inf)) for r in confirmations], default=-math.inf)
    if top_supported is not None:
        verdict = "hidden_state_supported"
        verdict_reason = "fast_template_test_rejected_h0"
        recommended = f"confirm {top_supported.get('mechanism')} with full model"
    elif max_t >= t_gate:
        verdict = "hidden_state_suspected_unconfirmed"
        verdict_reason = "fast_template_gain_without_bootstrap_support"
        recommended = f"run full confirmation for {confirmations[0].get('mechanism')}" if confirmations else "run full confirmation"
    else:
        verdict = "no_hidden_state_supported"
        verdict_reason = "fast_template_no_group_passed_bic_gate"
        recommended = "retain observable ODE; no hidden state supported"

    return {
        "type": "top_hidden_confirmation",
        "method": "fast_template",
        "baseline": {
            "bic": float(h0["bic"]),
            "mse_val": float(h0["sse"] / max(h0["n_val"], 1)),
            "terms": terms,
            "theta_hat": h0.get("theta", []),
        },
        "thresholds": {
            "t_bic_gate": t_gate,
            "t_bic_supported": t_supported,
            "bootstrap_alpha": alpha,
        },
        "group_scores": group_scores,
        "top_candidates": selected,
        "confirmations": confirmations,
        "hidden_state_verdict": verdict,
        "verdict_reason": verdict_reason,
        "recommended_confirmation": recommended,
    }
