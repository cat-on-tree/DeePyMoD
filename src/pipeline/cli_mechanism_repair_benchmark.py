import argparse
import concurrent.futures as cf
import json
import os
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.defaults import DEFAULTS
from src.data.simulate_pkpd import generate_population_data
from src.pipeline.cli_fast_surrogate_residuals import (
    DEFAULT_MODELS,
    RESIDUAL_FEATURE_COLUMNS as CLASSIFIER_FEATURE_COLUMNS,
    _label_for_model,
)
from src.pipeline.mechanism_hints import (
    _fit_surrogate_layer_grid,
    _residual_evidence_from_multitrajectory_fit,
    _residual_scores_from_evidence,
    build_fixed_surrogate_protocol_evidence,
)


H0_REPAIR_TERMS = ["1", "R", "C", "Emax(C)", "Hill(C)", "C*R", "Emax(C)*R", "Hill(C)*R", "R^2"]
H0_FALLBACK_TERM_SETS = [
    ["1", "R", "Hill(C)", "Hill(C)*R"],
    ["1", "R", "C", "Hill(C)"],
    ["1", "R", "Hill(C)"],
]

REPAIR_MODULE_TERMS = {
    "biophase": ["C_lag-C", "(C_lag-C)*R", "dCdt*R"],
    "delay": ["C_lag", "C_lag*R", "C_lag-C", "dCdt"],
    "feedback": ["R^3", "C*R^2", "Hill(C)*R^2"],
    "tolerance": ["t*Hill(C)", "t*Hill(C)*R"],
    "circadian": ["cos(2pi*t/24)", "sin(2pi*t/24)", "cos(2pi*t/24)*R", "sin(2pi*t/24)*R"],
    "disease": ["t"],
    "precursor": ["exp(-t/24)*R", "exp(-t/24)*R^2"],
}

GUIDE_FEATURE_COLUMNS = [
    "sync_correlation",
    "best_lag_correlation",
    "hysteresis_score",
    "zero_dose_trend_score",
    "early_late_drift_score",
    "late_overprediction_score",
    "residual_bias_score",
    "temporal_shape_score",
    "dose_stratified_rmse_score",
    "poor_fit_score",
    "raw_residual_structure_score",
    "amplitude_weighted_structure_score",
    "response_peak_lag_score",
    "residual_peak_lag_score",
    "residual_loop_area_score",
    "response_loop_area_score",
    "residual_autocorr_score",
    "residual_sign_persistence_score",
    "late_residual_slope_score",
    "dose_residual_correlation_score",
    "positive_residual_fraction_mean",
    "positive_residual_fraction_sd",
    "residual_iqr_score",
]

DERIVED_GATE_FEATURE_COLUMNS = [
    "auc_residual_correlation_score",
    "auc_hill_residual_correlation_score",
    "exposure_accumulation_score",
    "dcdt_residual_correlation_score",
    "effect_site_gap_correlation_score",
    "lag_advantage_score",
    "circadian_projection_score",
    "circadian_detrended_projection_score",
    "zero_dose_monotonic_score",
    "exposure_independence_score",
    "late_auc_residual_direction_score",
    "late_auc_group_shift_score",
    "response_state_residual_correlation_score",
    "lagged_response_residual_correlation_score",
    "low_exposure_recovery_score",
    "turnover_asymmetry_score",
]

LABEL_TO_MECHANISM = {
    "observable_sufficient": "observable",
    "biophase_like": "biophase",
    "transduction_like": "delay",
    "feedback_like": "feedback",
    "circadian_like": "circadian",
    "disease_like": "disease",
    "tolerance_like": "tolerance",
    "precursor_like": "precursor",
}

MECHANISM_TO_LABEL = {v: k for k, v in LABEL_TO_MECHANISM.items()}

MODULE_GATE_DESCRIPTIONS = {
    "biophase": "hysteresis/lag and residual-response loop evidence",
    "delay": "lagged response/residual timing evidence",
    "feedback": "response-state dependent residual evidence",
    "tolerance": "exposure-time adaptation and late overprediction evidence",
    "circadian": "periodic/temporal residual structure evidence",
    "disease": "dose-independent baseline drift evidence",
    "precursor": "turnover-like lag/recovery residual evidence",
}


def _parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _bic(mse, k, n):
    return float(n * np.log(float(mse) + 1e-12) + int(k) * np.log(max(int(n), 1)))


def _safe_gain(base, new):
    base = float(base)
    new = float(new)
    if not np.isfinite(base) or not np.isfinite(new) or base <= 1e-12:
        return 0.0
    return float((base - new) / base)


def _clip01(x):
    if not np.isfinite(float(x)):
        return 0.0
    return float(min(max(float(x), 0.0), 1.0))


def _ev(evidence, key, default=0.0, absolute=False):
    val = float(evidence.get(key, default) or 0.0)
    return abs(val) if absolute else val


def _safe_corr(x, y):
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3 or float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def _cumtrapz_by_time(t, y):
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    out = np.zeros_like(yy, dtype=float)
    if yy.size < 2:
        return out
    dt = np.diff(tt)
    area = 0.5 * (yy[1:] + yy[:-1]) * np.maximum(dt, 0.0)
    out[1:] = np.cumsum(area)
    return out


def _linear_slope(x, y):
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3 or float(np.std(xx)) <= 1e-12:
        return 0.0
    coef = np.polyfit(xx, yy, 1)
    return float(coef[0])


def _derived_mechanism_evidence(df, fit):
    pred_by_index = fit.get("pred_by_index", {}) or {}
    if not pred_by_index:
        return {k: 0.0 for k in DERIVED_GATE_FEATURE_COLUMNS}
    work = df[["sid", "time", "C_obs", "R_obs"]].copy()
    work["resid"] = work.index.map(lambda idx: pred_by_index.get(int(idx), (np.nan, np.nan))[1])
    work = work.dropna(subset=["resid"]).sort_values(["sid", "time"])
    if work.empty:
        return {k: 0.0 for k in DERIVED_GATE_FEATURE_COLUMNS}

    response_scale = float(np.std(work["R_obs"].to_numpy(dtype=float)) + 1e-8)
    all_auc_c, all_auc_hill, all_dcdt, all_gap, all_resid = [], [], [], [], []
    all_late_auc, all_late_resid = [], []
    all_r, all_r_lag, all_time, all_c = [], [], [], []
    recovery_scores, turnover_scores = [], []
    dose_rows = []
    ec50 = float(np.median(work.loc[work["C_obs"] > 1e-8, "C_obs"])) if np.any(work["C_obs"].to_numpy(dtype=float) > 1e-8) else 1.0

    for _, grp in work.groupby("sid", sort=False):
        grp = grp.sort_values("time")
        if len(grp) < 4:
            continue
        tt = grp["time"].to_numpy(dtype=float)
        cc = grp["C_obs"].to_numpy(dtype=float)
        rr = grp["R_obs"].to_numpy(dtype=float)
        ee = grp["resid"].to_numpy(dtype=float)
        hill = (cc ** 2.0) / ((ec50 ** 2.0) + (cc ** 2.0) + 1e-12)
        auc_c = _cumtrapz_by_time(tt, cc)
        auc_hill = _cumtrapz_by_time(tt, hill)
        dcdt = np.gradient(cc, tt, edge_order=1) if len(np.unique(tt)) > 2 else np.zeros_like(cc)
        c_lag = np.r_[cc[0], cc[:-1]]
        r_lag = np.r_[rr[0], rr[:-1]]
        gap = c_lag - cc
        all_auc_c.extend(auc_c.tolist())
        all_auc_hill.extend(auc_hill.tolist())
        all_dcdt.extend(dcdt.tolist())
        all_gap.extend(gap.tolist())
        all_resid.extend(ee.tolist())
        all_r.extend(rr.tolist())
        all_r_lag.extend(r_lag.tolist())
        all_time.extend(tt.tolist())
        all_c.extend(cc.tolist())
        dose_rows.append((float(np.max(cc)), float(np.sqrt(np.mean(ee ** 2)))))

        late = tt >= np.quantile(tt, 0.60)
        low_c = cc <= max(np.quantile(cc, 0.30), 1e-8)
        if np.any(late & low_c):
            recovery_scores.append(abs(float(np.mean(ee[late & low_c]))) / response_scale)
        if np.any(late):
            all_late_auc.extend(auc_hill[late].tolist())
            all_late_resid.extend(ee[late].tolist())
        early = tt <= np.quantile(tt, 0.35)
        if np.any(early) and np.any(late):
            turnover_scores.append(abs(float(np.mean(ee[late]) - np.mean(ee[early]))) / response_scale)

    resid = np.asarray(all_resid, dtype=float)
    time = np.asarray(all_time, dtype=float)
    c_arr = np.asarray(all_c, dtype=float)
    sin24 = np.sin(2.0 * np.pi * time / 24.0)
    cos24 = np.cos(2.0 * np.pi * time / 24.0)
    circ = 0.0
    circ_detrended = 0.0
    if resid.size >= 6 and float(np.var(resid)) > 1e-12:
        x = np.column_stack([np.ones_like(time), sin24, cos24])
        beta, *_ = np.linalg.lstsq(x, resid, rcond=None)
        pred = x @ beta
        circ = _clip01(float(np.var(pred)) / max(float(np.var(resid)), 1e-12))
        trend_x = np.column_stack([np.ones_like(time), time])
        trend_beta, *_ = np.linalg.lstsq(trend_x, resid, rcond=None)
        detrended = resid - trend_x @ trend_beta
        if float(np.var(detrended)) > 1e-12:
            beta_dt, *_ = np.linalg.lstsq(x, detrended, rcond=None)
            pred_dt = x @ beta_dt
            circ_detrended = _clip01(float(np.var(pred_dt)) / max(float(np.var(detrended)), 1e-12))

    zero_monotonic = 0.0
    cmax_by_sid = work.groupby("sid")["C_obs"].max()
    zero_sids = set(cmax_by_sid[cmax_by_sid <= 1e-8].index.tolist())
    if zero_sids:
        dz = work[work["sid"].isin(zero_sids)].groupby("time", as_index=False).agg(resid=("resid", "mean"))
        if len(dz) >= 4:
            slope = _linear_slope(dz["time"], dz["resid"])
            span = float(dz["time"].max() - dz["time"].min() + 1e-8)
            zero_monotonic = _clip01(abs(slope) * span / response_scale)

    exposure_dependence = abs(_safe_corr(c_arr, resid))
    if len(dose_rows) >= 3:
        dose_tab = pd.DataFrame(dose_rows, columns=["cmax", "rmse"])
        exposure_dependence = max(exposure_dependence, abs(_safe_corr(dose_tab["cmax"], dose_tab["rmse"])))

    auc_corr = abs(_safe_corr(all_auc_c, all_resid))
    auc_hill_corr = abs(_safe_corr(all_auc_hill, all_resid))
    late_auc_direction = abs(_safe_corr(all_late_auc, all_late_resid))
    late_auc_group_shift = 0.0
    if len(all_late_auc) >= 6:
        auc_arr = np.asarray(all_late_auc, dtype=float)
        res_arr = np.asarray(all_late_resid, dtype=float)
        hi = auc_arr >= np.quantile(auc_arr, 0.67)
        lo = auc_arr <= np.quantile(auc_arr, 0.33)
        if np.any(hi) and np.any(lo):
            late_auc_group_shift = _clip01(abs(float(np.mean(res_arr[hi]) - np.mean(res_arr[lo]))) / response_scale)
    dcdt_corr = abs(_safe_corr(all_dcdt, all_resid))
    gap_corr = abs(_safe_corr(all_gap, all_resid))
    r_corr = abs(_safe_corr(all_r, all_resid))
    r_lag_corr = abs(_safe_corr(all_r_lag, all_resid))
    sync_corr = abs(_safe_corr(c_arr, resid))
    lag_advantage = _clip01(max(0.0, gap_corr - sync_corr) / 0.35)

    return {
        "auc_residual_correlation_score": _clip01(auc_corr),
        "auc_hill_residual_correlation_score": _clip01(auc_hill_corr),
        "exposure_accumulation_score": _clip01(max(auc_corr, auc_hill_corr)),
        "dcdt_residual_correlation_score": _clip01(dcdt_corr),
        "effect_site_gap_correlation_score": _clip01(gap_corr),
        "lag_advantage_score": lag_advantage,
        "circadian_projection_score": circ,
        "circadian_detrended_projection_score": circ_detrended,
        "zero_dose_monotonic_score": zero_monotonic,
        "exposure_independence_score": _clip01(1.0 - exposure_dependence),
        "late_auc_residual_direction_score": _clip01(late_auc_direction),
        "late_auc_group_shift_score": late_auc_group_shift,
        "response_state_residual_correlation_score": _clip01(r_corr),
        "lagged_response_residual_correlation_score": _clip01(r_lag_corr),
        "low_exposure_recovery_score": _clip01(float(np.mean(recovery_scores)) / 1.5) if recovery_scores else 0.0,
        "turnover_asymmetry_score": _clip01(float(np.mean(turnover_scores)) / 1.5) if turnover_scores else 0.0,
    }


def _module_gate_score(module, evidence):
    hys = _ev(evidence, "hysteresis_score")
    sync_corr = _ev(evidence, "sync_correlation", absolute=True)
    zero = _ev(evidence, "zero_dose_trend_score")
    drift = _ev(evidence, "early_late_drift_score")
    late = _ev(evidence, "late_overprediction_score")
    bias = _ev(evidence, "residual_bias_score")
    temporal = _ev(evidence, "temporal_shape_score")
    dose = _ev(evidence, "dose_residual_correlation_score")
    response_lag = _ev(evidence, "response_peak_lag_score")
    residual_lag = _ev(evidence, "residual_peak_lag_score")
    residual_loop = _ev(evidence, "residual_loop_area_score")
    response_loop = _ev(evidence, "response_loop_area_score")
    autocorr = _ev(evidence, "residual_autocorr_score")
    persistence = _ev(evidence, "residual_sign_persistence_score")
    late_slope = _ev(evidence, "late_residual_slope_score")
    poor = _ev(evidence, "poor_fit_score")
    auc = _ev(evidence, "exposure_accumulation_score")
    auc_hill = _ev(evidence, "auc_hill_residual_correlation_score")
    dcdt = _ev(evidence, "dcdt_residual_correlation_score")
    gap = _ev(evidence, "effect_site_gap_correlation_score")
    lag_adv = _ev(evidence, "lag_advantage_score")
    circadian = _ev(evidence, "circadian_projection_score")
    circadian_dt = _ev(evidence, "circadian_detrended_projection_score")
    zero_mono = _ev(evidence, "zero_dose_monotonic_score")
    exposure_independent = _ev(evidence, "exposure_independence_score")
    late_auc_direction = _ev(evidence, "late_auc_residual_direction_score")
    late_auc_shift = _ev(evidence, "late_auc_group_shift_score")
    r_state = _ev(evidence, "response_state_residual_correlation_score")
    r_lag = _ev(evidence, "lagged_response_residual_correlation_score")
    recovery = _ev(evidence, "low_exposure_recovery_score")
    turnover = _ev(evidence, "turnover_asymmetry_score")

    if module == "biophase":
        return _clip01(0.35 * gap + 0.25 * dcdt + 0.20 * hys + 0.15 * response_lag + 0.05 * poor - 0.15 * auc)
    if module == "delay":
        return _clip01(0.35 * lag_adv + 0.25 * residual_lag + 0.20 * response_lag + 0.15 * gap + 0.05 * dcdt - 0.10 * auc)
    if module == "feedback":
        return _clip01(0.35 * r_state + 0.30 * r_lag + 0.15 * response_loop + 0.10 * residual_loop + 0.10 * sync_corr)
    if module == "tolerance":
        return _clip01(0.30 * late_auc_direction + 0.25 * late_auc_shift + 0.20 * auc + 0.10 * auc_hill + 0.10 * late + 0.05 * persistence - 0.10 * circadian_dt)
    if module == "circadian":
        return _clip01(0.80 * circadian_dt + 0.10 * autocorr + 0.05 * temporal + 0.05 * persistence)
    if module == "disease":
        if zero_mono < 0.20 or exposure_independent < 0.45:
            return _clip01(0.25 * zero_mono + 0.15 * exposure_independent)
        return _clip01(0.45 * zero_mono + 0.25 * exposure_independent + 0.15 * zero + 0.10 * late_slope + 0.05 * drift - 0.25 * auc)
    if module == "precursor":
        return _clip01(0.35 * recovery + 0.25 * turnover + 0.15 * residual_lag + 0.15 * persistence + 0.10 * residual_loop - 0.10 * auc)
    return 0.0


def _feature_values_for_classifier(evidence, residual_evidence):
    protocol = evidence.get("surrogate_protocol", {}) or {}
    minimal = protocol.get("minimal", {}) or {}
    mechanistic = protocol.get("mechanistic", {}) or {}
    empirical = protocol.get("empirical", protocol.get("full", {}) or {}) or {}
    values = {
        "mechanistic_mse": float(mechanistic.get("mse", 0.0) or 0.0),
        "empirical_mse": float(empirical.get("mse", 0.0) or 0.0),
        "minimal_to_mechanistic_gain": float(protocol.get("minimal_to_mechanistic_gain", 0.0) or 0.0),
        "mechanistic_to_empirical_gain": float(protocol.get("mechanistic_to_empirical_gain", 0.0) or 0.0),
    }
    if "mechanistic_mse" not in values:
        values["mechanistic_mse"] = float(minimal.get("mse", 0.0) or 0.0)
    for col in CLASSIFIER_FEATURE_COLUMNS:
        if col not in values:
            values[col] = float(residual_evidence.get(col, evidence.get(col, 0.0)) or 0.0)
    return values


def _proposal_from_evidence(evidence):
    scores = _residual_scores_from_evidence(evidence)
    hys = float(evidence.get("hysteresis_score", 0.0) or 0.0)
    drift = float(evidence.get("early_late_drift_score", 0.0) or 0.0)
    temporal = float(evidence.get("temporal_shape_score", 0.0) or 0.0)
    dose = float(evidence.get("dose_stratified_rmse_score", 0.0) or 0.0)
    poor = float(evidence.get("poor_fit_score", 0.0) or 0.0)
    scores["precursor"] = 100.0 * min(max(0.35 * drift + 0.30 * temporal + 0.20 * dose + 0.15 * poor, 0.0), 1.0)
    rows = [
        {"mechanism": mech, "residual_score": float(score)}
        for mech, score in scores.items()
        if mech in REPAIR_MODULE_TERMS
    ]
    rows.sort(key=lambda r: r["residual_score"], reverse=True)
    return rows


def _fit_terms(df, terms, ec50_grid, gamma_grid, theta_bound):
    fit = _fit_surrogate_layer_grid(df, terms, ec50_grid, gamma_grid, theta_bound=theta_bound)
    if not fit.get("ok", False):
        return fit
    n = int(fit.get("n_points", len(df)))
    fit["bic"] = _bic(float(fit.get("mse", np.nan)), len(terms), n)
    return fit


def _fit_h0_with_fallback(df, ec50_grid, gamma_grid, theta_bound):
    failures = []
    for terms in [H0_REPAIR_TERMS, *H0_FALLBACK_TERM_SETS]:
        fit = _fit_terms(df, list(terms), ec50_grid, gamma_grid, theta_bound)
        if fit.get("ok", False):
            fit["h0_terms"] = list(terms)
            fit["h0_fallback_used"] = list(terms) != list(H0_REPAIR_TERMS)
            return fit
        failures.append({"terms": list(terms), "reason": fit.get("reason", "unknown_failure")})
    return {"ok": False, "reason": "all_h0_term_sets_failed", "failures": failures}


def _strip_fit(fit):
    return {k: v for k, v in (fit or {}).items() if k not in {"residual", "prediction", "pred_by_index"}}


def _run_one_case(args):
    model_name = str(args["model_name"])
    seed = int(args["seed"])
    label = _label_for_model(model_name)
    true_mechanism = LABEL_TO_MECHANISM.get(label, "unknown")
    case_id = f"{model_name}__seed{seed}"
    try:
        pop_data, _, _ = generate_population_data(
            model_name=model_name,
            seed=seed,
            n_subjects=int(args["n_subjects"]),
            extra_pk_iiv_sigma=0.0,
            return_pk_scale=False,
            pk_route=args["pk_route"],
            pk_compartments=int(args["pk_compartments"]),
            dose_design_enabled=True,
            dose_levels=list(args["dose_levels"]),
            disable_iiv=bool(args["disable_iiv"]),
            disable_quality_guard=bool(args["disable_quality_guard"]),
        )
        df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
        df["case_id"] = case_id
        df["pd_model"] = model_name
        df["label"] = label
        df["true_mechanism"] = true_mechanism
        df["seed"] = seed

        config = dict(DEFAULTS)
        protocol = build_fixed_surrogate_protocol_evidence(df[["sid", "time", "C_obs", "R_obs"]], config=config)
        c_pos = df["C_obs"].to_numpy(dtype=float)
        c_pos = c_pos[np.isfinite(c_pos) & (c_pos > 1e-8)]
        ec50 = float(np.median(c_pos)) if c_pos.size else 4.0
        ec50_grid = sorted({ec50, 4.0})
        gamma_grid = list(config.get("surrogate_gamma_grid", [1.5, 2.0, 3.0]))
        theta_bound = float(config.get("surrogate_theta_abs_bound", 500.0))

        h0_fit = _fit_h0_with_fallback(df[["sid", "time", "C_obs", "R_obs"]], ec50_grid, gamma_grid, theta_bound)
        if not h0_fit.get("ok", False):
            return {
                "case": {
                    "case_id": case_id,
                    "pd_model": model_name,
                    "label": label,
                    "true_mechanism": true_mechanism,
                    "seed": seed,
                    "status": "failed",
                    "error": h0_fit.get("reason", "h0_fit_failed"),
                },
                "repairs": [],
                "raw_long": df,
                "detail": {"protocol": protocol, "h0_fit": _strip_fit(h0_fit)},
            }

        h0_evidence = _residual_evidence_from_multitrajectory_fit(df[["sid", "time", "C_obs", "R_obs"]], h0_fit)
        h0_evidence.update(_derived_mechanism_evidence(df[["sid", "time", "C_obs", "R_obs"]], h0_fit))
        proposals = _proposal_from_evidence(h0_evidence)
        proposed = proposals[0]["mechanism"] if proposals else None
        h0_mse = float(h0_fit["mse"])
        h0_bic = float(h0_fit["bic"])

        h0_terms = list(h0_fit.get("h0_terms", H0_REPAIR_TERMS))
        repair_rows = []
        best_row = None
        for module, extra_terms in REPAIR_MODULE_TERMS.items():
            gate_score = _module_gate_score(module, h0_evidence)
            terms = list(dict.fromkeys(list(h0_terms) + list(extra_terms)))
            fit = _fit_terms(df[["sid", "time", "C_obs", "R_obs"]], terms, ec50_grid, gamma_grid, theta_bound)
            ok = bool(fit.get("ok", False))
            mse = float(fit.get("mse", np.nan)) if ok else np.nan
            bic = float(fit.get("bic", np.nan)) if ok else np.nan
            row = {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "true_mechanism": true_mechanism,
                "seed": seed,
                "module": module,
                "is_true_module": bool(module == true_mechanism),
                "is_proposed_module": bool(module == proposed),
                "module_gate_score": float(gate_score),
                "module_gate_description": MODULE_GATE_DESCRIPTIONS.get(module, ""),
                "ok": ok,
                "n_terms": int(len(terms)),
                "h0_mse": h0_mse,
                "repair_mse": mse,
                "delta_mse": float(h0_mse - mse) if ok else np.nan,
                "rel_mse_gain": _safe_gain(h0_mse, mse) if ok else np.nan,
                "h0_bic": h0_bic,
                "repair_bic": bic,
                "delta_bic": float(h0_bic - bic) if ok else np.nan,
                "terms": ",".join(terms),
                "error": "" if ok else str(fit.get("reason", "repair_fit_failed")),
            }
            repair_rows.append(row)
            if ok and (best_row is None or float(row["delta_bic"]) > float(best_row["delta_bic"])):
                best_row = row

        best_module = None if best_row is None else str(best_row["module"])
        proposed_row = next((r for r in repair_rows if r["module"] == proposed), None)
        true_row = next((r for r in repair_rows if r["module"] == true_mechanism), None)
        best_negative_gain = max(
            [float(r["delta_bic"]) for r in repair_rows if r["module"] not in {true_mechanism, "observable"} and np.isfinite(float(r["delta_bic"]))],
            default=np.nan,
        )
        classifier_features = _feature_values_for_classifier(protocol, h0_evidence)
        case = {
            "case_id": case_id,
            "pd_model": model_name,
            "label": label,
            "true_mechanism": true_mechanism,
            "seed": seed,
            "status": "success",
            "expected_h0": bool(true_mechanism == "observable"),
            "h0_accepted": bool(protocol.get("h0_accepted", False)) if protocol.get("available", False) else False,
            "h0_verdict": str(protocol.get("h0_verdict", "unavailable")),
            "h0_mse": h0_mse,
            "h0_rmse": float(np.sqrt(h0_mse)),
            "h0_bic": h0_bic,
            "h0_terms": ",".join(h0_terms),
            "h0_fallback_used": bool(h0_fit.get("h0_fallback_used", False)),
            "h0_poor_fit_score": float(h0_evidence.get("poor_fit_score", 0.0) or 0.0),
            "h0_structure_score": float(h0_evidence.get("amplitude_weighted_structure_score", h0_evidence.get("raw_residual_structure_score", 0.0)) or 0.0),
            **{k: float(classifier_features.get(k, 0.0) or 0.0) for k in CLASSIFIER_FEATURE_COLUMNS},
            **{f"guide_{k}": float(h0_evidence.get(k, 0.0) or 0.0) for k in GUIDE_FEATURE_COLUMNS},
            **{f"derived_{k}": float(h0_evidence.get(k, 0.0) or 0.0) for k in DERIVED_GATE_FEATURE_COLUMNS},
            "rule_proposed_mechanism": proposed,
            "rule_proposal_score": float(proposals[0]["residual_score"]) if proposals else 0.0,
            "proposed_mechanism": proposed,
            "proposal_score": float(proposals[0]["residual_score"]) if proposals else 0.0,
            "proposal_rank_true": next((i + 1 for i, r in enumerate(proposals) if r["mechanism"] == true_mechanism), np.nan),
            "best_repair_module": best_module,
            "best_repair_delta_bic": np.nan if best_row is None else float(best_row["delta_bic"]),
            "best_repair_rel_mse_gain": np.nan if best_row is None else float(best_row["rel_mse_gain"]),
            "proposed_delta_bic": np.nan if proposed_row is None else float(proposed_row["delta_bic"]),
            "proposed_rel_mse_gain": np.nan if proposed_row is None else float(proposed_row["rel_mse_gain"]),
            "true_delta_bic": np.nan if true_row is None else float(true_row["delta_bic"]),
            "true_rel_mse_gain": np.nan if true_row is None else float(true_row["rel_mse_gain"]),
            "specificity_gap_bic": np.nan if true_row is None or not np.isfinite(best_negative_gain) else float(true_row["delta_bic"] - best_negative_gain),
            "proposal_matches_true": bool(proposed == true_mechanism),
            "best_repair_matches_true": bool(best_module == true_mechanism),
            "repair_success": bool(proposed_row is not None and float(proposed_row["delta_bic"]) > 0.0 and float(proposed_row["rel_mse_gain"]) > 0.02),
        }
        return {
            "case": case,
            "repairs": repair_rows,
            "raw_long": df,
            "detail": {
                "protocol": protocol,
                "h0_fit": _strip_fit(h0_fit),
                "h0_evidence": h0_evidence,
                "proposals": proposals,
            },
        }
    except Exception as exc:
        return {
            "case": {
                "case_id": case_id,
                "pd_model": model_name,
                "label": label,
                "true_mechanism": true_mechanism,
                "seed": seed,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            },
            "repairs": [],
            "raw_long": pd.DataFrame(),
            "detail": {"traceback": traceback.format_exc()},
        }


def _plot_heatmap(path, matrix, title, fmt=".2f"):
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(matrix.columns)), max(4.5, 0.6 * len(matrix.index))))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdYlGn")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            ax.text(j, i, format(float(val), fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_h0_vs_repair(path, cases):
    data = cases[cases["status"].eq("success")].copy()
    data = data[~data["expected_h0"].astype(bool)].copy()
    if data.empty:
        return
    agg = data.groupby("true_mechanism", as_index=False).agg(
        h0_rmse=("h0_rmse", "mean"),
        repaired_rmse=("proposed_rel_mse_gain", lambda x: np.nan),
        rel_gain=("proposed_rel_mse_gain", "mean"),
    )
    agg["repaired_rmse"] = agg["h0_rmse"] * (1.0 - agg["rel_gain"].fillna(0.0))
    x = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, agg["h0_rmse"], width, label="H0 surrogate")
    ax.bar(x + width / 2, agg["repaired_rmse"], width, label="proposed repair")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["true_mechanism"], rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("H0 vs proposed mechanism repair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_classifier(seed):
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            random_state=int(seed),
            class_weight="balanced",
        ),
    )


def _basic_repair_success(row):
    if row is None:
        return False
    delta = float(row.get("delta_bic", np.nan))
    gain = float(row.get("rel_mse_gain", np.nan))
    return bool(delta > 0.0 and gain > 0.02)


def _specificity_repair_success(row):
    return bool(_basic_repair_success(row) and float(row.get("specificity_score", np.nan)) > 0.0)


def _annotate_repair_specificity(repairs, observable_fp_penalty, gate_threshold, gate_penalty):
    if repairs.empty:
        return repairs
    out = repairs.copy()
    out["repair_pass_basic"] = (
        out["ok"].astype(bool)
        & (pd.to_numeric(out["delta_bic"], errors="coerce") > 0.0)
        & (pd.to_numeric(out["rel_mse_gain"], errors="coerce") > 0.02)
    )
    obs = out[out["true_mechanism"].eq("observable") & out["ok"].astype(bool)].copy()
    if obs.empty:
        fp_rate = pd.Series(0.0, index=sorted(out["module"].dropna().unique()))
    else:
        fp_rate = obs.groupby("module")["repair_pass_basic"].mean()
    out["observable_false_positive_rate"] = out["module"].map(fp_rate).fillna(0.0).astype(float)
    out["observable_fp_penalty"] = out["observable_false_positive_rate"] * float(observable_fp_penalty)
    if "module_gate_score" in out.columns:
        gate_score = pd.to_numeric(out["module_gate_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        gate_score = pd.Series(1.0, index=out.index)
    out["module_gate_score"] = gate_score
    out["module_gate_supported"] = out["module_gate_score"] >= float(gate_threshold)
    out["module_gate_penalty"] = (float(gate_threshold) - out["module_gate_score"]).clip(lower=0.0) * float(gate_penalty)

    delta = pd.to_numeric(out["delta_bic"], errors="coerce")
    competitor = []
    for idx, row in out.iterrows():
        same = out[
            out["case_id"].eq(row["case_id"])
            & out["ok"].astype(bool)
            & ~out["module"].eq(row["module"])
        ]
        competitor.append(float(pd.to_numeric(same["delta_bic"], errors="coerce").max()) if not same.empty else 0.0)
    out["best_competing_delta_bic"] = competitor
    out["specificity_score"] = (
        delta.fillna(-np.inf)
        - out["best_competing_delta_bic"].fillna(0.0)
        - out["observable_fp_penalty"]
        - out["module_gate_penalty"]
    )
    out.loc[~out["ok"].astype(bool), "specificity_score"] = -np.inf
    out["specificity_success"] = out["repair_pass_basic"] & (out["specificity_score"] > 0.0)
    out["specificity_score_for_mean"] = out["specificity_score"].replace([np.inf, -np.inf], np.nan)
    return out


def _update_case_from_proposal(row, repairs_by_case, proposed, score=np.nan, rank=np.nan):
    case_id = row["case_id"]
    repair_rows = repairs_by_case.get(case_id, pd.DataFrame())
    proposed_row = repair_rows[repair_rows["module"].eq(proposed)] if not repair_rows.empty and proposed is not None else pd.DataFrame()
    true_mechanism = row.get("true_mechanism")
    best_row = None
    best_bic_row = None
    if not repair_rows.empty:
        ok_rows = repair_rows[repair_rows["ok"].astype(bool)].copy()
        if not ok_rows.empty:
            best_bic_row = ok_rows.sort_values("delta_bic", ascending=False).iloc[0]
            if "specificity_score" in ok_rows.columns:
                best_row = ok_rows.sort_values("specificity_score", ascending=False).iloc[0]
            else:
                best_row = best_bic_row
    true_row = repair_rows[repair_rows["module"].eq(true_mechanism)] if not repair_rows.empty else pd.DataFrame()
    best_negative_gain = np.nan
    if not repair_rows.empty:
        neg = repair_rows[
            repair_rows["ok"].astype(bool)
            & ~repair_rows["module"].isin([true_mechanism, "observable"])
        ]
        if not neg.empty:
            best_negative_gain = float(neg["delta_bic"].max())
    out = row.copy()
    out["proposed_mechanism"] = proposed
    out["proposal_score"] = float(score) if np.isfinite(float(score)) else np.nan
    out["proposal_rank_true"] = rank
    out["proposal_matches_true"] = bool(proposed == true_mechanism)
    out["best_repair_module"] = None if best_row is None else str(best_row["module"])
    out["best_repair_delta_bic"] = np.nan if best_row is None else float(best_row["delta_bic"])
    out["best_repair_rel_mse_gain"] = np.nan if best_row is None else float(best_row["rel_mse_gain"])
    out["best_repair_specificity_score"] = np.nan if best_row is None else float(best_row.get("specificity_score", np.nan))
    out["best_bic_repair_module"] = None if best_bic_row is None else str(best_bic_row["module"])
    out["best_bic_repair_delta_bic"] = np.nan if best_bic_row is None else float(best_bic_row["delta_bic"])
    if proposed_row.empty:
        out["proposed_delta_bic"] = np.nan
        out["proposed_rel_mse_gain"] = np.nan
        out["proposed_specificity_score"] = np.nan
        out["proposed_observable_fp_rate"] = np.nan
        out["proposed_gate_score"] = np.nan
        out["proposed_gate_penalty"] = np.nan
        out["proposed_gate_supported"] = False
        out["repair_success"] = False
        out["specificity_success"] = False
    else:
        pr = proposed_row.iloc[0]
        out["proposed_delta_bic"] = float(pr["delta_bic"])
        out["proposed_rel_mse_gain"] = float(pr["rel_mse_gain"])
        out["proposed_specificity_score"] = float(pr.get("specificity_score", np.nan))
        out["proposed_observable_fp_rate"] = float(pr.get("observable_false_positive_rate", np.nan))
        out["proposed_gate_score"] = float(pr.get("module_gate_score", np.nan))
        out["proposed_gate_penalty"] = float(pr.get("module_gate_penalty", np.nan))
        out["proposed_gate_supported"] = bool(pr.get("module_gate_supported", False))
        out["repair_success"] = _basic_repair_success(pr)
        out["specificity_success"] = _specificity_repair_success(pr)
    if true_row.empty:
        out["true_delta_bic"] = np.nan
        out["true_rel_mse_gain"] = np.nan
        out["specificity_gap_bic"] = np.nan
        out["true_specificity_score"] = np.nan
        out["true_gate_score"] = np.nan
    else:
        tr = true_row.iloc[0]
        out["true_delta_bic"] = float(tr["delta_bic"])
        out["true_rel_mse_gain"] = float(tr["rel_mse_gain"])
        out["specificity_gap_bic"] = np.nan if not np.isfinite(best_negative_gain) else float(tr["delta_bic"] - best_negative_gain)
        out["true_specificity_score"] = float(tr.get("specificity_score", np.nan))
        out["true_gate_score"] = float(tr.get("module_gate_score", np.nan))
    out["best_repair_matches_true"] = bool(out["best_repair_module"] == true_mechanism)
    out["best_bic_repair_matches_true"] = bool(out["best_bic_repair_module"] == true_mechanism)
    return out


def _apply_rule_guide(cases, repairs):
    repairs_by_case = {cid: grp for cid, grp in repairs.groupby("case_id")} if not repairs.empty else {}
    rows = []
    for _, row in cases.iterrows():
        if row.get("status") != "success" or bool(row.get("expected_h0", False)):
            rows.append(row.copy())
            continue
        rows.append(
            _update_case_from_proposal(
                row,
                repairs_by_case,
                row.get("rule_proposed_mechanism"),
                score=row.get("rule_proposal_score", np.nan),
                rank=row.get("proposal_rank_true", np.nan),
            )
        )
    return pd.DataFrame(rows)


def _apply_oracle_guide(cases, repairs):
    repairs_by_case = {cid: grp for cid, grp in repairs.groupby("case_id")} if not repairs.empty else {}
    rows = []
    for _, row in cases.iterrows():
        if row.get("status") != "success" or bool(row.get("expected_h0", False)):
            rows.append(row.copy())
            continue
        rows.append(_update_case_from_proposal(row, repairs_by_case, row["true_mechanism"], score=1.0, rank=1.0))
    return pd.DataFrame(rows)


def _prepare_classifier_matrix(data, feature_cols):
    usable = []
    mapped = {}
    for col in feature_cols:
        if col in data.columns:
            mapped[col] = col
            usable.append(col)
        elif f"guide_{col}" in data.columns:
            mapped[col] = f"guide_{col}"
            usable.append(col)
    if not usable:
        raise ValueError("No classifier feature columns are available in repair cases.")
    x = pd.DataFrame(index=data.index)
    for col in feature_cols:
        source = mapped.get(col)
        if source is None:
            x[col] = 0.0
        else:
            x[col] = pd.to_numeric(data[source], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _label_prediction_to_mechanism(label):
    text = str(label)
    return LABEL_TO_MECHANISM.get(text, text)


def _apply_classifier_guide(cases, repairs, seed):
    data = cases[cases["status"].eq("success") & ~cases["expected_h0"].astype(bool)].copy()
    repairs_by_case = {cid: grp for cid, grp in repairs.groupby("case_id")} if not repairs.empty else {}
    if data.empty or data["true_mechanism"].nunique() < 2 or data["seed"].nunique() < 2:
        return cases, pd.DataFrame()
    feature_cols = [c for c in CLASSIFIER_FEATURE_COLUMNS if c in data.columns or f"guide_{c}" in data.columns]
    x_all = _prepare_classifier_matrix(data, feature_cols)

    pred_rows = []
    proposed_by_case = {}
    for holdout_seed in sorted(data["seed"].unique()):
        train = data["seed"].ne(holdout_seed)
        test = data["seed"].eq(holdout_seed)
        if data.loc[train, "true_mechanism"].nunique() < 2:
            continue
        clf = _make_classifier(int(seed) + int(holdout_seed))
        clf.fit(x_all.loc[train, feature_cols], data.loc[train, "true_mechanism"].astype(str))
        pred = clf.predict(x_all.loc[test, feature_cols])
        proba = clf.predict_proba(x_all.loc[test, feature_cols])
        classes = list(clf.classes_)
        for j, idx in enumerate(data.index[test]):
            probs = {cls: float(val) for cls, val in zip(classes, proba[j])}
            ranking = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            true_mech = str(data.loc[idx, "true_mechanism"])
            rank_true = next((k + 1 for k, (cls, _) in enumerate(ranking) if cls == true_mech), np.nan)
            case_id = str(data.loc[idx, "case_id"])
            proposed_by_case[case_id] = {
                "mechanism": str(pred[j]),
                "score": float(probs.get(str(pred[j]), np.nan)),
                "rank_true": rank_true,
            }
            row = {
                "case_id": case_id,
                "seed": int(holdout_seed),
                "true_mechanism": true_mech,
                "predicted_mechanism": str(pred[j]),
                "proposal_rank_true": rank_true,
            }
            for cls, val in probs.items():
                row[f"p_{cls}"] = val
            pred_rows.append(row)

    rows = []
    for _, row in cases.iterrows():
        case_id = str(row.get("case_id"))
        if row.get("status") != "success" or bool(row.get("expected_h0", False)) or case_id not in proposed_by_case:
            rows.append(row.copy())
            continue
        pred = proposed_by_case[case_id]
        rows.append(_update_case_from_proposal(row, repairs_by_case, pred["mechanism"], pred["score"], pred["rank_true"]))
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _apply_frozen_classifier_guide(cases, repairs, classifier_dir):
    model_path = os.path.join(str(classifier_dir), "stage2_hidden_type_classifier.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Frozen classifier not found: {model_path}")
    payload = joblib.load(model_path)
    clf = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feature_cols = list(payload.get("feature_cols", CLASSIFIER_FEATURE_COLUMNS)) if isinstance(payload, dict) else list(CLASSIFIER_FEATURE_COLUMNS)
    data = cases[cases["status"].eq("success") & ~cases["expected_h0"].astype(bool)].copy()
    repairs_by_case = {cid: grp for cid, grp in repairs.groupby("case_id")} if not repairs.empty else {}
    if data.empty:
        return cases, pd.DataFrame()
    x = _prepare_classifier_matrix(data, feature_cols)
    pred_labels = clf.predict(x[feature_cols])
    proba = clf.predict_proba(x[feature_cols]) if hasattr(clf, "predict_proba") else None
    classes = list(getattr(clf, "classes_", []))
    proposed_by_case = {}
    pred_rows = []
    for j, idx in enumerate(data.index):
        case_id = str(data.loc[idx, "case_id"])
        pred_label = str(pred_labels[j])
        pred_mech = _label_prediction_to_mechanism(pred_label)
        probs = {}
        if proba is not None:
            probs = {_label_prediction_to_mechanism(cls): float(val) for cls, val in zip(classes, proba[j])}
        ranking = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        true_mech = str(data.loc[idx, "true_mechanism"])
        rank_true = next((k + 1 for k, (cls, _) in enumerate(ranking) if cls == true_mech), np.nan)
        score = float(probs.get(pred_mech, np.nan)) if probs else np.nan
        proposed_by_case[case_id] = {"mechanism": pred_mech, "score": score, "rank_true": rank_true}
        pred_row = {
            "case_id": case_id,
            "seed": int(data.loc[idx, "seed"]),
            "true_mechanism": true_mech,
            "predicted_label": pred_label,
            "predicted_mechanism": pred_mech,
            "proposal_rank_true": rank_true,
        }
        for cls, val in probs.items():
            pred_row[f"p_{cls}"] = val
        pred_rows.append(pred_row)

    rows = []
    for _, row in cases.iterrows():
        case_id = str(row.get("case_id"))
        if row.get("status") != "success" or bool(row.get("expected_h0", False)) or case_id not in proposed_by_case:
            rows.append(row.copy())
            continue
        pred = proposed_by_case[case_id]
        rows.append(_update_case_from_proposal(row, repairs_by_case, pred["mechanism"], pred["score"], pred["rank_true"]))
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _apply_guide(cases, repairs, guide_mode, seed, classifier_dir):
    mode = str(guide_mode or "classifier").lower()
    if mode == "rule":
        return _apply_rule_guide(cases, repairs), pd.DataFrame()
    if mode == "oracle":
        return _apply_oracle_guide(cases, repairs), pd.DataFrame()
    if mode == "classifier":
        return _apply_classifier_guide(cases, repairs, seed)
    if mode in {"frozen_classifier", "frozen-classifier"}:
        return _apply_frozen_classifier_guide(cases, repairs, classifier_dir)
    raise ValueError(f"Unknown guide mode: {guide_mode}")


def _write_report(path, manifest, cases, repairs):
    ok_cases = cases[cases["status"].eq("success")].copy()
    hidden = ok_cases[~ok_cases["expected_h0"].astype(bool)].copy()
    lines = [
        "# Mechanism repair benchmark",
        "",
        "## Design",
        "",
        "- Fit an observable H0 surrogate with blind-safe observable terms.",
        f"- Guide mode: `{manifest['guide_mode']}`.",
        "- Refit H0 plus each module-specific observable proxy term set.",
        "- Evaluate whether the proposed/true module reduces residual error and BIC.",
        "- Rank repair modules by specificity-adjusted score: module BIC gain minus strongest competing gain minus observable false-positive penalty.",
        "",
        "## Dataset",
        "",
        f"- cases: {len(cases)}",
        f"- successful cases: {len(ok_cases)}",
        f"- hidden successful cases: {len(hidden)}",
        f"- seeds: {manifest['n_seeds']}",
        f"- models: {manifest['n_models']}",
        "",
        "## Summary",
        "",
    ]
    if not hidden.empty:
        lines.extend(
            [
                f"- proposal top-1 accuracy on hidden: {float(hidden['proposal_matches_true'].mean()):.4f}",
                f"- best repair module accuracy on hidden: {float(hidden['best_repair_matches_true'].mean()):.4f}",
                f"- best raw-BIC repair module accuracy on hidden: {float(hidden['best_bic_repair_matches_true'].mean()):.4f}" if "best_bic_repair_matches_true" in hidden else "- best raw-BIC repair module accuracy on hidden: unavailable",
                f"- proposed repair success rate on hidden: {float(hidden['repair_success'].mean()):.4f}",
                f"- proposed specificity success rate on hidden: {float(hidden['specificity_success'].mean()):.4f}" if "specificity_success" in hidden else "- proposed specificity success rate on hidden: unavailable",
                f"- proposed gate support rate on hidden: {float(hidden['proposed_gate_supported'].mean()):.4f}" if "proposed_gate_supported" in hidden else "- proposed gate support rate on hidden: unavailable",
                f"- mean proposed BIC improvement on hidden: {float(hidden['proposed_delta_bic'].mean()):.4f}",
                f"- mean proposed relative MSE gain on hidden: {float(hidden['proposed_rel_mse_gain'].mean()):.4f}",
                f"- mean proposed specificity score on hidden: {float(hidden['proposed_specificity_score'].mean()):.4f}" if "proposed_specificity_score" in hidden else "- mean proposed specificity score on hidden: unavailable",
                "",
            ]
        )
        by_mech = hidden.groupby("true_mechanism").agg(
            n=("case_id", "count"),
            proposal_acc=("proposal_matches_true", "mean"),
            best_repair_acc=("best_repair_matches_true", "mean"),
            best_bic_repair_acc=("best_bic_repair_matches_true", "mean"),
            repair_success=("repair_success", "mean"),
            specificity_success=("specificity_success", "mean"),
            gate_support=("proposed_gate_supported", "mean"),
            proposed_gate_score=("proposed_gate_score", "mean"),
            proposed_delta_bic=("proposed_delta_bic", "mean"),
            proposed_rel_mse_gain=("proposed_rel_mse_gain", "mean"),
            specificity_gap_bic=("specificity_gap_bic", "mean"),
            proposed_specificity_score=("proposed_specificity_score", "mean"),
        ).reset_index()
        lines.extend(["## By true mechanism", "", "|mechanism|n|proposal_acc|best_spec_acc|best_bic_acc|repair_success|specificity_success|gate_support|gate_score|delta_bic|rel_mse_gain|specificity_gap_bic|specificity_score|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for _, row in by_mech.iterrows():
            lines.append(
                f"|{row['true_mechanism']}|{int(row['n'])}|{row['proposal_acc']:.4f}|"
                f"{row['best_repair_acc']:.4f}|{row['best_bic_repair_acc']:.4f}|{row['repair_success']:.4f}|{row['specificity_success']:.4f}|"
                f"{row['gate_support']:.4f}|{row['proposed_gate_score']:.4f}|"
                f"{row['proposed_delta_bic']:.4f}|{row['proposed_rel_mse_gain']:.4f}|"
                f"{row['specificity_gap_bic']:.4f}|{row['proposed_specificity_score']:.4f}|"
            )
    if not repairs.empty:
        heat = repairs[repairs["ok"].astype(bool)].pivot_table(
            index="true_mechanism",
            columns="module",
            values="rel_mse_gain",
            aggfunc="mean",
        ).fillna(0.0)
        lines.extend(["", "## Mean relative MSE gain heatmap table", ""])
        lines.append("|true\\module|" + "|".join(str(c) for c in heat.columns) + "|")
        lines.append("|" + "|".join(["---"] + ["---:"] * len(heat.columns)) + "|")
        for idx, row in heat.iterrows():
            lines.append("|" + str(idx) + "|" + "|".join(f"{float(v):.4f}" for v in row) + "|")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Benchmark residual-guided mechanism repair/refinement with surrogate-level module expansions.")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "pkpd_mechanism_repair_v1"))
    parser.add_argument("--pd-models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--pk-route", default="oral", choices=["oral", "bolus"])
    parser.add_argument("--pk-compartments", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dose-levels", default="0,0.2,1,5")
    parser.add_argument("--disable-iiv", action="store_true", default=True)
    parser.add_argument("--disable-quality-guard", action="store_true", default=True)
    parser.add_argument("--guide-mode", default="classifier", choices=["classifier", "frozen-classifier", "frozen_classifier", "rule", "oracle"])
    parser.add_argument("--classifier-dir", default=os.path.join("artifacts", "pkpd_residual_classifiers_v1"))
    parser.add_argument("--observable-fp-penalty", type=float, default=50.0)
    parser.add_argument("--gate-threshold", type=float, default=0.35)
    parser.add_argument("--gate-penalty", type=float, default=200.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    models = [x.strip() for x in args.pd_models.split(",") if x.strip()]
    dose_levels = _parse_float_list(args.dose_levels)

    jobs = []
    for rep in range(max(1, int(args.n_replicates))):
        seed = int(args.seed) + rep
        for model_name in models:
            jobs.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "n_subjects": int(args.n_subjects),
                    "pk_route": args.pk_route,
                    "pk_compartments": int(args.pk_compartments),
                    "dose_levels": dose_levels,
                    "disable_iiv": bool(args.disable_iiv),
                    "disable_quality_guard": bool(args.disable_quality_guard),
                }
            )

    outputs = []
    if int(args.workers) <= 1:
        for job in jobs:
            outputs.append(_run_one_case(job))
    else:
        with cf.ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            for out in ex.map(_run_one_case, jobs):
                outputs.append(out)

    cases = pd.DataFrame([o["case"] for o in outputs])
    repairs = pd.DataFrame([row for o in outputs for row in o["repairs"]])
    repairs = _annotate_repair_specificity(repairs, args.observable_fp_penalty, args.gate_threshold, args.gate_penalty)
    cases, guide_predictions = _apply_guide(cases, repairs, args.guide_mode, args.seed, args.classifier_dir)
    raw_long_parts = [o["raw_long"] for o in outputs if isinstance(o.get("raw_long"), pd.DataFrame) and not o["raw_long"].empty]
    raw_long = pd.concat(raw_long_parts, ignore_index=True) if raw_long_parts else pd.DataFrame()

    cases.to_csv(os.path.join(args.out_dir, "repair_cases.csv"), index=False)
    repairs.to_csv(os.path.join(args.out_dir, "repair_predictions_long.csv"), index=False)
    if not guide_predictions.empty:
        guide_predictions.to_csv(os.path.join(args.out_dir, "guide_classifier_predictions.csv"), index=False)
    if not raw_long.empty:
        raw_long.to_csv(os.path.join(args.out_dir, "raw_observations_long.csv"), index=False)

    summary = pd.DataFrame()
    if not cases.empty and not repairs.empty:
        hidden = cases[cases["status"].eq("success") & ~cases["expected_h0"].astype(bool)]
        if not hidden.empty:
            summary = hidden.groupby("true_mechanism").agg(
                n=("case_id", "count"),
                proposal_acc=("proposal_matches_true", "mean"),
                best_repair_acc=("best_repair_matches_true", "mean"),
                best_bic_repair_acc=("best_bic_repair_matches_true", "mean"),
                repair_success=("repair_success", "mean"),
                specificity_success=("specificity_success", "mean"),
                gate_support=("proposed_gate_supported", "mean"),
                proposed_gate_score=("proposed_gate_score", "mean"),
                proposed_gate_penalty=("proposed_gate_penalty", "mean"),
                proposed_delta_bic=("proposed_delta_bic", "mean"),
                proposed_rel_mse_gain=("proposed_rel_mse_gain", "mean"),
                specificity_gap_bic=("specificity_gap_bic", "mean"),
                proposed_specificity_score=("proposed_specificity_score", "mean"),
            ).reset_index()
    summary.to_csv(os.path.join(args.out_dir, "repair_summary.csv"), index=False)

    manifest = {
        "n_cases": int(len(cases)),
        "n_success": int(cases["status"].eq("success").sum()) if "status" in cases else 0,
        "n_seeds": int(cases["seed"].nunique()) if "seed" in cases else 0,
        "n_models": int(cases["pd_model"].nunique()) if "pd_model" in cases else 0,
        "guide_mode": str(args.guide_mode),
        "classifier_dir": str(args.classifier_dir),
        "observable_fp_penalty": float(args.observable_fp_penalty),
        "gate_threshold": float(args.gate_threshold),
        "gate_penalty": float(args.gate_penalty),
        "h0_terms": list(H0_REPAIR_TERMS),
        "h0_fallback_term_sets": H0_FALLBACK_TERM_SETS,
        "repair_module_terms": REPAIR_MODULE_TERMS,
        "module_gate_descriptions": MODULE_GATE_DESCRIPTIONS,
        "guide_feature_columns": GUIDE_FEATURE_COLUMNS,
        "derived_gate_feature_columns": DERIVED_GATE_FEATURE_COLUMNS,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if not repairs.empty:
        gain_heat = repairs[repairs["ok"].astype(bool)].pivot_table(index="true_mechanism", columns="module", values="rel_mse_gain", aggfunc="mean").fillna(0.0)
        bic_heat = repairs[repairs["ok"].astype(bool)].pivot_table(index="true_mechanism", columns="module", values="delta_bic", aggfunc="mean").fillna(0.0)
        spec_heat = repairs[repairs["ok"].astype(bool)].pivot_table(index="true_mechanism", columns="module", values="specificity_score_for_mean", aggfunc="mean").fillna(0.0)
        gate_heat = repairs[repairs["ok"].astype(bool)].pivot_table(index="true_mechanism", columns="module", values="module_gate_score", aggfunc="mean").fillna(0.0)
        gain_heat.to_csv(os.path.join(args.out_dir, "repair_specificity_rel_mse_gain.csv"))
        bic_heat.to_csv(os.path.join(args.out_dir, "repair_specificity_delta_bic.csv"))
        spec_heat.to_csv(os.path.join(args.out_dir, "repair_specificity_adjusted_score.csv"))
        gate_heat.to_csv(os.path.join(args.out_dir, "repair_module_gate_score.csv"))
        _plot_heatmap(os.path.join(plot_dir, "repair_specificity_rel_mse_gain.png"), gain_heat, "Mean relative MSE gain by true mechanism and repair module")
        _plot_heatmap(os.path.join(plot_dir, "repair_specificity_delta_bic.png"), bic_heat, "Mean delta BIC by true mechanism and repair module")
        _plot_heatmap(os.path.join(plot_dir, "repair_specificity_adjusted_score.png"), spec_heat, "Mean specificity-adjusted repair score")
        _plot_heatmap(os.path.join(plot_dir, "repair_module_gate_score.png"), gate_heat, "Mean mechanism evidence gate score")
    if not cases.empty:
        _plot_h0_vs_repair(os.path.join(plot_dir, "h0_vs_proposed_repair_rmse.png"), cases)

    report_path = os.path.join(args.out_dir, "repair_report.md")
    _write_report(report_path, manifest, cases, repairs)
    print(report_path)
    print(os.path.join(args.out_dir, "repair_summary.csv"))
    print(plot_dir)


if __name__ == "__main__":
    main()
