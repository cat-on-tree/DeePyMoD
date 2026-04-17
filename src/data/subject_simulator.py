import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from src.configs.pkpd_registry import t_dense, t_obs, pk_noise_cv, pd_noise_cv


def hill(C, EC50, gamma):
    C = max(float(C), 1e-10)
    return (C**gamma) / (EC50**gamma + C**gamma)


def emx(C, Emax, EC50):
    C = max(float(C), 1e-10)
    return Emax * C / (EC50 + C)


def sample_individual_params(tv_pk, omega_pk, tv_pd, omega_pd):
    p = {}

    p["D"] = tv_pk["D"]; p["F"] = tv_pk["F"]
    p["V"]  = tv_pk["V"]  * np.exp(np.random.normal(0, omega_pk["V"]))
    p["ka"] = tv_pk["ka"] * np.exp(np.random.normal(0, omega_pk["ka"]))
    p["ke"] = tv_pk["ke"] * np.exp(np.random.normal(0, omega_pk["ke"]))
    if abs(p["ka"] - p["ke"]) < 1e-3:
        p["ka"] = p["ke"] + 1e-3

    keys = ["E0","S","Emax","EC50","gamma","ke0","Kin","Kout","Imax","Smax"]
    for k in keys:
        base = tv_pd.get(k, 0.0)
        om = omega_pd.get(k, 0.0)
        p[k] = base * np.exp(np.random.normal(0, om)) if base > 0 else 0.0

    p["EC50"] = max(p["EC50"], 1e-3)
    p["gamma"] = np.clip(max(p["gamma"], 1e-3), 0.3, 8.0)
    p["ke0"] = max(p["ke0"], 1e-4) if p["ke0"] > 0 else 0.0
    p["Kin"] = max(p["Kin"], 1e-4) if p["Kin"] > 0 else 0.0
    p["Kout"] = max(p["Kout"], 1e-4) if p["Kout"] > 0 else 0.0
    p["Imax"] = np.clip(p["Imax"], 0.0, 1.5)
    p["Smax"] = np.clip(p["Smax"], 0.0, 2.0)
    return p


def simulate_subject(cfg, p):
    pre = (p["F"] * p["D"] * p["ka"]) / (p["V"] * (p["ka"] - p["ke"]))
    C_dense = pre * (np.exp(-p["ke"] * t_dense) - np.exp(-p["ka"] * t_dense))
    C_dense = np.clip(C_dense, 0.0, None)
    C_func = interp1d(t_dense, C_dense, kind="cubic", fill_value="extrapolate")

    fam = cfg["family"]
    eff = cfg.get("effect_form", None)

    if fam == "direct":
        E0 = p["E0"]
        if eff == "linear":
            E_dense = E0 + p["S"] * C_dense
        elif eff == "emax":
            E_dense = E0 + np.array([emx(c, p["Emax"], p["EC50"]) for c in C_dense])
        elif eff == "sigemax":
            E_dense = E0 + p["Emax"] * np.array([hill(c, p["EC50"], p["gamma"]) for c in C_dense])
        else:
            raise ValueError(f"Unknown direct effect_form: {eff}")
        R_dense = E_dense

    elif fam == "biophase":
        E0 = p["E0"]
        ke0 = p["ke0"]

        def rhs(z, t):
            Ce, E = z
            C_t = float(C_func(t))
            dCe = ke0 * (C_t - Ce)
            if eff == "emax":
                E_inf = E0 + emx(Ce, p["Emax"], p["EC50"])
            elif eff == "sigemax":
                E_inf = E0 + p["Emax"] * hill(Ce, p["EC50"], p["gamma"])
            else:
                raise ValueError(f"Unknown biophase effect_form: {eff}")
            dE = (E_inf - E)
            return [dCe, dE]

        z0 = [0.0, E0]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 1]

    elif fam == "idr":
        mode = cfg.get("idr_mode", "base")
        target = cfg.get("mod_target", "none")
        mtype = cfg.get("mod_type", "none")

        def rhs(R, t):
            C_t = float(C_func(t))
            H = hill(C_t, p["EC50"], p["gamma"])
            Kin, Kout = p["Kin"], p["Kout"]

            if mode == "base":
                return Kin - Kout * R

            if target == "Kin":
                if mtype == "inhib_sigmoid":
                    Kin_eff = Kin * (1 - p["Imax"] * H)
                elif mtype == "stim_sigmoid":
                    Kin_eff = Kin * (1 + p["Smax"] * H)
                else:
                    raise ValueError(f"Unknown mod_type for Kin: {mtype}")
                return Kin_eff - Kout * R

            elif target == "Kout":
                if mtype == "stim_sigmoid":
                    Kout_eff = Kout * (1 + p["Imax"] * H)
                elif mtype == "inhib_sigmoid":
                    Kout_eff = Kout * (1 - p["Smax"] * H)
                else:
                    raise ValueError(f"Unknown mod_type for Kout: {mtype}")
                Kout_eff = max(Kout_eff, 1e-8)
                return Kin - Kout_eff * R

            else:
                raise ValueError(f"Unknown mod_target: {target}")

        R0 = p["Kin"] / p["Kout"]
        R_dense = odeint(lambda r, tt: rhs(r, tt), R0, t_dense).flatten()

    else:
        raise ValueError(f"Unknown family: {fam}")

    C_obs_clean = interp1d(t_dense, C_dense, kind="cubic")(t_obs)
    R_obs_clean = interp1d(t_dense, R_dense, kind="cubic")(t_obs)

    C_obs = C_obs_clean * (1.0 + np.random.normal(0, pk_noise_cv, size=t_obs.shape))
    R_obs = R_obs_clean * (1.0 + np.random.normal(0, pd_noise_cv, size=t_obs.shape))

    C_obs = np.clip(C_obs, 0.0, None)
    R_obs = np.clip(R_obs, 1e-8, None)
    return C_obs, R_obs