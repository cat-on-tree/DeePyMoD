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


def emax_hill(C, Emax, EC50, gamma):
    return Emax * hill(C, EC50, gamma)


def sample_individual_params(tv_pk, omega_pk, tv_pd, omega_pd):
    p = {}

    p["D"] = tv_pk["D"]; p["F"] = tv_pk["F"]
    p["V"]  = tv_pk["V"]  * np.exp(np.random.normal(0, omega_pk["V"]))
    p["ka"] = tv_pk["ka"] * np.exp(np.random.normal(0, omega_pk["ka"]))
    p["ke"] = tv_pk["ke"] * np.exp(np.random.normal(0, omega_pk["ke"]))
    if abs(p["ka"] - p["ke"]) < 1e-3:
        p["ka"] = p["ke"] + 1e-3

    keys = [
        "E0","S","Emax","EC50","gamma","ke0","Kin","Kout","Imax","Smax",
        "Kon","Koff","keDR","Kin_R","Kout_R","Kin_PD1m","Kout_PD1","Kout_PD2",
        "Emax1","Emax2","EC50_2","gamma2","kt1","kt2",
        "Kin_Rm","Kin_Rb","phi1","Kin_PD1m","Kin_PD1b","phi2",
        "k_grow","Emax_kill","EC50_kill","gamma_kill","lambda_kill",
        "k_prog","k_rem","Emax_dis","EC50_dis","gamma_dis",
        "k_in_tol","k_out_tol","Smax_tol","SC50_tol","gamma_tol",
        "ka_int","Vint","ke_int","KI","Kin_P","ktr",
        "CLp","Q","V2","Vt",
    ]
    for k in keys:
        base = tv_pd.get(k, 0.0)
        om = omega_pd.get(k, 0.0)
        p[k] = base * np.exp(np.random.normal(0, om)) if base > 0 else 0.0

    p["EC50"] = max(p.get("EC50", 0.0), 1e-3) if p.get("EC50", 0.0) > 0 else 0.0
    p["gamma"] = np.clip(max(p.get("gamma", 0.0), 1e-3), 0.3, 8.0) if p.get("gamma", 0.0) > 0 else 0.0
    p["ke0"] = max(p.get("ke0", 0.0), 1e-4) if p.get("ke0", 0.0) > 0 else 0.0
    p["Kin"] = max(p.get("Kin", 0.0), 1e-4) if p.get("Kin", 0.0) > 0 else 0.0
    p["Kout"] = max(p.get("Kout", 0.0), 1e-4) if p.get("Kout", 0.0) > 0 else 0.0
    p["Imax"] = np.clip(p.get("Imax", 0.0), 0.0, 1.5) if p.get("Imax", 0.0) > 0 else 0.0
    p["Smax"] = np.clip(p.get("Smax", 0.0), 0.0, 2.0) if p.get("Smax", 0.0) > 0 else 0.0
    return p


def _bateman_conc(p):
    pre = (p["F"] * p["D"] * p["ka"]) / (p["V"] * (p["ka"] - p["ke"]))
    C_dense = pre * (np.exp(-p["ke"] * t_dense) - np.exp(-p["ka"] * t_dense))
    return np.clip(C_dense, 0.0, None)


def simulate_subject(cfg, p):
    fam = cfg["family"]
    eff = cfg.get("effect_form", None)

    # Standard PK (Bateman) for most models
    C_dense = _bateman_conc(p)
    C_func = interp1d(t_dense, C_dense, kind="cubic", fill_value="extrapolate")

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

    elif fam == "tmdd":
        def rhs(z, t):
            R, CpR, PD1, PD2 = z
            Cp = float(C_func(t))
            dR = p["Kin_R"] - p["Kout_R"] * R - p["Kon"] * Cp * R + p["Koff"] * CpR
            dCpR = p["Kon"] * Cp * R - p["Koff"] * CpR - p["keDR"] * CpR
            E1 = emax_hill(CpR, p["Emax1"], p["EC50"], p["gamma"])
            dPD1 = p["Kin_PD1m"] * (1 + E1) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dR, dCpR, dPD1, dPD2]

        R0 = p["Kin_R"] / p["Kout_R"]
        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [R0, 0.0, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 3]

    elif fam == "transduction":
        def rhs(z, t):
            T1, T2, T3, PD1, PD2 = z
            Cp = float(C_func(t))
            E_in = hill(Cp, p["EC50"], p["gamma"])
            dT1 = p["kt1"] * (E_in - T1)
            dT2 = p["kt1"] * (T1 - T2)
            dT3 = p["kt1"] * (T2 - T3)
            E1 = emax_hill(T3, p["Emax1"], p["EC50"], p["gamma"])
            dPD1 = p["Kin_PD1m"] * (1 + E1) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dT1, dT2, dT3, dPD1, dPD2]

        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [0.0, 0.0, 0.0, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 4]

    elif fam == "feedback":
        def rhs(z, t):
            T4, T5, T6, PD1, PD2 = z
            Cp = float(C_func(t))
            E1 = emax_hill(Cp, p["Emax1"], p["EC50"], p["gamma"])
            E2 = emax_hill(T6, p["Emax2"], p["EC50_2"], p["gamma2"])
            dPD1 = p["Kin_PD1m"] * (1 + E1) * (1 + E2) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            dT4 = p["kt2"] * (PD2 - T4)
            dT5 = p["kt2"] * (T4 - T5)
            dT6 = p["kt2"] * (T5 - T6)
            return [dT4, dT5, dT6, dPD1, dPD2]

        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [PD20, PD20, PD20, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 4]

    elif fam == "circadian":
        def kin_r(t):
            return p["Kin_Rm"] + p["Kin_Rb"] * np.cos((t - p["phi1"]) * 2 * np.pi / 24.0)

        def kin_pd1(t):
            return p["Kin_PD1m"] + p["Kin_PD1b"] * np.cos((t - p["phi2"]) * 2 * np.pi / 24.0)

        def rhs(z, t):
            PD1, PD2 = z
            Cp = float(C_func(t))
            E1 = emax_hill(Cp, p["Emax1"], p["EC50"], p["gamma"])
            dPD1 = kin_pd1(t) * (1 + E1) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dPD1, dPD2]

        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 1]

    elif fam == "tgi":
        def rhs(PD1, t):
            Cp = float(C_func(t))
            kill = emax_hill(Cp, p["Emax_kill"], p["EC50_kill"], p["gamma_kill"]) * np.exp(-p["lambda_kill"] * t)
            return p["k_grow"] * PD1 - kill * PD1

        PD10 = 1.0
        R_dense = odeint(lambda z, tt: rhs(z, tt), PD10, t_dense).flatten()

    elif fam == "disease":
        def rhs(z, t):
            PD1, PD2 = z
            Cp = float(C_func(t))
            Edis = emax_hill(Cp, p["Emax_dis"], p["EC50_dis"], p["gamma_dis"])
            dPD1 = p["k_prog"] - p["k_rem"] * PD1 - Edis
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dPD1, dPD2]

        PD10 = p["k_prog"] / max(p["k_rem"], 1e-6)
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 1]

    elif fam == "tolerance":
        def rhs(z, t):
            Tol, PD1, PD2 = z
            Cp = float(C_func(t))
            E1 = emax_hill(Cp, p["Emax1"], p["EC50"], p["gamma"])
            Stol = emax_hill(Cp, p["Smax_tol"], p["SC50_tol"], p["gamma_tol"])
            dTol = p["k_in_tol"] * (1 + Stol) - p["k_out_tol"] * Tol
            dPD1 = p["Kin_PD1m"] * (1 + E1) / (1 + Tol) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dTol, dPD1, dPD2]

        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [0.0, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 2]

    elif fam == "interaction":
        # Interaction drug concentration
        D_int = p.get("D", 100.0)
        ka_int = p["ka_int"]
        ke_int = p["ke_int"]
        Vint = p["Vint"]
        pre = (D_int * ka_int) / (Vint * (ka_int - ke_int))
        Cint_dense = pre * (np.exp(-ke_int * t_dense) - np.exp(-ka_int * t_dense))
        Cint_dense = np.clip(Cint_dense, 0.0, None)
        Cint_func = interp1d(t_dense, Cint_dense, kind="cubic", fill_value="extrapolate")

        def rhs(z, t):
            PD1, PD2 = z
            Cp = float(C_func(t))
            Cint = float(Cint_func(t))
            EC50_eff = p["EC50"] * (1 + Cint / max(p["KI"], 1e-6))
            Eint = emax_hill(Cp, p["Emax1"], EC50_eff, p["gamma"])
            dPD1 = p["Kin_PD1m"] * (1 + Eint) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dPD1, dPD2]

        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 1]

    elif fam == "precursor":
        def rhs(z, t):
            P, PD1, PD2 = z
            Cp = float(C_func(t))
            E1 = emax_hill(Cp, p["Emax1"], p["EC50"], p["gamma"])
            dP = p["Kin_P"] * (1 + E1) - p["ktr"] * P
            dPD1 = p["ktr"] * P - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dP, dPD1, dPD2]

        P0 = p["Kin_P"] / p["ktr"]
        PD10 = p["ktr"] * P0 / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [P0, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        R_dense = z_dense[:, 2]

    elif fam == "antibody":
        V2 = p.get("V2", p["V"])
        Vt = p.get("Vt", V2)

        def rhs(z, t):
            A1, Cp, Ct, R, CpR, PD1, PD2 = z
            dA1 = -p["ka"] * A1
            dCp = (p["ka"] / V2) * A1 - p["CLp"] * Cp / V2 - p["Q"] * (Cp - Ct) / V2 - p["Kon"] * Cp * R + p["Koff"] * CpR
            dCt = p["Q"] * (Cp - Ct) / Vt
            dR = p["Kin_R"] - p["Kout_R"] * R - p["Kon"] * Cp * R + p["Koff"] * CpR
            dCpR = p["Kon"] * Cp * R - p["Koff"] * CpR - p["keDR"] * CpR
            E1 = emax_hill(CpR, p["Emax1"], p["EC50"], p["gamma"])
            dPD1 = p["Kin_PD1m"] * (1 + E1) - p["Kout_PD1"] * PD1
            dPD2 = p["Kout_PD1"] * PD1 - p["Kout_PD2"] * PD2
            return [dA1, dCp, dCt, dR, dCpR, dPD1, dPD2]

        A10 = p["F"] * p["D"]
        R0 = p["Kin_R"] / p["Kout_R"]
        PD10 = p["Kin_PD1m"] / p["Kout_PD1"]
        PD20 = PD10 * p["Kout_PD1"] / p["Kout_PD2"]
        z0 = [A10, 0.0, 0.0, R0, 0.0, PD10, PD20]
        z_dense = odeint(lambda z, tt: rhs(z, tt), z0, t_dense)
        C_dense = np.clip(z_dense[:, 1], 0.0, None)
        C_func = interp1d(t_dense, C_dense, kind="cubic", fill_value="extrapolate")
        R_dense = z_dense[:, 6]

    else:
        raise ValueError(f"Unknown family: {fam}")

    C_obs_clean = interp1d(t_dense, C_dense, kind="cubic")(t_obs)
    R_obs_clean = interp1d(t_dense, R_dense, kind="cubic")(t_obs)

    C_obs = C_obs_clean * (1.0 + np.random.normal(0, pk_noise_cv, size=t_obs.shape))
    R_obs = R_obs_clean * (1.0 + np.random.normal(0, pd_noise_cv, size=t_obs.shape))

    C_obs = np.clip(C_obs, 0.0, None)
    R_obs = np.clip(R_obs, 1e-8, None)
    return C_obs, R_obs
