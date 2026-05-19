import numpy as np

# ----------------------------
# 0) 全局设置
# ----------------------------
GLOBAL_SEED = 42

n_subjects = 12
t_dense = np.linspace(0, 24, 241)
t_obs = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 24], dtype=float)

pk_noise_cv = 0.01
pd_noise_cv = 0.01

# PK典型参数（单次口服Bateman）
tv_pk = dict(D=100.0, F=1.0, V=10.0, ka=1.0, ke=0.2)
omega_pk = dict(V=0.00, ka=0.00, ke=0.00)

# ----------------------------
# 1) 主流PD模型注册表
# ----------------------------
MODEL_REGISTRY = {
    "DIRECT_LINEAR": {
        "family": "direct",
        "effect_form": "linear",
        "tv_pd": dict(E0=50.0, S=1.5, Emax=0.0, EC50=4.0, gamma=1.0, ke0=0.0, Kin=0.0, Kout=0.0, Imax=0.0, Smax=0.0),
        "omega_pd": dict(E0=0.15, S=0.15, Emax=0.00, EC50=0.00, gamma=0.00, ke0=0.00, Kin=0.00, Kout=0.00, Imax=0.00, Smax=0.00),
    },
    "DIRECT_EMAX": {
        "family": "direct",
        "effect_form": "emax",
        "tv_pd": dict(E0=50.0, S=0.0, Emax=30.0, EC50=4.0, gamma=1.0, ke0=0.0, Kin=0.0, Kout=0.0, Imax=0.0, Smax=0.0),
        "omega_pd": dict(E0=0.15, S=0.00, Emax=0.20, EC50=0.15, gamma=0.00, ke0=0.00, Kin=0.00, Kout=0.00, Imax=0.00, Smax=0.00),
    },
    "DIRECT_SIGEMAX": {
        "family": "direct",
        "effect_form": "sigemax",
        "tv_pd": dict(E0=50.0, S=0.0, Emax=30.0, EC50=4.0, gamma=3.0, ke0=0.0, Kin=0.0, Kout=0.0, Imax=0.0, Smax=0.0),
        "omega_pd": dict(E0=0.15, S=0.00, Emax=0.20, EC50=0.15, gamma=0.20, ke0=0.00, Kin=0.00, Kout=0.00, Imax=0.00, Smax=0.00),
    },
    "BIOPHASE_EMAX": {
        "family": "biophase",
        "effect_form": "emax",
        "tv_pd": dict(E0=50.0, S=0.0, Emax=30.0, EC50=4.0, gamma=1.0, ke0=0.8, Kin=0.0, Kout=0.0, Imax=0.0, Smax=0.0),
        "omega_pd": dict(E0=0.15, S=0.00, Emax=0.20, EC50=0.15, gamma=0.00, ke0=0.20, Kin=0.00, Kout=0.00, Imax=0.00, Smax=0.00),
    },
    "BIOPHASE_SIGEMAX": {
        "family": "biophase",
        "effect_form": "sigemax",
        "tv_pd": dict(E0=50.0, S=0.0, Emax=30.0, EC50=4.0, gamma=3.0, ke0=0.8, Kin=0.0, Kout=0.0, Imax=0.0, Smax=0.0),
        "omega_pd": dict(E0=0.15, S=0.00, Emax=0.20, EC50=0.15, gamma=0.20, ke0=0.20, Kin=0.00, Kout=0.00, Imax=0.00, Smax=0.00),
    },
    "IDR_BASE": {
        "family": "idr",
        "idr_mode": "base",
        "mod_target": "none",
        "mod_type": "none",
        "tv_pd": dict(Kin=25.0, Kout=0.5, Imax=0.0, Smax=0.0, EC50=4.0, gamma=1.0, E0=0.0, S=0.0, Emax=0.0, ke0=0.0),
        "omega_pd": dict(Kin=0.25, Kout=0.20, Imax=0.00, Smax=0.00, EC50=0.00, gamma=0.00, E0=0.00, S=0.00, Emax=0.00, ke0=0.00),
    },
    "IDR_INHIB_KIN_SIG": {
        "family": "idr",
        "idr_mode": "modulated",
        "mod_target": "Kin",
        "mod_type": "inhib_sigmoid",
        "tv_pd": dict(Kin=25.0, Kout=0.5, Imax=1.0, Smax=0.0, EC50=4.0, gamma=3.0, E0=0.0, S=0.0, Emax=0.0, ke0=0.0),
        "omega_pd": dict(Kin=0.25, Kout=0.20, Imax=0.00, Smax=0.00, EC50=0.00, gamma=0.20, E0=0.00, S=0.00, Emax=0.00, ke0=0.00),
    },
    "IDR_STIM_KIN_SIG": {
        "family": "idr",
        "idr_mode": "modulated",
        "mod_target": "Kin",
        "mod_type": "stim_sigmoid",
        "tv_pd": dict(Kin=25.0, Kout=0.5, Imax=0.0, Smax=1.0, EC50=4.0, gamma=2.5, E0=0.0, S=0.0, Emax=0.0, ke0=0.0),
        "omega_pd": dict(Kin=0.25, Kout=0.20, Imax=0.00, Smax=0.15, EC50=0.00, gamma=0.20, E0=0.00, S=0.00, Emax=0.00, ke0=0.00),
    },
    "IDR_INHIB_KOUT_SIG": {
        "family": "idr",
        "idr_mode": "modulated",
        "mod_target": "Kout",
        "mod_type": "stim_sigmoid",
        "tv_pd": dict(Kin=25.0, Kout=0.5, Imax=1.0, Smax=0.0, EC50=4.0, gamma=2.5, E0=0.0, S=0.0, Emax=0.0, ke0=0.0),
        "omega_pd": dict(Kin=0.25, Kout=0.20, Imax=0.00, Smax=0.00, EC50=0.00, gamma=0.20, E0=0.00, S=0.00, Emax=0.00, ke0=0.00),
    },
    "IDR_STIM_KOUT_SIG": {
        "family": "idr",
        "idr_mode": "modulated",
        "mod_target": "Kout",
        "mod_type": "inhib_sigmoid",
        "tv_pd": dict(Kin=25.0, Kout=0.5, Imax=0.0, Smax=0.8, EC50=4.0, gamma=2.5, E0=0.0, S=0.0, Emax=0.0, ke0=0.0),
        "omega_pd": dict(Kin=0.25, Kout=0.20, Imax=0.00, Smax=0.15, EC50=0.00, gamma=0.20, E0=0.00, S=0.00, Emax=0.00, ke0=0.00),
    },
    # ----------------------------
    # 2) 扩展PD模型（来自 PD模型/*.md）
    # ----------------------------
    "TMDD_BASE": {
        "family": "tmdd",
        "effect_form": "cpR_emax",
        "tv_pd": dict(
            Kon=0.05, Koff=0.01, keDR=0.02,
            Kin_R=10.0, Kout_R=0.2,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            Kon=0.10, Koff=0.10, keDR=0.10,
            Kin_R=0.20, Kout_R=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "TRANSDUCTION_DELAY": {
        "family": "transduction",
        "n_steps": 3,
        "tv_pd": dict(
            kt1=0.6,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            kt1=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "FEEDBACK_REGULATION": {
        "family": "feedback",
        "n_steps": 3,
        "tv_pd": dict(
            kt2=0.4,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
            Emax2=0.6, EC50_2=1.5, gamma2=2.0,
        ),
        "omega_pd": dict(
            kt2=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
            Emax2=0.20, EC50_2=0.20, gamma2=0.20,
        ),
    },
    "CIRCADIAN_REGULATION": {
        "family": "circadian",
        "tv_pd": dict(
            Kin_Rm=10.0, Kin_Rb=2.0, phi1=0.0,
            Kin_PD1m=15.0, Kin_PD1b=3.0, phi2=0.0,
            Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            Kin_Rm=0.20, Kin_Rb=0.20, phi1=0.00,
            Kin_PD1m=0.20, Kin_PD1b=0.20, phi2=0.00,
            Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "TGI_BASIC": {
        "family": "tgi",
        "tv_pd": dict(
            k_grow=0.15,
            Emax_kill=1.0, EC50_kill=2.0, gamma_kill=2.0,
            lambda_kill=0.02,
        ),
        "omega_pd": dict(
            k_grow=0.20,
            Emax_kill=0.20, EC50_kill=0.20, gamma_kill=0.20,
            lambda_kill=0.20,
        ),
    },
    "DISEASE_PROGRESSION": {
        "family": "disease",
        "tv_pd": dict(
            k_prog=0.2, k_rem=0.05,
            Emax_dis=1.0, EC50_dis=2.0, gamma_dis=2.0,
            Kout_PD1=0.3, Kout_PD2=0.2,
        ),
        "omega_pd": dict(
            k_prog=0.20, k_rem=0.20,
            Emax_dis=0.20, EC50_dis=0.20, gamma_dis=0.20,
            Kout_PD1=0.20, Kout_PD2=0.20,
        ),
    },
    "TOLERANCE_ADAPTATION": {
        "family": "tolerance",
        "tv_pd": dict(
            k_in_tol=0.2, k_out_tol=0.05,
            Smax_tol=1.0, SC50_tol=2.0, gamma_tol=2.0,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            k_in_tol=0.20, k_out_tol=0.20,
            Smax_tol=0.20, SC50_tol=0.20, gamma_tol=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "DRUG_INTERACTION": {
        "family": "interaction",
        "tv_pd": dict(
            ka_int=1.0, Vint=10.0, ke_int=0.2, KI=2.0,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            ka_int=0.20, Vint=0.20, ke_int=0.20, KI=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "PRECURSOR_POOL": {
        "family": "precursor",
        "tv_pd": dict(
            Kin_P=10.0, ktr=0.3,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            Kin_P=0.20, ktr=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
    "ANTIBODY_PKPD": {
        "family": "antibody",
        "tv_pd": dict(
            CLp=0.2, Q=0.3, V2=10.0, Vt=15.0,
            Kon=0.05, Koff=0.01, keDR=0.02,
            Kin_R=10.0, Kout_R=0.2,
            Kin_PD1m=15.0, Kout_PD1=0.3, Kout_PD2=0.2,
            Emax1=1.0, EC50=2.0, gamma=2.0,
        ),
        "omega_pd": dict(
            CLp=0.20, Q=0.20, V2=0.20, Vt=0.20,
            Kon=0.10, Koff=0.10, keDR=0.10,
            Kin_R=0.20, Kout_R=0.20,
            Kin_PD1m=0.20, Kout_PD1=0.20, Kout_PD2=0.20,
            Emax1=0.20, EC50=0.20, gamma=0.20,
        ),
    },
}

ACTIVE_MODEL = "IDR_INHIB_KIN_SIG"
