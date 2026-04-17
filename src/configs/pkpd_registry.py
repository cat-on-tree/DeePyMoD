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
}

ACTIVE_MODEL = "IDR_INHIB_KIN_SIG"