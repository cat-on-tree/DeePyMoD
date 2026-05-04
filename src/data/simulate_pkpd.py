import numpy as np
import torch

from src.configs.pkpd_registry import MODEL_REGISTRY, tv_pk, omega_pk, t_obs
from src.data.subject_simulator import sample_individual_params, simulate_subject


def generate_population_data(
    model_name,
    seed=42,
    n_subjects=12,
    extra_pk_iiv_sigma=0.0,
    return_pk_scale=False,
):
    """
    extra_pk_iiv_sigma:
      0.0 表示不额外加
      >0 表示对每个受试者 C_obs 乘以 exp(N(0, sigma))
    """
    cfg = MODEL_REGISTRY[model_name]
    tv_pd = cfg["tv_pd"]
    omega_pd = cfg["omega_pd"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed + 10086)

    all_rows = []
    subject_params = []
    pk_scale_by_sid = {}

    for sid in range(n_subjects):
        p = sample_individual_params(tv_pk, omega_pk, tv_pd, omega_pd)
        p["model_name"] = model_name
        p["family"] = cfg["family"]
        subject_params.append(p)

        C_obs, R_obs = simulate_subject(cfg, p)

        if extra_pk_iiv_sigma > 0:
            scale = float(np.exp(rng.normal(0.0, extra_pk_iiv_sigma)))
            C_obs = np.clip(C_obs * scale, 0.0, None)
        else:
            scale = 1.0
        pk_scale_by_sid[int(sid)] = scale

        for j, tt in enumerate(t_obs):
            all_rows.append([sid, tt, C_obs[j], R_obs[j]])

    pop_data = np.array(all_rows, dtype=float)

    if return_pk_scale:
        return pop_data, subject_params, cfg, pk_scale_by_sid
    return pop_data, subject_params, cfg