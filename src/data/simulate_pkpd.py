import numpy as np
import torch

from src.configs.pkpd_registry import MODEL_REGISTRY, tv_pk, omega_pk, t_obs
from src.data.subject_simulator import sample_individual_params, simulate_subject


def generate_population_data(model_name, seed=42, n_subjects=12):
    cfg = MODEL_REGISTRY[model_name]
    tv_pd = cfg["tv_pd"]
    omega_pd = cfg["omega_pd"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_rows = []
    subject_params = []

    for sid in range(n_subjects):
        p = sample_individual_params(tv_pk, omega_pk, tv_pd, omega_pd)
        p["model_name"] = model_name
        p["family"] = cfg["family"]
        subject_params.append(p)

        C_obs, R_obs = simulate_subject(cfg, p)
        for j, tt in enumerate(t_obs):
            all_rows.append([sid, tt, C_obs[j], R_obs[j]])

    pop_data = np.array(all_rows, dtype=float)
    return pop_data, subject_params, cfg