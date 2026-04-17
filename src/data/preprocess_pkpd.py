import numpy as np
import pandas as pd
import torch


def to_population_mean(pop_data):
    """
    pop_data columns: [sid, time, C_obs, R_obs]
    return agg DataFrame columns: [time, C_mean, R_mean]
    """
    df = pd.DataFrame(pop_data, columns=["sid", "time", "C_obs", "R_obs"])
    agg = (
        df.groupby("time", as_index=False)
          .agg(C_mean=("C_obs", "mean"), R_mean=("R_obs", "mean"))
          .sort_values("time")
          .reset_index(drop=True)
    )
    return agg


def build_tensors_from_agg(agg_df):
    """
    return:
      X_all: [N,2] = [t_norm, C]
      Y_all: [N,1] = normalized R
      raw: dict of raw tensors
    """
    Xg = torch.tensor(agg_df["time"].values.reshape(-1, 1), dtype=torch.float32)
    Cg = torch.tensor(agg_df["C_mean"].values.reshape(-1, 1), dtype=torch.float32)
    Yg = torch.tensor(agg_df["R_mean"].values.reshape(-1, 1), dtype=torch.float32)

    Xn = (Xg - Xg.mean()) / (Xg.std() + 1e-8)
    Yn = (Yg - Yg.mean()) / (Yg.std() + 1e-8)

    X_all = torch.cat([Xn, Cg], dim=1).float()
    Y_all = Yn.float()

    return X_all, Y_all, {"time_raw": Xg, "C_raw": Cg, "R_raw": Yg}


def split_train_val(X_all, Y_all, train_frac=0.7, keep_time_order=True, seed=42):
    """
    时间序列默认保持顺序切分。
    """
    n = X_all.shape[0]
    n_train = int(np.floor(train_frac * n))

    if keep_time_order:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

    tr_idx = idx[:n_train]
    va_idx = idx[n_train:]

    X_train, Y_train = X_all[tr_idx], Y_all[tr_idx]
    X_val, Y_val = X_all[va_idx], Y_all[va_idx]
    return X_train, Y_train, X_val, Y_val