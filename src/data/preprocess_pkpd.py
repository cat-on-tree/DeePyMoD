import numpy as np
import pandas as pd
import torch


def to_population_mean(pop_data):
    """
    Supports:
    - ndarray with columns [sid, time, C_obs, R_obs, ...]
    - DataFrame with at least sid,time,C_obs,R_obs and optional Ct_obs,C_int_obs

    return agg DataFrame columns:
    [time, C_mean, R_mean, (optional) Ct_mean, (optional) C_int_mean]
    """
    if isinstance(pop_data, pd.DataFrame):
        df = pop_data.copy()
    else:
        arr = np.asarray(pop_data)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("pop_data must be DataFrame or 2D array with at least 4 columns.")
        cols = ["sid", "time", "C_obs", "R_obs"] + [f"extra_{i}" for i in range(arr.shape[1] - 4)]
        df = pd.DataFrame(arr, columns=cols)

    required = {"sid", "time", "C_obs", "R_obs"}
    if not required.issubset(df.columns):
        raise ValueError(f"pop_data must include columns {required}. got={list(df.columns)}")

    agg_dict = {"C_mean": ("C_obs", "mean"), "R_mean": ("R_obs", "mean")}
    if "Ct_obs" in df.columns:
        agg_dict["Ct_mean"] = ("Ct_obs", "mean")
    if "C_int_obs" in df.columns:
        agg_dict["C_int_mean"] = ("C_int_obs", "mean")

    agg = df.groupby("time", as_index=False).agg(**agg_dict).sort_values("time").reset_index(drop=True)
    return agg


def build_tensors_from_agg(agg_df):
    """
    return:
      X_all: [N,4] = [t_norm, C, C_int, Ct]
      Y_all: [N,1] = normalized R
      raw: dict of raw tensors
    """
    Xg = torch.tensor(agg_df["time"].values.reshape(-1, 1), dtype=torch.float32)
    Cg = torch.tensor(agg_df["C_mean"].values.reshape(-1, 1), dtype=torch.float32)
    Cint_arr = agg_df["C_int_mean"].values.reshape(-1, 1) if "C_int_mean" in agg_df.columns else np.zeros_like(agg_df["C_mean"].values.reshape(-1, 1))
    Ct_arr = agg_df["Ct_mean"].values.reshape(-1, 1) if "Ct_mean" in agg_df.columns else np.zeros_like(agg_df["C_mean"].values.reshape(-1, 1))
    Cintg = torch.tensor(Cint_arr, dtype=torch.float32)
    Ctg = torch.tensor(Ct_arr, dtype=torch.float32)
    Yg = torch.tensor(agg_df["R_mean"].values.reshape(-1, 1), dtype=torch.float32)

    Xn = (Xg - Xg.mean()) / (Xg.std() + 1e-8)
    Yn = (Yg - Yg.mean()) / (Yg.std() + 1e-8)

    X_all = torch.cat([Xn, Cg, Cintg, Ctg], dim=1).float()
    Y_all = Yn.float()

    return X_all, Y_all, {"time_raw": Xg, "C_raw": Cg, "C_int_raw": Cintg, "Ct_raw": Ctg, "R_raw": Yg}


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
