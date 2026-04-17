import numpy as np
import torch
import torch.nn.functional as F
from deepymod.data import Dataset, get_train_test_loader


def build_train_loader(X_train, Y_train, device="cpu"):
    def data_loader_func_train():
        return X_train, Y_train

    dataset = Dataset(
        data_loader_func_train,
        preprocess_kwargs=dict(noise_level=0.0, normalize_coords=False, normalize_data=False),
        device=device
    )
    train_loader, _ = get_train_test_loader(dataset, train_test_split=1.0)
    return train_loader


def train_epochs(
    model,
    train_loader,
    opt_nn,
    opt_lib,
    lambda_reg=0.2,
    lambda_gamma_pen=1e-4,
):
    last = {"loss": np.nan, "mse": np.nan, "reg": np.nan}
    for xb, yb in train_loader:
        xb = xb.to(next(model.parameters()).device).clone().detach().requires_grad_(True)
        yb = yb.to(next(model.parameters()).device)

        pred, dts, ths = model(xb)
        mse = torch.mean((pred - yb) ** 2)

        coeffs = model.constraint_coeffs(sparse=True, scaled=False)
        reg = torch.mean(torch.stack([
            torch.mean((dt - th @ cf) ** 2) for dt, th, cf in zip(dts, ths, coeffs)
        ]))

        gamma_now = F.softplus(model.library.raw_gamma)
        loss = mse + lambda_reg * reg + lambda_gamma_pen * gamma_now

        opt_nn.zero_grad()
        opt_lib.zero_grad()
        loss.backward()
        opt_nn.step()
        opt_lib.step()

        last = {"loss": float(loss.item()), "mse": float(mse.item()), "reg": float(reg.item())}
    return last


def run_train_loop(
    model, train_loader, opt_nn, opt_lib, n_epochs,
    lambda_reg=0.2, lambda_gamma_pen=1e-4
):
    last = None
    for _ in range(n_epochs):
        last = train_epochs(
            model, train_loader, opt_nn, opt_lib,
            lambda_reg=lambda_reg, lambda_gamma_pen=lambda_gamma_pen
        )
    return last


def eval_mse(model, X_eval, Y_eval):
    device = next(model.parameters()).device
    xb = X_eval.to(device).clone().detach().requires_grad_(True)
    yb = Y_eval.to(device)
    pred, _, _ = model(xb)
    mse = torch.mean((pred - yb) ** 2)
    return float(mse.item())


def get_coeff_and_mask(model):
    constraint = model.constraint
    mask = constraint.sparsity_masks[0].detach().clone()
    coeff = model.constraint_coeffs(sparse=True, scaled=False)[0].detach().flatten()
    return coeff, mask