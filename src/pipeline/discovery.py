import numpy as np
import torch
from deepymod import DeepMoD

from src.models.network import TimeOnlyNet, ModularTimeNet
from src.models.library import PDLibraryExpanded, ModularPDLibrary
from src.models.estimator import FixedMaskEstimator
from src.models.constraint import RidgeConstraint

from src.training.trainer import (
    build_train_loader,
    run_train_loop,
    get_coeff_and_mask,
)
from src.training.pruning import prune_mask_general
from src.training.ranking import rank_topk_validation_bic

from src.data.preprocess_pkpd import to_population_mean, build_tensors_from_agg, split_train_val
from src.configs.pd_module_registry import MODULE_COMBINATIONS


def run_single_discovery(
    pop_data,
    active_model: str,
    config: dict,
    device: str = None,
    module_combo: str = None,
):
    """
    输入:
      pop_data: columns [sid, time, C_obs, R_obs]
      active_model: e.g. "IDR_INHIB_KIN_SIG"
      config: 超参数字典（可用 src/configs/defaults.py 的 DEFAULTS）
      module_combo: e.g. "idr+delay" (optional). If None, use legacy single-state library.
    返回:
      dict(results)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) preprocess
    agg = to_population_mean(pop_data)
    X_all, Y_all, raw = build_tensors_from_agg(agg)
    X_train, Y_train, X_val, Y_val = split_train_val(
        X_all, Y_all,
        train_frac=config["train_frac"],
        keep_time_order=True,
        seed=config.get("seed", 42),
    )

    train_loader = build_train_loader(X_train, Y_train, device=device)

    # 2) model init
    if module_combo:
        network = ModularTimeNet(module_combo=module_combo).to(device)
        library = ModularPDLibrary(module_combo=module_combo).to(device)
        term_names = library.term_names()
    else:
        network = TimeOnlyNet().to(device)
        library = PDLibraryExpanded().to(device)
        term_names = PDLibraryExpanded.term_names()

    n_terms = len(term_names)
    init_mask = torch.ones(n_terms, dtype=torch.bool, device=device)
    estimator = FixedMaskEstimator(init_mask).to(device)
    constraint = RidgeConstraint(lam=config["ridge_lam"]).to(device)

    model = DeepMoD(network, library, estimator, constraint).to(device)

    opt_nn = torch.optim.Adam(model.func_approx.parameters(), lr=config["lr_nn"])
    opt_lib = torch.optim.Adam(model.library.parameters(), lr=config["lr_lib"])

    # 3) warmup
    print("\n[Warmup]")
    last = run_train_loop(
        model, train_loader, opt_nn, opt_lib,
        n_epochs=config["n_epochs_warmup"],
        lambda_reg=config["lambda_reg"],
        lambda_gamma_pen=config["lambda_gamma_pen"],
    )
    print(f"warmup done | loss={last['loss']:.6f} mse={last['mse']:.6f} reg={last['reg']:.6f}")

    # 4) iterative pruning
    print("\n[Iterative pruning]")
    stable_cnt = 0
    for rd in range(1, config["max_prune_rounds"] + 1):
        coeff, mask = get_coeff_and_mask(model)

        new_mask, changed = prune_mask_general(
            coeff=coeff,
            old_mask=mask,
            term_names=term_names,
            active_model=active_model,
            rel_thr_main=config["rel_thr_main"],
            rel_thr_interaction=config["rel_thr_interaction"],
            min_terms_keep=config["min_terms_keep"],
        )

        old_terms = np.array(term_names)[mask.detach().cpu().numpy().astype(bool)]
        new_terms = np.array(term_names)[new_mask.detach().cpu().numpy().astype(bool)]

        print(f"Round {rd}:")
        print("  active(old):", list(old_terms))
        print("  active(new):", list(new_terms))

        if not changed:
            stable_cnt += 1
            print(f"  -> unchanged ({stable_cnt}/{config['stable_rounds_required']})")
            if stable_cnt >= config["stable_rounds_required"]:
                print("  -> mask stable enough, stop.")
                break
        else:
            stable_cnt = 0
            estimator.set_mask(new_mask)
            constraint.sparsity_masks = [new_mask]
            last = run_train_loop(
                model, train_loader, opt_nn, opt_lib,
                n_epochs=config["n_epochs_prune"],
                lambda_reg=config["lambda_reg"],
                lambda_gamma_pen=config["lambda_gamma_pen"],
            )
            print(f"  retrain done | loss={last['loss']:.6f} mse={last['mse']:.6f} reg={last['reg']:.6f}")

    # 5) top-k ranking (validation BIC)
    print("\n[Top-K candidate generation with Validation-BIC]")
    best, top_results = rank_topk_validation_bic(
        model=model,
        estimator=estimator,
        constraint=constraint,
        network=network,
        library=library,
        opt_nn=opt_nn,
        opt_lib=opt_lib,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        train_loader=train_loader,
        term_names=term_names,
        topk=config["topk"],
        candidate_extra_terms_max=config["candidate_extra_terms_max"],
        candidate_refit_epochs=config["candidate_refit_epochs"],
        lambda_reg=config["lambda_reg"],
        lambda_gamma_pen=config["lambda_gamma_pen"],
    )

    print("\n=== Top-K Candidate Structures (Validation BIC) ===")
    for i, r in enumerate(top_results, 1):
        print(
            f"[Rank {i}] BIC_val={r['score']:.6f} | "
            f"mse_val={r['mse_val']:.6f} | mse_train={r['mse_train']:.6f} | k={r['k']}"
        )
        print("  terms:", r["terms"])

    print("\n=== Selected (Top-1) ===")
    print("ACTIVE_MODEL:", active_model)
    if module_combo:
        print("MODULE_COMBO:", module_combo)
    print("terms:", best["terms"])
    print(
        f"BIC_val={best['score']:.6f}, "
        f"mse_val={best['mse_val']:.6f}, mse_train={best['mse_train']:.6f}, k={best['k']}"
    )

    ec50_hat = float(torch.nn.functional.softplus(library.raw_ec50).item())
    gamma_hat = float(torch.nn.functional.softplus(library.raw_gamma).item())
    print(f"EC50_hat={ec50_hat:.6f}, gamma_hat={gamma_hat:.6f}")

    return {
        "active_model": active_model,
        "module_combo": module_combo,
        "best": best,
        "top_results": top_results,
        "ec50_hat": ec50_hat,
        "gamma_hat": gamma_hat,
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "agg": agg,
        "raw": raw,
    }


def run_module_discovery(
    pop_data,
    active_model: str,
    config: dict,
    device: str = None,
    module_combos=None,
):
    module_combos = module_combos or MODULE_COMBINATIONS
    results = {}
    for combo in module_combos:
        print("\n=============================")
        print(f"[Module combo] {combo}")
        print("=============================")
        results[combo] = run_single_discovery(
            pop_data=pop_data,
            active_model=active_model,
            config=config,
            device=device,
            module_combo=combo,
        )
    return results
