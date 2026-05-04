import os
import pandas as pd
import matlab.engine

from src.pipeline.export_for_nlme import export_nlme_inputs


def run_simbiology_validation(pop_data, top_results, project_root=".", out_dir=None, use_population_mean=False):
    """
    pop_data: [sid, time, C_obs, R_obs]
    top_results: candidates list (from topk json payload["candidates"])
    project_root: repo root
    out_dir: 指定输出目录（若为None则默认 artifacts/nlme）
    use_population_mean: True → 用群体均值数据拟合（与 Step 4 一致的优化问题）
    """
    project_root = os.path.abspath(project_root)

    if out_dir is None:
        out_dir = os.path.join(project_root, "artifacts", "nlme")
    else:
        out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    data_csv, cand_json = export_nlme_inputs(
        pop_data, top_results, out_dir=out_dir, use_population_mean=use_population_mean
    )

    eng = matlab.engine.start_matlab()
    matlab_root = os.path.join(project_root, "matlab")
    eng.addpath(matlab_root, nargout=0)

    out_csv = os.path.join(out_dir, "simbiology_results.csv")
    eng.fit_topk_simbiology(data_csv, cand_json, out_csv, nargout=0)

    return pd.read_csv(out_csv)


def run_simbiology_diagnostics(data_csv, simbio_csv, topk_json, fig_dir, project_root=".", bin_edges=None, skip_bootstrap=False):
    """
    在 Step 5 NLME 结果上做 VPC + Bootstrap 诊断图。

    Parameters
    ----------
    data_csv   : pkpd_long.csv
    simbio_csv : simbiology_results.csv（含 theta1..thetaK 列）
    topk_json  : topk_candidates.json（含 ec50_hat, gamma_hat）
    fig_dir    : 图表输出目录
    project_root: repo root
    bin_edges  : 时间分箱边界，默认 [0, 1, 4, 8, 24]
    skip_bootstrap: True → 跳过 Bootstrap（节省时间）
    """
    project_root = os.path.abspath(project_root)
    os.makedirs(fig_dir, exist_ok=True)

    if bin_edges is None:
        bin_edges = [0.0, 1.0, 4.0, 8.0, 24.0]

    eng = matlab.engine.start_matlab()
    matlab_root = os.path.join(project_root, "matlab")
    eng.addpath(matlab_root, nargout=0)

    eng.diagnostics_plots(data_csv, simbio_csv, topk_json, fig_dir, bin_edges, skip_bootstrap, nargout=0)