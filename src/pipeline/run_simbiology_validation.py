import os
import pandas as pd
import matlab.engine

from src.pipeline.export_for_nlme import export_nlme_inputs


def run_simbiology_validation(pop_data, top_results, project_root="."):
    project_root = os.path.abspath(project_root)
    out_dir = os.path.join(project_root, "artifacts", "nlme")
    data_csv, cand_json = export_nlme_inputs(pop_data, top_results, out_dir=out_dir)

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(os.path.join(project_root, "matlab")), nargout=0)

    out_csv = os.path.join(out_dir, "simbiology_results.csv")
    eng.fit_topk_simbiology(data_csv, cand_json, out_csv, nargout=0)

    return pd.read_csv(out_csv)