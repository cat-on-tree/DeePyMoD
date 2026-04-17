import os
import pandas as pd
import matlab.engine

from src.pipeline.export_for_nlme import export_nlme_inputs


def run_nlme_validation(pop_data, top_results, project_root="."):
    out_dir = os.path.join(project_root, "artifacts", "nlme")
    data_csv, cand_json = export_nlme_inputs(pop_data, top_results, out_dir=out_dir)

    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(project_root, "matlab"), nargout=0)

    out_csv = os.path.join(out_dir, "nlme_results.csv")
    eng.fit_topk_nlme(data_csv, cand_json, out_csv, nargout=0)

    df = pd.read_csv(out_csv)
    return df