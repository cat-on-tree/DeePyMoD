GUIDE_MODE=classifier \
GATE_THRESHOLD=0.30 \
GATE_PENALTY=200 \
OBSERVABLE_FP_PENALTY=50 \
N_SUBJECTS=4 \
WORKERS=4 \
PD_MODELS=DIRECT_SIGEMAX,BIOPHASE_EMAX,TRANSDUCTION_DELAY,TOLERANCE_ADAPTATION,DISEASE_PROGRESSION,CIRCADIAN_REGULATION,FEEDBACK_REGULATION,PRECURSOR_POOL \
bash run_pkpd_pd_refinement.sh repair-benchmark artifacts/pkpd_mechanism_repair_dev_v4 3 42

bash run_pkpd_pd_refinement.sh analyze-repair artifacts/pkpd_mechanism_repair_dev_v4