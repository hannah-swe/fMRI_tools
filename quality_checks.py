import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# Config
base_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/workdir_precuneus/derivatives/halfpipe")
PIPELINE_PATTERNS = {
    "noscrub": "setting-image",
    "scrub": "setting-imageScrubSetting",
}

# Function to get the mean of a nifti image
def mean_in_mask(img_path, mask_path=None):
    img = nib.load(img_path).get_fdata()
    if mask_path is None:
        return float(np.nanmean(img))
    mask = nib.load(mask_path).get_fdata() > 0
    return float(np.nanmean(img[mask]))

# Function to get QC-metrics per subject and per run
def extract_qc_metrics(sub, run, pipeline, confounds_tsv, tsnr_map, brain_mask=None):
    conf = pd.read_csv(confounds_tsv, sep="\t")

    fd = conf["framewise_displacement"].fillna(0)

    n_tr = len(conf)

    if pipeline == "scrub":
        n_scrub = len([c for c in conf.columns if c.startswith("motion_outlier")])
    else:
        n_scrub = 0

    mean_tsnr = mean_in_mask(tsnr_map, brain_mask)

    return {
        "sub": sub,
        "run": run,
        "pipeline": pipeline,
        "n_tr": int(n_tr),
        "mean_fd": float(fd.mean()),
        "median_fd": float(fd.median()),
        "max_fd": float(fd.max()),
        "n_scrub": int(n_scrub),
        "pct_scrub": float(n_scrub / n_tr * 100),
        "remaining_tr": int(n_tr - n_scrub),
        "mean_tsnr": float(mean_tsnr),
    }

rows = []

# Subject loop to get subjects data for all pipelines
for sub_dir in sorted(base_dir.glob("sub-*")):
    sub = sub_dir.name
    func_dir = sub_dir / "func"

    # One tsnr-map per run as it is independent of motion scrubbing in HALFpipe
    tsnr_map = func_dir / f"{sub}_task-rest_run-01_stat-tsnr_boldmap.nii.gz"
    if not tsnr_map.exists():
        print(f"[SKIP] No tSNR map for {sub}")
        continue

    # Loop over all defined pipelines
    for pipeline, pat in PIPELINE_PATTERNS.items():
        confounds_tsv = func_dir / f"{sub}_task-rest_run-01_{pat}_desc-confounds_regressors.tsv"
        brain_mask = func_dir / f"{sub}_task-rest_run-01_{pat}_desc-brain_mask.nii.gz"

        if not confounds_tsv.exists():
            print(f"[SKIP] No confounds for {sub} ({pipeline})")
            continue

        rows.append(
            extract_qc_metrics(
                sub=sub,
                run="run-01",
                pipeline=pipeline,
                confounds_tsv=confounds_tsv,
                tsnr_map=tsnr_map,
                brain_mask=brain_mask if brain_mask.exists() else None,
            )
        )

# Get complete dataframe
qc_df = pd.DataFrame(rows).sort_values(["sub", "pipeline"])

