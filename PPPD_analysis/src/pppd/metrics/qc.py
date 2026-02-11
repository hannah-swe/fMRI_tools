from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pppd.metrics.nifti import mean_in_mask_nii


def extract_qc_metrics(
    sub: str,
    run: str,
    pipeline: str,
    confounds_tsv: Path,
    tsnr_map: Path,
    brain_mask: Path | None = None,
) -> dict:
    """Extract subject/run/pipeline QC metrics from HALFpipe outputs."""
    conf = pd.read_csv(confounds_tsv, sep="\t")

    fd = conf["framewise_displacement"].fillna(0)
    n_tr = len(conf)

    # HALFpipe scrub: one regressor per censored volume (motion_outlierXX)
    if pipeline == "scrub":
        n_scrub = sum(c.startswith("motion_outlier") for c in conf.columns)
    else:
        n_scrub = 0

    mean_tsnr = mean_in_mask_nii(tsnr_map, brain_mask)

    return {
        "sub": sub,
        "run": run,
        "pipeline": pipeline,
        "n_tr": int(n_tr),
        "mean_fd": float(fd.mean()),
        "median_fd": float(fd.median()),
        "max_fd": float(fd.max()),
        "n_scrub": int(n_scrub),
        "pct_scrub": float(n_scrub / n_tr * 100) if n_tr else np.nan,
        "remaining_tr": int(n_tr - n_scrub),
        "mean_tsnr": float(mean_tsnr),
    }
