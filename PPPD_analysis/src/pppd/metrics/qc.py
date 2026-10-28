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


def compute_motion_outlier_metrics(
    confounds: pd.DataFrame,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5,
) -> dict[str, float | int]:

    outlier_cols = [
        col
        for col in confounds.columns
        if col.startswith("motion_outlier")
    ]

    if outlier_cols:
        outlier_mask = (
            confounds[outlier_cols]
            .fillna(0)
            .astype(bool)
            .any(axis=1)
        )
    else:
        outlier_mask = pd.Series(
            False,
            index=confounds.index,
        )

    fd = confounds.get("framewise_displacement")
    dvars = confounds.get("std_dvars")

    if fd is not None:
        fd_mask = fd > fd_threshold
    else:
        fd_mask = pd.Series(
            False,
            index=confounds.index,
        )

    if dvars is not None:
        dvars_mask = dvars > dvars_threshold
    else:
        dvars_mask = pd.Series(
            False,
            index=confounds.index,
        )

    n_volumes = len(confounds)
    n_outliers = int(outlier_mask.sum())

    return {
        "n_volumes": n_volumes,
        "n_motion_outliers": n_outliers,
        "motion_outlier_perc": (
            100 * n_outliers / n_volumes
            if n_volumes
            else float("nan")
        ),
        "n_fd_outliers": int(fd_mask.sum()),
        "n_dvars_outliers": int(dvars_mask.sum()),
        "n_fd_only_outliers": int(
            (fd_mask & ~dvars_mask).sum()
        ),
        "n_dvars_only_outliers": int(
            (~fd_mask & dvars_mask).sum()
        ),
        "n_fd_and_dvars_outliers": int(
            (fd_mask & dvars_mask).sum()
        ),
    }