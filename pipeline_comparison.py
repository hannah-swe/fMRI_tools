from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config / Paths
base_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/workdir_precuneus/derivatives/halfpipe")

pipeline_patterns = {
    "noscrub": "setting-image",
    "scrub": "setting-imageScrubSetting",
}

participants_path = Path("/data_wgs04/ag-sensomotorik/PPPD/BIDS_run1/participants.tsv")
reportvals_path = Path("/data_wgs04/ag-sensomotorik/PPPD/workdir_precuneus/reports/reportvals.txt")

# DMN mask must be in same grid as HALFpipe statmaps (you already resampled it)
dmn_mask = Path("/data_wgs04/ag-sensomotorik/PPPD/masks/dmn_mask_resampled.nii.gz")


# Helper functions
def mean_in_mask_nii(img_path, mask_path=None):
    """Mean of NIfTI image; optionally restricted to binary mask."""
    img = nib.load(str(img_path)).get_fdata()
    if mask_path is None:
        return float(np.nanmean(img))
    mask = nib.load(str(mask_path)).get_fdata() > 0
    return float(np.nanmean(img[mask]))


def extract_qc_metrics(sub, run, pipeline, confounds_tsv, tsnr_map, brain_mask=None):
    """Extract subject/run/pipeline QC metrics from HALFpipe outputs."""
    conf = pd.read_csv(confounds_tsv, sep="\t")

    fd = conf["framewise_displacement"].fillna(0)
    n_tr = len(conf)

    # In HALFpipe scrub is implemented via one regressor per censored volume (motion_outlierXX)
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


def statmap_path(sub_dir, sub, variant):
    """Paths to seed-based outputs."""
    func_task = sub_dir / "func" / "task-rest"
    if variant == "noscrub":
        return func_task / f"{sub}_task-rest_run-01_feature-seedbasedPrec_seed-Precuneus_stat-effect_statmap.nii.gz"
    else:
        return func_task / f"{sub}_task-rest_run-01_feature-seedbasedPrecScrub_seed-Precuneus_stat-effect_statmap.nii.gz"



# Load participants.tsv (group labels)
participants_df = pd.read_csv(participants_path, sep="\t")

# Load HALFpipe reportvals.txt (dashboard-consistent QC)
reportvals_df = pd.read_fwf(reportvals_path)
reportvals_df.columns = [c.strip() for c in reportvals_df.columns]

reportvals_qc = reportvals_df[["sub", "fd_mean", "fd_max", "fd_perc"]].copy()
reportvals_qc["sub"] = "sub-" + reportvals_qc["sub"].astype(str).str.zfill(2)

reportvals_qc = reportvals_qc.rename(
    columns={
        "fd_mean": "fd_mean_halfpipe",
        "fd_max": "fd_max_halfpipe",
        "fd_perc": "fd_perc_halfpipe",
    }
)

num_cols = ["fd_mean_halfpipe", "fd_max_halfpipe", "fd_perc_halfpipe"]
reportvals_qc[num_cols] = reportvals_qc[num_cols].apply(pd.to_numeric, errors="coerce")


# Build QC table (confounds + tSNR)
qc_rows = []

for sub_dir in sorted(base_dir.glob("sub-*")):
    sub = sub_dir.name
    func_dir = sub_dir / "func"

    # One tSNR map per run (independent of scrubbing)
    tsnr_map = func_dir / f"{sub}_task-rest_run-01_stat-tsnr_boldmap.nii.gz"
    if not tsnr_map.exists():
        print(f"[SKIP] No tSNR map for {sub}")
        continue

    for pipeline, pat in pipeline_patterns.items():
        confounds_tsv = func_dir / f"{sub}_task-rest_run-01_{pat}_desc-confounds_regressors.tsv"
        brain_mask = func_dir / f"{sub}_task-rest_run-01_{pat}_desc-brain_mask.nii.gz"

        if not confounds_tsv.exists():
            print(f"[SKIP] No confounds for {sub} ({pipeline})")
            continue

        qc_rows.append(
            extract_qc_metrics(
                sub=sub,
                run="run-01",
                pipeline=pipeline,
                confounds_tsv=confounds_tsv,
                tsnr_map=tsnr_map,
                brain_mask=brain_mask if brain_mask.exists() else None,
            )
        )

qc_df = pd.DataFrame(qc_rows).sort_values(["sub", "pipeline"])

# Merge: group labels
qc_df = qc_df.merge(
    participants_df[["participant_id", "group"]],
    left_on="sub",
    right_on="participant_id",
    how="left",
).drop(columns=["participant_id"])

# Merge: HALFpipe dashboard QC metrics
qc_df = qc_df.merge(reportvals_qc, on="sub", how="left")


# Build FC table (DMN mean from seed-based statmaps) and merge into qc_df
fc_rows = []
if not dmn_mask.exists():
    raise FileNotFoundError(f"DMN mask not found: {dmn_mask}")

for sub_dir in sorted(base_dir.glob("sub-*")):
    sub = sub_dir.name
    for variant in ["noscrub", "scrub"]:
        p = statmap_path(sub_dir, sub, variant)
        if not p.exists():
            # It's fine to skip missing maps (e.g., subject failed for a feature)
            continue
        fc_rows.append(
            {
                "sub": sub,
                "pipeline": variant,
                "fc_dmn_mean": mean_in_mask_nii(p, dmn_mask),
            }
        )

fc_df = pd.DataFrame(fc_rows)

qc_df = qc_df.merge(fc_df, on=["sub", "pipeline"], how="left")


# Optional quick sanity checks
# 1) any subjects without group?
print(qc_df["group"].isna().sum(), "rows without group label")

# 2) any missing halfpipe fd metrics?
print(qc_df["fd_mean_halfpipe"].isna().sum(), "rows without fd_mean_halfpipe")

# 3) any missing fc values?
print(qc_df["fc_dmn_mean"].isna().sum(), "rows without fc_dmn_mean")
