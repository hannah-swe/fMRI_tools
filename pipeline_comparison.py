from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

# Config / Paths
base_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/derivatives/halfpipe")

pipeline_patterns = {
    "no_scrub": "setting-imageNoscrub",
    "scrub": "setting-imageScrubSetting",
}
palette = {"no_scrub": "darkred", "scrub": "navy"}

participants_path = Path("/data_wgs04/ag-sensomotorik/PPPD/data/part2_pre/participants.tsv")
reportvals_path = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/reports/reportvals.txt")
subs_to_drop = ["sub-118", "sub-124", "sub-126", "sub-164"]

# DMN mask (same space as HALFpipe statmaps)
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
    if variant == "no_scrub":
        return func_task / f"{sub}_task-rest_run-01_feature-seedPrecNoscrub_seed-Precuneus_stat-effect_statmap.nii.gz"
    else:
        return func_task / f"{sub}_task-rest_run-01_feature-seedPrecScrub_seed-Precuneus_stat-effect_statmap.nii.gz"



# Load participants.tsv (group labels)
participants_df = pd.read_csv(participants_path, sep="\t")
participants_df["participant_id"] = "sub-" + participants_df["participant_id"].astype(str)

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
    for variant in ["no_scrub", "scrub"]:
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
qc_df = qc_df[~qc_df["sub"].isin(subs_to_drop)]


# Optional quick sanity checks
# 1) any subjects without group?
print(qc_df["group"].isna().sum(), "rows without group label")

# 2) any missing halfpipe fd metrics?
print(qc_df["fd_mean_halfpipe"].isna().sum(), "rows without fd_mean_halfpipe")

# 3) any missing fc values?
print(qc_df["fc_dmn_mean"].isna().sum(), "rows without fc_dmn_mean")


# QC Plots
# 1. subject-wise FD over scan
def plot_fd_over_tr(confounds_tsv, sub, fd_threshold=None):
    conf = pd.read_csv(confounds_tsv, sep="\t")
    fd = conf["framewise_displacement"].fillna(0)

    plt.figure(figsize=(4, 4))
    plt.plot(fd.values, linewidth=1)
    plt.xlabel("TR")
    plt.ylabel("Framewise Displacement")
    plt.title(f"{sub} run-01")
    sns.despine()

    if fd_threshold is not None:
        plt.axhline(fd_threshold, color="red", linestyle="--", linewidth=1)

    ymax = max(10, fd.max())
    plt.ylim(0, ymax)

    plt.tight_layout()
    plt.show()

for sub_dir in sorted(base_dir.glob("sub-*")):
    sub = sub_dir.name
    func_dir = sub_dir / "func"

    confounds_tsv = func_dir / f"{sub}_task-rest_run-01_setting-imageNoscrub_desc-confounds_regressors.tsv"
    if not confounds_tsv.exists():
        continue

    plot_fd_over_tr(
        confounds_tsv=confounds_tsv,
        sub=sub,
        fd_threshold=5
    )

# 2. percentage of scrubbed volumes on mean FD (only scrub pipeline)
df_no = qc_df.loc[
    qc_df["pipeline"] == "no_scrub",
    ["sub", "mean_fd"]
].rename(columns={"mean_fd": "mean_fd_no_scrub"})

df_scrub = qc_df.loc[
    qc_df["pipeline"] == "scrub",
    ["sub", "pct_scrub"]
].rename(columns={"pct_scrub": "pct_scrub_scrub"})

plot_df = df_no.merge(df_scrub, on="sub", how="inner")

sns.regplot(
    data=plot_df,
    x="mean_fd_no_scrub",
    y="pct_scrub_scrub",
    color="black",
)
plt.xlabel("Mean FD (no_scrub)")
plt.ylabel("% scrubbed volumes (scrub)")
sns.despine()
plt.tight_layout()
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/scrubbed_volumes_on_mean_fd.png")
plt.show()

# 3. QC-FC plot: correlation of subject-wise motion and pipeline-wise functional connectivity in dmn
# quick stats
# pearson's r
for pipeline in ["no_scrub", "scrub"]:
    dfp = qc_df.query("pipeline == @pipeline").dropna(
        subset=["fc_dmn_mean", "fd_mean_halfpipe"]
    )
    r, p = pearsonr(dfp["fd_mean_halfpipe"], dfp["fc_dmn_mean"])
    print(f"{pipeline}: r = {r:.3f}, p = {p:.3g}")
# linear regression
df = qc_df.dropna(subset=["fc_dmn_mean", "fd_mean_halfpipe"])
model = smf.ols(
    "fc_dmn_mean ~ fd_mean_halfpipe * pipeline + group",
    data=df
).fit()
print(model.summary())

plt.figure(figsize=(10, 8))
g = sns.lmplot(
    data=qc_df,
    x="fd_mean_halfpipe",
    y="fc_dmn_mean",
    col="pipeline",
    hue="pipeline",
    ci=95,
    legend=False,
    palette=palette
)
g.set_axis_labels("Framewise Displacement", "DMN Connectivity")
plt.tight_layout()
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/fc_dmn_on_mean_fd.png")
plt.show()

sns.boxplot(
    data=qc_df,
    x="pipeline",
    y="fc_dmn_mean",
    palette=palette,
    hue="pipeline"
)
sns.lineplot(
    data=qc_df,
    x="pipeline",
    y="fc_dmn_mean",
    units="sub",
    estimator=None,
    legend=False,
    color="black",
    linewidth=1.2,
    alpha=0.4
)
sns.despine()
plt.tight_layout()
plt.xlabel("")
plt.ylabel("DMN Connectivity")
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/mean_fc_dmn_on_pipelines.png")
plt.show()

# 4. Mean FD in different pipelines
sns.boxplot(
    data=qc_df,
    x="pipeline",
    y="mean_fd",
    palette=palette,
    hue="pipeline"
)
sns.lineplot(
    data=qc_df,
    x="pipeline",
    y="mean_fd",
    units="sub",
    estimator=None,
    legend=False,
    color="black",
    linewidth=1.2,
    alpha=0.4
)
sns.despine()
plt.tight_layout()
plt.xlabel("")
plt.ylabel("Mean Framewise Displacement")
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/mean_fd_on_pipelines.png")
plt.show()

# 4. Lineplot
sns.lineplot(
    data=qc_df,
    x="pipeline",
    y="remaining_tr",
    hue="sub",
    legend=False
)
sns.despine()
plt.ylabel("Remaining TRs")
plt.tight_layout()
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/scrubbed_volumes_on_pipelines.png")
plt.show()

# 5. Halfpipe mean fd and other mean fd
sns.lmplot(
    data=qc_df[qc_df.pipeline == "no_scrub"],
    x="mean_fd",
    y="fd_mean_halfpipe",
    palette=palette,
    hue="pipeline",
    legend=False,
    ci=95,
)
plt.xlabel("Mean FD confounds")
plt.ylabel("Mean FD reports")
sns.despine()
plt.tight_layout()
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/different_mean_fds.png")
plt.show()

# 6. FD difference between no scrubbing and scrubbing
diff_df = (
    qc_df
    .pivot(index="sub", columns="pipeline", values="mean_fd")
    .dropna(subset=["scrub", "no_scrub"])
)

diff_df["mean_fd_diff"] = diff_df["scrub"] - diff_df["no_scrub"]
diff_df = diff_df.reset_index()

plt.figure(figsize=(4, 6))
sns.boxplot(
    data=diff_df,
    y="mean_fd_diff",
    color="lightgray",
    width=0.4,
    fliersize=0
)
sns.stripplot(
    data=diff_df,
    y="mean_fd_diff",
    color="black",
    size=5,
    jitter=True,
    alpha=0.8
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("FD mean (scrub − no_scrub)")
sns.despine()
plt.tight_layout()
plt.savefig("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots/mean_fd_difference.png")
plt.show()
