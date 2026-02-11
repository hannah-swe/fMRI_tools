from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from pppd.config.paths import fmriprep_dir
from pppd.config.pipelines import palette
from pppd.io.halfpipe import find_confounds_timeseries


# -----------------------------
# Config
# -----------------------------
FEATURE = "seedfc"  # "falff" oder "seedfc"
RUN = "run-01"
FD_THRESHOLD = 0.5

IN_TABLE = Path.cwd() / "outputs" / f"qc_{FEATURE}_table.csv"
OUT_DIR = Path.cwd() / "outputs" / "plots" / FEATURE
OUT_DIR.mkdir(parents=True, exist_ok=True)

# sns.set_context("talk")


# -----------------------------
# Plot 1: FD over TR (fMRIPrep)
# -----------------------------
def plot_fd_over_tr(confounds_tsv: Path, sub: str, fd_threshold: float | None = None) -> None:
    conf = pd.read_csv(confounds_tsv, sep="\t")
    fd = conf["framewise_displacement"].fillna(0)

    plt.figure(figsize=(4, 4))
    plt.plot(fd.values, linewidth=0.8)
    plt.xlabel("TR")
    plt.ylabel("Framewise Displacement")
    plt.title(f"{sub} {RUN}".strip())
    sns.despine()

    if fd_threshold is not None:
        plt.axhline(fd_threshold, color="red", linestyle="--", linewidth=1)

    ymax = max(1, float(fd.max()))
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.show()


def run_fd_over_tr_plots() -> None:
    # Nur anzeigen, nicht speichern
    for sub_dir in sorted(fmriprep_dir.glob("sub-*")):
        sub = sub_dir.name
        confounds_tsv = find_confounds_timeseries(sub_dir=sub_dir, sub=sub, run=RUN)
        if confounds_tsv is None:
            continue
        plot_fd_over_tr(confounds_tsv, sub=sub, fd_threshold=FD_THRESHOLD)




# -----------------------------
# Load table
# -----------------------------
if not IN_TABLE.exists():
    raise FileNotFoundError(f"Input table not found: {IN_TABLE}")

qc_df = pd.read_csv(IN_TABLE)

outcome_dmn = f"{FEATURE}_dmn_mean"
if outcome_dmn not in qc_df.columns:
    raise KeyError(
        f"Expected column '{outcome_dmn}' not found in table. "
        f"Check that build_feature_table.py created it."
    )

pipelines = sorted(qc_df["pipeline"].unique().tolist())
print("Pipelines in table:", pipelines)
print("Rows:", len(qc_df), "| Unique subs:", qc_df["sub"].nunique())

# -----------------------------
# 1) FD over TR (fMRIPrep) - anzeigen
# -----------------------------
run_fd_over_tr_plots()

# -----------------------------
# 2) % scrubbed volumes on mean FD (no_scrub vs scrub)
# -----------------------------
df_no = qc_df.loc[qc_df["pipeline"] == "no_scrub", ["sub", "mean_fd"]].rename(
    columns={"mean_fd": "mean_fd_no_scrub"}
)
df_scrub = qc_df.loc[qc_df["pipeline"] == "scrub", ["sub", "pct_scrub"]].rename(
    columns={"pct_scrub": "pct_scrub_scrub"}
)
plot_df = df_no.merge(df_scrub, on="sub", how="inner")

plt.figure(figsize=(6, 5))
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
plt.savefig(OUT_DIR / "scrubbed_volumes_on_mean_fd.png", dpi=150)
plt.show()

# -----------------------------
# 3) QC–Outcome: FD vs DMN outcome (Pearson + LM)
# -----------------------------
print("\n=== QC–Outcome correlations (fd_mean_halfpipe vs DMN outcome) ===")
for pipeline in ["no_scrub", "scrub"]:
    dfp = qc_df.query("pipeline == @pipeline").dropna(subset=[outcome_dmn, "fd_mean_halfpipe"])
    if len(dfp) < 3:
        print(f"{pipeline}: too few rows after NA drop")
        continue
    r, p = pearsonr(dfp["fd_mean_halfpipe"], dfp[outcome_dmn])
    print(f"{pipeline}: r = {r:.3f}, p = {p:.3g}, n = {len(dfp)}")

df_lm = qc_df.dropna(subset=[outcome_dmn, "fd_mean_halfpipe", "group"]).copy()
model = smf.ols(
    f"{outcome_dmn} ~ fd_mean_halfpipe * pipeline + group",
    data=df_lm
).fit()
print("\n=== Linear model summary ===")
print(model.summary())

g = sns.lmplot(
    data=qc_df,
    x="fd_mean_halfpipe",
    y=outcome_dmn,
    col="pipeline",
    hue="pipeline",
    ci=95,
    legend=False,
    palette=palette,
    height=5,
    aspect=1
)
g.set_axis_labels("Framewise Displacement (HALFpipe)", f"DMN {FEATURE.upper()}")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{FEATURE}_dmn_on_mean_fd.png", dpi=150)
plt.show()

# Outcome box/line plot across pipelines
plt.figure(figsize=(6, 5))
sns.boxplot(
    data=qc_df,
    x="pipeline",
    y=outcome_dmn,
    palette=palette,
    hue="pipeline",
    dodge=False,
)
sns.lineplot(
    data=qc_df,
    x="pipeline",
    y=outcome_dmn,
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
plt.ylabel(f"DMN {FEATURE.upper()}")
plt.savefig(OUT_DIR / f"mean_{FEATURE}_dmn_on_pipelines.png", dpi=150)
plt.show()

# -----------------------------
# 4) Mean FD across pipelines
# -----------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(
    data=qc_df,
    x="pipeline",
    y="mean_fd",
    palette=palette,
    hue="pipeline",
    dodge=False,
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
plt.savefig(OUT_DIR / "mean_fd_on_pipelines.png", dpi=150)
plt.show()

# -----------------------------
# 5) Remaining TRs
# -----------------------------
plt.figure(figsize=(7, 4))
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
plt.savefig(OUT_DIR / "remaining_tr_on_pipelines.png", dpi=150)
plt.show()

# -----------------------------
# 6) Halfpipe mean fd vs other mean fd (no_scrub only)
# -----------------------------
df_ns = qc_df[qc_df.pipeline == "no_scrub"].copy()

plt.figure(figsize=(6, 5))
sns.regplot(
    data=df_ns,
    x="mean_fd",
    y="fd_mean_halfpipe",
    color="black",
)
plt.xlabel("Mean FD (confounds)")
plt.ylabel("Mean FD (HALFpipe reportvals)")
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "different_mean_fds.png", dpi=150)
plt.show()

# -----------------------------
# 7) FD difference scrub - no_scrub
# -----------------------------
diff_df = (
    qc_df.pivot(index="sub", columns="pipeline", values="mean_fd")
    .dropna(subset=["scrub", "no_scrub"])
    .copy()
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
plt.savefig(OUT_DIR / "mean_fd_difference.png", dpi=150)
plt.show()