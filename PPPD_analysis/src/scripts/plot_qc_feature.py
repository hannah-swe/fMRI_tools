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
# Config (edit here)
# -----------------------------
FEATURE = "falff"  # "falff" oder "seedfc"
RUN = "run-01"
FD_THRESHOLD = 0.5

DO_GENERAL = True   # allgemeine QC plots (feature-unabhängig)
DO_FEATURE = True   # QC–Outcome plots (feature-spezifisch)

QC_TABLE = Path.cwd() / "outputs" / "qc_table.csv"
FEATURE_TABLE = Path.cwd() / "outputs" / f"qc_{FEATURE}_table.csv"

OUT_DIR_GENERAL = Path.cwd() / "outputs" / "plots" / "general"
OUT_DIR_FEATURE = Path.cwd() / "outputs" / "plots" / FEATURE
OUT_DIR_GENERAL.mkdir(parents=True, exist_ok=True)
OUT_DIR_FEATURE.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers: FD-over-TR (only show, don't save)
# -----------------------------
def plot_fd_over_tr(confounds_tsv: Path, sub: str, fd_threshold: float | None = None) -> None:
    conf = pd.read_csv(confounds_tsv, sep="\t")
    fd = conf["framewise_displacement"].fillna(0)

    plt.figure(figsize=(4, 4))
    plt.plot(fd.values, linewidth=0.8, color="mediumblue")
    plt.xlabel("TR")
    plt.ylabel("Framewise Displacement")
    plt.title(f"{sub} {RUN}".strip())
    sns.despine()

    if fd_threshold is not None:
        plt.axhline(fd_threshold, color="black", linestyle="--", linewidth=1)

    ymax = max(1, float(fd.max()))
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.show()

def run_fd_over_tr_plots() -> None:
    for sub_dir in sorted(fmriprep_dir.glob("sub-*")):
        sub = sub_dir.name
        confounds_tsv = find_confounds_timeseries(sub_dir=sub_dir, sub=sub, run=RUN)
        if confounds_tsv is None:
            continue
        plot_fd_over_tr(confounds_tsv, sub=sub, fd_threshold=FD_THRESHOLD)


# =========================================================
# GENERAL QC PLOTS (feature-independent)
# Uses: outputs/qc_table.csv
# =========================================================
if DO_GENERAL:
    if not QC_TABLE.exists():
        raise FileNotFoundError(f"QC table not found: {QC_TABLE} (run build_qc_table.py)")

    qc = pd.read_csv(QC_TABLE)
    print("\n[GENERAL] Loaded:", QC_TABLE)
    print("[GENERAL] Rows:", len(qc), "| Unique subs:", qc["sub"].nunique())

    # Plot 1: FD over TR
    run_fd_over_tr_plots()

    # Plot 2: % scrubbed volumes on mean FD
    plot_df = qc.loc[qc["pipeline"] == "scrub", ["sub", "fd_mean_halfpipe", "pct_scrub"]]
    plt.figure(figsize=(6, 5))
    sns.regplot( data=plot_df, x="fd_mean_halfpipe", y="pct_scrub", color="black")
    plt.xlabel("Mean FD")
    plt.ylabel("% scrubbed volumes")
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUT_DIR_GENERAL / "scrubbed_volumes_on_mean_fd.png", dpi=150)
    plt.show()

    # Plot 4: Mean FD across pipelines
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        data=qc,
        x="pipeline",
        y="fd_mean_halfpipe",
        palette=palette,
        hue="pipeline",
        dodge=False,
        linewidth=1,
        linecolor="black",
    )
    sns.lineplot(
        data=qc,
        x="pipeline",
        y="fd_mean_halfpipe",
        units="sub",
        estimator=None,
        legend=False,
        color="black",
        linewidth=1,
        alpha=0.5
    )
    sns.despine()
    plt.tight_layout()
    plt.xlabel("")
    plt.ylabel("Mean Framewise Displacement")
    plt.savefig(OUT_DIR_GENERAL / "mean_fd_on_pipelines.png", dpi=150)
    plt.show()

    # Plot 5: Remaining TRs
    plt.figure(figsize=(5, 5))
    sns.lineplot( data=qc, x="pipeline", y="remaining_tr", hue="sub", legend=False)
    sns.despine()
    plt.xlabel("")
    plt.ylabel("Remaining TRs")
    plt.tight_layout()
    plt.savefig(OUT_DIR_GENERAL / "remaining_tr_on_pipelines.png", dpi=150)
    plt.show()

    print("\n[GENERAL] Plots saved to:", OUT_DIR_GENERAL)


# =========================================================
# FEATURE-SPECIFIC PLOTS (QC–Outcome)
# Uses: outputs/qc_{FEATURE}_table.csv
# =========================================================
if DO_FEATURE:
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE} "
            f"(run build_feature_table.py for FEATURE='{FEATURE}')"
        )

    df = pd.read_csv(FEATURE_TABLE)
    outcome_dmn = f"{FEATURE}_dmn_mean"
    if outcome_dmn not in df.columns:
        raise KeyError(f"Missing column '{outcome_dmn}' in {FEATURE_TABLE}")

    print("\n[FEATURE] Loaded:", FEATURE_TABLE)
    print("[FEATURE] Outcome column:", outcome_dmn)
    print("[FEATURE] Rows:", len(df), "| Unique subs:", df["sub"].nunique())

    # Some stats (correlations and linear regression)
    print("\n=== QC–Outcome correlations (fd_mean_halfpipe vs DMN outcome) ===")
    for pipeline in ["no_scrub", "scrub"]:
        dfp = df.query("pipeline == @pipeline").dropna(subset=[outcome_dmn, "fd_mean_halfpipe"])
        if len(dfp) < 3:
            print(f"{pipeline}: too few rows after NA drop")
            continue
        r, p = pearsonr(dfp["fd_mean_halfpipe"], dfp[outcome_dmn])
        print(f"{pipeline}: r = {r:.3f}, p = {p:.3g}, n = {len(dfp)}")

    df_lm = df.dropna(subset=[outcome_dmn, "fd_mean_halfpipe", "group"]).copy()
    model = smf.ols(
        f"{outcome_dmn} ~ fd_mean_halfpipe * pipeline + group",
        data=df_lm
    ).fit()
    print("\n=== Linear model summary ===")
    print(model.summary())

    # Plot 1: QC–Outcome (Pearson + LM + lmplot + box/line)
    g = sns.lmplot(
        data=df,
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
    g.set_axis_labels("Mean Framewise Displacement", f"DMN {FEATURE.upper()}")
    plt.tight_layout()
    plt.savefig(OUT_DIR_FEATURE / f"{FEATURE}_dmn_on_mean_fd.png", dpi=150)
    plt.show()

    # Plot 2:
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        data=df,
        x="pipeline",
        y=outcome_dmn,
        palette=palette,
        hue="pipeline",
        dodge=False,
        linewidth=1,
        linecolor="black",
    )
    sns.lineplot(
        data=df,
        x="pipeline",
        y=outcome_dmn,
        units="sub",
        estimator=None,
        legend=False,
        color="black",
        linewidth=1,
        alpha=0.5
    )
    sns.despine()
    plt.tight_layout()
    plt.xlabel("")
    plt.ylabel(f"DMN {FEATURE.upper()}")
    plt.savefig(OUT_DIR_FEATURE / f"mean_{FEATURE}_dmn_on_pipelines.png", dpi=150)
    plt.show()

    print("\n[FEATURE] Plots saved to:", OUT_DIR_FEATURE)
