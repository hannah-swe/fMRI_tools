import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, rankdata
from PPPD import get_participants_tsv, get_selected_subject_list
from PPPD.subjects import subs, subjects_to_exclude

# Configuration
palette = {"control": "black", "patient": "teal"}
part = None

selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)
participants_df = get_participants_tsv()

report_path = "/data_wgs04/ag-sensomotorik/PPPD/data/reportvals_fd/"
plot_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/framewise_displacement/"
filenames = ["reportvals_1.txt", "reportvals_2.txt"]
groups = ["control", "patient"]
variables = ["fd_mean", "fd_max"]

# Read and combine QC files
qc_list = []

for f in filenames:
    report_df = pd.read_fwf(os.path.join(report_path, f))
    report_df.columns = report_df.columns.str.strip()

    qc_tmp = report_df.loc[
        report_df["run"] == "01",
        ["sub", "fd_mean", "fd_max", "fd_perc"]
    ].copy()

    qc_tmp = qc_tmp.rename(columns={"sub": "subject_num"})
    qc_tmp["subject_id"] = "sub-" + qc_tmp["subject_num"].astype(str).str.zfill(2)

    num_cols = ["subject_num", "fd_mean", "fd_max", "fd_perc"]
    qc_tmp[num_cols] = qc_tmp[num_cols].apply(pd.to_numeric, errors="coerce")

    qc_list.append(qc_tmp)

qc = pd.concat(qc_list, ignore_index=True)

# Keep selected subjects and sort
qc = qc[qc["subject_num"].isin(selected_subs)]
qc = qc.sort_values("subject_num").reset_index(drop=True)

# Add group information
participants_df = participants_df[["participant_id", "group"]].rename(
    columns={"participant_id": "subject_num"}
)

qc = qc.merge(participants_df, on="subject_num", how="left")

# Descriptive statistics
for var in variables:
    print(f"\n{'=' * 50}")
    print(f"Descriptive statistics: {var}")
    print(f"{'=' * 50}")
    print(qc.groupby("group")[var].describe().round(3))

# Statistical analyses
for var in variables:
    print(f"\n{'#' * 60}")
    print(f"VARIABLE: {var}")
    print(f"{'#' * 60}")

    # Normality checks
    for group in groups:
        x = qc.loc[qc["group"] == group, var].dropna()
        W, p = shapiro(x)

        print(f"\n{'=' * 50}")
        print(f"Normality check: {var} | group = {group}")
        print(f"{'=' * 50}")
        print(f"n = {len(x)}")
        print(f"Shapiro-Wilk: W = {W:.3f}, p = {p:.4f}")

        if p < 0.05:
            print("Interpretation: the distribution significantly deviates from normality")
        else:
            print("Interpretation: the distribution does not significantly deviate from normality")

        plt.figure(figsize=(6, 4))
        x.plot(kind="density")
        plt.title(f"Density plot: {var} ({group})")
        plt.xlabel(var)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        stats.probplot(x, dist="norm", plot=plt)
        plt.title(f"Q-Q plot: {var} ({group})")
        plt.tight_layout()
        plt.show()

    patients = qc.loc[qc["group"] == "patient", var].dropna()
    controls = qc.loc[qc["group"] == "control", var].dropna()

    # Welch's t-test
    result = ttest_ind(patients, controls, equal_var=False)

    print("\nWelch's t-test")
    print(f"t = {result.statistic:.3f}")
    print(f"df = {result.df:.3f}")
    print(f"p = {result.pvalue:.4f}")

    # Mann-Whitney U test
    n1, n2 = len(patients), len(controls)

    combined = pd.concat([
        pd.DataFrame({"value": patients.to_numpy(), "group": "patient"}),
        pd.DataFrame({"value": controls.to_numpy(), "group": "control"})
    ], ignore_index=True)

    combined["rank"] = rankdata(combined["value"], method="average")

    rank_statistics = (
        combined.groupby("group")["rank"]
        .agg(n="count", mean_rank="mean", rank_sum="sum")
        .reindex(groups)
    )

    u1, p_mw = mannwhitneyu(
        patients,
        controls,
        method="asymptotic",
        alternative="two-sided"
    )

    u2 = n1 * n2 - u1
    U = min(u1, u2)

    # Expected U
    mu_u = n1 * n2 / 2

    # Tie correction
    N = n1 + n2
    _, tie_counts = np.unique(combined["value"], return_counts=True)

    tie_term = np.sum(tie_counts ** 3 - tie_counts)

    sigma_u = np.sqrt((n1 * n2 / 12) * (N + 1 - tie_term / (N * (N - 1))))

    # Continuity correction
    if U < mu_u:
        correction = 0.5
    elif U > mu_u:
        correction = -0.5
    else:
        correction = 0

    z = (U - mu_u + correction) / sigma_u

    # Effect size
    r = abs(z) / np.sqrt(N)

    print("\nMann-Whitney U test")
    print(rank_statistics.round(3))
    print(f"\nU = {U:.1f}")
    print(f"z = {z:.3f}")
    print(f"p = {p_mw:.4f}")
    print(f"r = {r:.3f}")

    # Boxplot with individual observations
    plt.figure(figsize=(2.5, 4))

    sns.boxplot(
        data=qc,
        x="group",
        y=var,
        hue="group",
        order=groups,
        hue_order=groups,
        showfliers=False,
        palette=palette,
        linewidth=1.5,
        fill=False,
        legend=False
    )

    sns.stripplot(
        data=qc,
        x="group",
        y=var,
        hue="group",
        order=groups,
        hue_order=groups,
        jitter=True,
        alpha=0.6,
        palette=palette,
        legend=False,
        size=6
    )

    plt.xlabel("")
    plt.ylabel(var)
    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{var}_boxplot_difference_by_group.svg")
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()