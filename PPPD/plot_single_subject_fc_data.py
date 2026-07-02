import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PPPD import (get_main_values_tables_path, get_connectivity_path)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore


# function to get asteriks
def p_to_stars(p):
    if pd.isna(p):
        return "n.s."
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# --- Output directory
plot_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/boxplots/"


# --- Plot configuration
# define color palette
palette = {"control": "black", "patient": "teal"}
# palette = {"control": "teal", "patient": "hotpink"} # old palette
# set plot style
sns.set_theme(style="ticks")
sns.set_context()


# --- Get connectivity dataframe with subject values for all significant cluster per seed
# connectivity dataframe in wide format for mean values of functional connectivity per subject
connectivity_df_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_wide_format.csv")
connectivity_df = pd.read_csv(connectivity_df_path)
# connectivity dataframe in long format for p-values
connectivity_long_df_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_long_format.csv")
connectivity_long_df = pd.read_csv(connectivity_long_df_path)


# --- Get falff dataframe with subject values for significant cluster
# falff dataframe in wide format for mean values of functional connectivity per subject
falff_df_path = os.path.join(get_connectivity_path(), "falff_pre-data_dataframe_wide_format.csv")
falff_df = pd.read_csv(falff_df_path)
# connectivity dataframe in long format for p-values
falff_long_df_path = os.path.join(get_connectivity_path(), "falff_pre-data_dataframe_long_format.csv")
falff_long_df = pd.read_csv(falff_long_df_path)


# --- Merge connectivity and falff dataframe (wide format)
assert connectivity_df["subject_num"].is_unique
assert falff_df["subject_num"].is_unique
df_full = connectivity_df.merge(
    falff_df,
    on="subject_num",
    how="left",
    suffixes=("_conn", "_falff")
)
# quality check: ensure group labels are identical
if not (df_full["group_conn"] == df_full["group_falff"]).all():
    mismatches = df_full.loc[
        df_full["group_conn"] != df_full["group_falff"],
        ["subject_num", "group_conn", "group_falff"]
    ]
    raise ValueError(
        f"Mismatch found in group labels:\n{mismatches}"
    )
# clean up columns
df_full = df_full.drop(columns=["group_falff", "subject_id_falff"])
df_full = df_full.rename(columns={"group_conn": "group"})


# --- Concatenate connectivity and falff dataframe (long format)
df_full_long = pd.concat([connectivity_long_df, falff_long_df], ignore_index=True)


# --- Prepare z-scored dataframe
# find all columns with mean values
fc_cols = [col for col in df_full.columns if col.endswith("mean")]
# copy dataframe
connectivity_df_z = df_full.copy()
# z-transformation
connectivity_df_z[fc_cols] = connectivity_df_z[fc_cols].apply(
    zscore,
    nan_policy="omit"
)


# PLOT 1:
# --- Loop over all relevant FC columns to get one boxplot for each FC cluster with group difference
for fc_col in fc_cols:
    # get cluster_label from column header
    cluster_label = fc_col.replace("_mean", "")

    # get p_value from connectivity_long_df
    p = df_full_long.loc[
        df_full_long["cluster_label"] == cluster_label,
        "p_value"
    ].dropna().unique()

    # get asteriks
    stars = p_to_stars(p)

    # boxplot with single subject point
    plt.figure(figsize=(2.5, 4))
    sns.boxplot(
        data=connectivity_df_z,
        x="group",
        y=fc_col,
        hue="group",
        showfliers=False,
        palette=palette,
        linewidth=1.5,
        fill=False,
        legend=False
    )
    sns.stripplot(
        data=connectivity_df_z,
        x="group",
        y=fc_col,
        hue="group",
        jitter=True,
        alpha=0.6,
        palette=palette,
        legend=False,
        size=6
    )
    # significant brackets
    y_max = connectivity_df_z[fc_col].max()
    y_min = connectivity_df_z[fc_col].min()
    h = 0.02 * (y_max - y_min)
    bracket_offset = 3 * h
    y = y_max + bracket_offset
    x1, x2 = 0, 1
    plt.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=1.5,
        c="black"
    )
    plt.text(
        (x1 + x2) * 0.5,
        y + h,
        stars,
        ha="center",
        va="bottom",
        fontsize=16,
        weight="bold"
    )

    plt.xlabel("")
    plt.ylabel(cluster_label)
    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{cluster_label}_boxplot_difference_by_group.svg")
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()


# --- Mapping to define which FCs should go in one boxplot
connectivity_plot_groups = {
    "plot_01_panel_a": [
        "V5L__Cerebellum_6_R",
        "V5R__Cerebellum_Crus1_L",
    ],
    "plot_02_panel_b": [
        "IPLPFcmL__Vermis_8",
        "OperculumOP1L__Vermis_8",
        "OperculumOP1R__Vermis_9",
        "IPLPFcmL__Putamen_R",
    ],
    "plot_03_panel_c": [
        "OperculumOP4L__Thal_VL_R",
        "InsulaOP3RAnat__Cerebellum_Crus2_L"
    ],
    "plot_04_panel_d": [
        "InsulaIg2L__Lingual_L",
        "InsulaIg2L__SupraMarginal_L"
    ],
    "plot_05_falff": [
        "falff__Hippocampus_R"
    ]
}


# --- Long format dataframe
plot_long_df = connectivity_df_z.melt(
    id_vars=["subject_num", "group"],
    value_vars=fc_cols,
    var_name="fc_col",
    value_name="z_scored_connectivity"
)
# cluster_label from column header
plot_long_df["cluster_label"] = (
    plot_long_df["fc_col"]
    .str.replace("_mean", "", regex=False)
)


# PLOT 2:
# --- Boxplots with multiple FCs (defined in connectivity_plot_groups)
for plot_name, cluster_labels in connectivity_plot_groups.items():
    # get cluster_label
    plot_df = plot_long_df[
        plot_long_df["cluster_label"].isin(cluster_labels)
    ].copy()

    # keep order of mapping for x-axis
    plot_df["cluster_label"] = pd.Categorical(
        plot_df["cluster_label"],
        categories=cluster_labels,
        ordered=True
    )

    # use numbers instead of cluster_labels for x-axis
    cluster_numbers = {
        label: str(i + 1)
        for i, label in enumerate(cluster_labels)
    }
    plot_df["cluster_number"] = plot_df["cluster_label"].map(cluster_numbers)

    # plot
    plt.figure(figsize=(1.7 * len(cluster_labels), 3.5))
    sns.boxplot(
        data=plot_df,
        x="cluster_number",
        y="z_scored_connectivity",
        hue="group",
        palette=palette,
        showfliers=False,
        linewidth=1.5,
        fill=False,
        legend=False,
    )
    sns.stripplot(
        data=plot_df,
        x="cluster_number",
        y="z_scored_connectivity",
        hue="group",
        palette=palette,
        dodge=True,
        jitter=True,
        alpha=0.5,
        size=5,
        legend=False,
    )
    ax = plt.gca()
    # n_clusters = len(cluster_labels)
    # ax.set_xlim(-0.6, n_clusters - 0.3)

    # significance brackets
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.5)
    y = ymax + 0.15  # Position der Klammer
    h = 0.08  # Höhe der Klammer
    offset = 0.2  # Abstand der beiden Gruppen vom Zentrum
    for i, cluster_label in enumerate(cluster_labels):
        # get p-value
        p = df_full_long.loc[
            df_full_long["cluster_label"] == cluster_label,
            "p_value"
        ].dropna().unique()
        stars = p_to_stars(p)
        x1 = i - offset
        x2 = i + offset
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1.2,
            c="black"
        )
        ax.text(
            i,
            y + h,
            stars,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    plt.xlabel("")
    plt.ylabel("z-scored connectivity")
    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{plot_name}_combined_connectivities_by_group.svg")
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()
