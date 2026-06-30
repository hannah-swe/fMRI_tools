import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PPPD import (get_main_values_tables_path, get_connectivity_path)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

plot_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/boxplots/"

# define color palette for plotting
palette = {"control": "black", "patient": "teal"}
# palette = {"control": "gold", "patient": "hotpink"}


# --- Get connectivity dataframe with subject values for all significant cluster per seed
connectivity_df_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_wide_format.csv")
connectivity_df = pd.read_csv(connectivity_df_path)
connectivity_long_df_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_long_format.csv")
connectivity_long_df = pd.read_csv(connectivity_long_df_path)

# --- Get falff dataframe with subject values for significant cluster
falff_df_path = os.path.join(get_connectivity_path(), "falff_pre-data_dataframe_wide_format.csv")
falff_df = pd.read_csv(falff_df_path)

# --- Merge connectivity and main values dataframe
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


# Alle FC-Spalten finden
fc_cols = [col for col in connectivity_df.columns if col.endswith("mean")]

sns.set_theme(style="ticks")
sns.set_context()

for fc_col in fc_cols:

    # cluster_label aus Spaltennamen ableiten
    # z.B. IPLPFcmL__Putamen_R_mean -> IPLPFcmL__Putamen_R
    cluster_label = fc_col.replace("_mean", "")

    # p_value aus connectivity_long_df holen
    p_vals = connectivity_long_df.loc[
        connectivity_long_df["cluster_label"] == cluster_label,
        "p_value"
    ].dropna().unique()

    if len(p_vals) == 0:
        print(f"Kein p_value gefunden für {cluster_label}")
        cluster_p = np.nan
        stars = "n.s."
    else:
        cluster_p = p_vals[0]

        if cluster_p < 0.001:
            stars = "***"
        elif cluster_p < 0.01:
            stars = "**"
        elif cluster_p < 0.05:
            stars = "*"
        else:
            stars = "n.s."

    plt.figure(figsize=(2.5, 4))

    sns.boxplot(
        data=connectivity_df,
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
        data=connectivity_df,
        x="group",
        y=fc_col,
        hue="group",
        jitter=True,
        alpha=0.6,
        palette=palette,
        legend=False,
        size=6
    )

    y_max = connectivity_df[fc_col].max()
    y_min = connectivity_df[fc_col].min()
    h = 0.02 * (y_max - y_min)
    y = y_max + h

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


