import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind
from PPPD import (get_connectivity_path, get_main_values_tables_path)
from IPython.display import display


# --- config:
palette = {"control": "black", "patient": "teal"}

group_order = [1.0, 2.0, 3.0]
group_labels = {
    1.0: "Primary PPPD",
    2.0: "Secondary PPPD",
    3.0: "Control"
}
palette_group = {
    1.0: "darkturquoise",
    2.0: "seagreen",
    3.0: palette["control"]
}


# --- Load main values table
main_values_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")
main_df = pd.read_csv(main_values_path)


# --- Variables of vestibular tests
variables = [
    'vhit_right_median_gain',
    'vhit_right_sd',
    'vhit_left_median_gain',
    'vhit_left_sd',
    'vhit_total_average',
    'svv_average_s',
    'svv_sd_s',
    'svv_average_d',
    'svv_sd_d',
    'ovemp_left_amp_diff',
    'ovemp_right_amp_diff',
    'ovemp_average_amp_diff'
]


# --- Get descriptive statistics for each group
descriptives_Group = (
    main_df.groupby('Group')[variables]
      .agg(['count', 'mean', 'std', 'median', 'min', 'max'])
      .round(3)
)

descriptives_group = (
    main_df.groupby('group')[variables]
      .agg(['count', 'mean', 'std', 'median', 'min', 'max'])
      .round(3)
)

display(descriptives_Group)
display(descriptives_group)


# --- vHIT variables boxplot with individual observations
vhit_variables = [
    'vhit_right_median_gain',
    'vhit_left_median_gain',
    'vhit_total_average',
]

for var in vhit_variables:
    plt.figure(figsize=(3, 4.75))

    plt.axhline(0.7, color="crimson", linewidth=1.5, alpha=0.5)

    sns.boxplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        showfliers=False,
        palette=palette_group,
        linewidth=1.5,
        fill=False,
        legend=False
    )

    sns.stripplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        jitter=True,
        alpha=0.6,
        palette=palette_group,
        legend=False,
        size=6
    )

    plt.xticks(
        ticks=range(len(group_order)),
        labels=[group_labels[g] for g in group_order],
        rotation=45
    )
    plt.xlabel("")
    plt.ylabel(var)
    sns.despine()
    plt.tight_layout()
    # out_path = os.path.join(plot_dir, f"{var}_boxplot_difference_by_group.svg")
    # plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()


# --- SVV variables boxplot with individual observations
svv_average_variables = [
    'svv_average_s',
    'svv_average_d',
]

for var in svv_average_variables:
    plt.figure(figsize=(3, 4.75))

    plt.axhline(2.5, color="crimson", linewidth=1.5, alpha=0.5)

    sns.boxplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        showfliers=False,
        palette=palette_group,
        linewidth=1.5,
        fill=False,
        legend=False
    )

    sns.stripplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        jitter=True,
        alpha=0.6,
        palette=palette_group,
        legend=False,
        size=6
    )

    plt.xticks(
        ticks=range(len(group_order)),
        labels=[group_labels[g] for g in group_order],
        rotation=45
    )
    plt.xlabel("")
    plt.ylabel(var)
    sns.despine()
    plt.tight_layout()
    # out_path = os.path.join(plot_dir, f"{var}_boxplot_difference_by_group.svg")
    # plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()

svv_sd_variables = [
    'svv_sd_s',
    'svv_sd_d',
]

for var in svv_sd_variables:
    plt.figure(figsize=(3, 4.75))

    sns.boxplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        showfliers=False,
        palette=palette_group,
        linewidth=1.5,
        fill=False,
        legend=False
    )

    sns.stripplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        jitter=True,
        alpha=0.6,
        palette=palette_group,
        legend=False,
        size=6
    )

    plt.xticks(
        ticks=range(len(group_order)),
        labels=[group_labels[g] for g in group_order],
        rotation=45
    )
    plt.xlabel("")
    plt.ylabel(var)
    sns.despine()
    plt.tight_layout()
    # out_path = os.path.join(plot_dir, f"{var}_boxplot_difference_by_group.svg")
    # plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()


# --- oVEMP variables boxplot with individual observations
ovemp_variables = [
    'ovemp_left_amp_diff',
    'ovemp_right_amp_diff',
    'ovemp_average_amp_diff'
]

for var in ovemp_variables:
    plt.figure(figsize=(3, 4.75))

    sns.boxplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        showfliers=False,
        palette=palette_group,
        linewidth=1.5,
        fill=False,
        legend=False
    )

    sns.stripplot(
        data=main_df,
        x="Group",
        y=var,
        hue="Group",
        order=group_order,
        hue_order=group_order,
        jitter=True,
        alpha=0.6,
        palette=palette_group,
        legend=False,
        size=6
    )

    plt.xticks(
        ticks=range(len(group_order)),
        labels=[group_labels[g] for g in group_order],
        rotation=45
    )
    plt.xlabel("")
    plt.ylabel(var)
    sns.despine()
    plt.tight_layout()
    # out_path = os.path.join(plot_dir, f"{var}_boxplot_difference_by_group.svg")
    # plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.show()


# --- Variables for group comparisons
test_variables = [
    'vhit_right_median_gain',
    'vhit_left_median_gain',
    'svv_average_s',
    'svv_average_d',
    'svv_sd_s',
    'svv_sd_d',
    'ovemp_left_amp_diff',
    'ovemp_right_amp_diff',
]


# ============================================================
# Welch's t-test: patients vs. controls
# Variable "group": patient / control
# ============================================================

results_group = []

for var in test_variables:

    patient = main_df.loc[
        main_df["group"] == "patient", var
    ].dropna()

    control = main_df.loc[
        main_df["group"] == "control", var
    ].dropna()

    # Welch's t-test
    result = ttest_ind(
        patient,
        control,
        equal_var=False
    )

    results_group.append({
        "variable": var,
        "n_patient": len(patient),
        "mean_patient": patient.mean(),
        "sd_patient": patient.std(ddof=1),
        "n_control": len(control),
        "mean_control": control.mean(),
        "sd_control": control.std(ddof=1),
        "t": result.statistic,
        "df": result.df,
        "p": result.pvalue
    })


welch_group = pd.DataFrame(results_group)

# Round values for display
welch_group = welch_group.round({
    "mean_patient": 3,
    "sd_patient": 3,
    "mean_control": 3,
    "sd_control": 3,
    "t": 3,
    "df": 2,
    "p": 4
})

print("\nWelch's t-tests: patients vs. controls")
display(welch_group)

# ============================================================
# Welch's t-test: Primary PPPD vs. Secondary PPPD
# Variable "Group": 1 = Primary, 2 = Secondary
# ============================================================

results_pppd = []

for var in test_variables:

    primary = main_df.loc[
        main_df["Group"] == 1.0, var
    ].dropna()

    secondary = main_df.loc[
        main_df["Group"] == 2.0, var
    ].dropna()

    # Welch's t-test
    result = ttest_ind(
        primary,
        secondary,
        equal_var=False
    )

    results_pppd.append({
        "variable": var,
        "n_primary": len(primary),
        "mean_primary": primary.mean(),
        "sd_primary": primary.std(ddof=1),
        "n_secondary": len(secondary),
        "mean_secondary": secondary.mean(),
        "sd_secondary": secondary.std(ddof=1),
        "t": result.statistic,
        "df": result.df,
        "p": result.pvalue
    })


welch_pppd = pd.DataFrame(results_pppd)

welch_pppd = welch_pppd.round({
    "mean_primary": 3,
    "sd_primary": 3,
    "mean_secondary": 3,
    "sd_secondary": 3,
    "t": 3,
    "df": 2,
    "p": 4
})

print("\nWelch's t-tests: Primary PPPD vs. Secondary PPPD")
display(welch_pppd)
