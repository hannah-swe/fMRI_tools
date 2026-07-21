import os
from itertools import combinations
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from PPPD import (get_connectivity_path, get_main_values_tables_path)


# --- Script Configuration:
ALPHA = 0.05
MIN_N = 3
GROUPS = ("control", "patient")

PALETTE = {
    "control": "teal",
    "patient": "hotpink",
}

BRAIN_VARS = [
    "V5L__Cerebellum_6_R_median",
    "V5R__Cerebellum_Crus1_L_median",
    "IPLPFcmL__Vermis_8_median",
    "OperculumOP1L__Vermis_8_median",
    "OperculumOP1R__Vermis_9_median",
    "InsulaOP3RAnat__Cerebellum_Crus2_L_median",
    "InsulaIg2L__Lingual_L_median",
    "InsulaIg2L__SupraMarginal_L_median",
    "falff__Hippocampus_R_median",
]

BEHAVIOR_VARS = [
    "age",
    "disease_duration",
    "GVS_threshold_mri",
    "ALQ_total",
    "Niigata_total",
    "MSSQ_raw",
    "HADS_A_total",
    "HADS_D_total",
    "Neo.Skala_n",
    "EOfirm_speed",
    "ECfirm_speed",
    "EOfirm_rating",
    "ECfirm_rating",
]

BRAIN_LABELS = {
    "IPLPFcmL__Vermis_8_median": "IPL–Verm",
    "InsulaIg2L__Lingual_L_median": "Ig2–LingG",
    "InsulaIg2L__SupraMarginal_L_median": "Ig2–SMG",
    "InsulaOP3RAnat__Cerebellum_Crus2_L_median": "OP3R–Crus2",
    "OperculumOP1L__Vermis_8_median": "OP1L–Verm",
    "OperculumOP1R__Vermis_9_median": "OP1R–Verm",
    "V5L__Cerebellum_6_R_median": "V5L–Crus1R",
    "V5R__Cerebellum_Crus1_L_median": "V5R–Crus1L",
    "falff__Hippocampus_R_median": "fALFF–Hippo",
}

BEHAVIOR_LABELS = {
    "age": "Age",
    "disease_duration": "DoD",
    "GVS_threshold_mri": "GVS threshold",
    "ALQ_total": "ALQ",
    "Niigata_total": "Niigata",
    "MSSQ_raw": "MSSQ",
    "HADS_A_total": "HADS-A",
    "HADS_D_total": "HADS-D",
    "Neo.Skala_n": "NEO-N",
    "EOfirm_speed": "Sway speed EO",
    "ECfirm_speed": "Sway speed EC",
    "EOfirm_rating": "Sway rating EO",
    "ECfirm_rating": "Sway rating EC",
}



# --- General helper functions
def ensure_columns_exist(df, columns, dataframe_name):
    """
    Checks whether all required columns are available.
    """
    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in {dataframe_name}: "
            f"{missing_columns}"
        )


def save_and_show_figure(output_path):
    """
    Saves the current Matplotlib figure and closes it afterwards.
    """
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def format_p_value(p_value):
    """
    Formats p-values for plot annotations.
    """
    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"

def get_significance_stars(p_value):
    """
    Returns significance stars based on the supplied p-value.
    """
    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return ""


# --- Data loading and merging
def merge_feature_table(
    base_df,
    feature_df,
    feature_name,
):
    """
    Merges a feature table into the main dataframe and verifies that group
    labels agree.
    The feature dataframe must contain:
        - subject_num
        - group
    """
    required_columns = ["subject_num", "group"]

    ensure_columns_exist(
        base_df,
        required_columns,
        "base dataframe",
    )
    ensure_columns_exist(
        feature_df,
        required_columns,
        feature_name,
    )

    if not base_df["subject_num"].is_unique:
        raise ValueError(
            "subject_num is not unique in the base dataframe."
        )

    if not feature_df["subject_num"].is_unique:
        raise ValueError(
            f"subject_num is not unique in {feature_name}."
        )

    feature_data = feature_df.rename(columns={"group": f"group_{feature_name}"})

    merged_df = base_df.merge(
        feature_data,
        on="subject_num",
        how="left",
        validate="one_to_one",
    )

    feature_group_column = f"group_{feature_name}"

    comparable_rows = merged_df[feature_group_column].notna()

    mismatches = merged_df.loc[
        comparable_rows
        & (
            merged_df["group"]
            != merged_df[feature_group_column]
        ),
        [
            "subject_num",
            "group",
            feature_group_column,
        ],
    ]

    if not mismatches.empty:
        raise ValueError(
            f"Group-label mismatch after merging {feature_name}:\n"
            f"{mismatches}"
        )

    return merged_df.drop(columns=[feature_group_column])


def load_analysis_data():
    """
    Loads and combines the behavioral, connectivity and fALFF data.
    """
    connectivity_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_wide_format.csv")
    falff_path = os.path.join(get_connectivity_path(), "falff_pre-data_dataframe_wide_format.csv")
    main_values_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")

    main_df = pd.read_csv(main_values_path)
    connectivity_df = pd.read_csv(connectivity_path)
    falff_df = pd.read_csv(falff_path)

    ensure_columns_exist(
        main_df,
        ["subject_num", "group"],
        "main dataframe",
    )

    df_full = merge_feature_table(
        base_df=main_df,
        feature_df=connectivity_df,
        feature_name="connectivity",
    )

    df_full = merge_feature_table(
        base_df=df_full,
        feature_df=falff_df,
        feature_name="falff",
    )

    required_analysis_columns = (
        ["subject_num", "group"]
        + BRAIN_VARS
        + BEHAVIOR_VARS
    )

    ensure_columns_exist(
        df_full,
        required_analysis_columns,
        "combined analysis dataframe",
    )

    return df_full


# --- Statistical functions:
def calculate_spearman_rlm(
    df,
    x_var,
    y_var,
    min_n=MIN_N,
):
    """
    Calculates:
        1. Spearman correlation between x and y
        2. Robust linear model y ~ x using Huber's T norm
    Returns NaN values if there are too few valid observations.
    """
    data = df[[x_var, y_var]].dropna().copy()

    result = {
        "rho": np.nan,
        "p": np.nan,
        "beta": np.nan,
        "intercept": np.nan,
        "n": len(data),
        "status": "too_few_valid_cases",
    }

    if len(data) < min_n:
        return result

    x = pd.to_numeric(
        data[x_var],
        errors="coerce",
    )

    y = pd.to_numeric(
        data[y_var],
        errors="coerce",
    )

    numeric_data = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    result["n"] = len(numeric_data)

    if len(numeric_data) < min_n:
        return result

    x = numeric_data["x"]
    y = numeric_data["y"]

    if x.nunique() < 2 or y.nunique() < 2:
        result["status"] = "constant_variable"
        return result

    rho, p_value = spearmanr(x, y)

    result["rho"] = rho
    result["p"] = p_value
    result["status"] = "ok"

    design_matrix = sm.add_constant(
        x,
        has_constant="add",
    )

    try:
        model = sm.RLM(
            y,
            design_matrix,
            M=sm.robust.norms.HuberT(),
        )

        fit = model.fit()

        result["intercept"] = fit.params.iloc[0]
        result["beta"] = fit.params.iloc[1]

    except Exception as error:
        result["status"] = (
            f"correlation_ok_rlm_failed: "
            f"{type(error).__name__}"
        )

    return result


def add_bonferroni_per_behavior(
    results_df,
    group_column="group",
    behavior_column="behavior",
    p_column="p",
    alpha=ALPHA,
):
    """
    Bonferroni correction separately

        - for each group
        - for each behavioral variable

    i.e. one correction over all brain variables belonging to one
    behavioral variable.

    New columns
    -----------
    p_bonf
    significant_bonf
    """

    corrected_df = results_df.copy()

    corrected_df["p_bonf"] = np.nan
    corrected_df["significant_bonf"] = False

    for (group, behavior), subset in corrected_df.groupby(
        [group_column, behavior_column],
        sort=False,
    ):

        valid_idx = subset.index[subset[p_column].notna()]

        if len(valid_idx) == 0:
            continue

        reject, p_corrected, _, _ = multipletests(
            corrected_df.loc[valid_idx, p_column],
            alpha=alpha,
            method="bonferroni",
        )

        corrected_df.loc[valid_idx, "p_bonf"] = p_corrected
        corrected_df.loc[valid_idx, "significant_bonf"] = reject

    return corrected_df


def add_fdr_correction(
    results_df,
    group_column="group",
    p_column="p",
    alpha=ALPHA,
):
    """
    Applies Benjamini-Hochberg FDR correction separately within each group.
    Added columns:
        p_fdr
        significant_uncorrected
        significant_fdr
    Only non-missing p-values are included in each correction family.
    """
    corrected_df = results_df.copy()

    corrected_df["p_fdr"] = np.nan
    corrected_df["significant_uncorrected"] = (
        corrected_df[p_column] < alpha
    )
    corrected_df["significant_fdr"] = False

    for _, group_df in corrected_df.groupby(
        group_column,
        sort=False,
    ):
        valid_indices = group_df.index[
            group_df[p_column].notna()
        ]

        if len(valid_indices) == 0:
            continue

        reject, corrected_p_values, _, _ = multipletests(
            corrected_df.loc[valid_indices, p_column],
            alpha=alpha,
            method="fdr_bh",
        )

        corrected_df.loc[
            valid_indices,
            "p_fdr",
        ] = corrected_p_values

        corrected_df.loc[
            valid_indices,
            "significant_fdr",
        ] = reject

    return corrected_df


# --- Brain–behavior correlations
def calculate_brain_behavior_correlations(
    group_data,
    brain_vars,
    behavior_vars,
):
    """
    Calculates every brain–behavior correlation separately for each group.
    FDR-BH correction is subsequently applied separately within each group
    across all valid brain–behavior tests.
    """
    results = []

    for group in GROUPS:
        df_group = group_data[group]

        for brain_var in brain_vars:
            for behavior_var in behavior_vars:

                if behavior_var == "EOfirm_speed":
                    df_group = df_group[df_group["subject_num"] != 37]

                result = calculate_spearman_rlm(
                    df=df_group,
                    x_var=behavior_var,
                    y_var=brain_var,
                )

                result.update({
                    "group": group,
                    "brain": brain_var,
                    "behavior": behavior_var,
                })

                results.append(result)

    results_df = pd.DataFrame(results)

    results_df = add_bonferroni_per_behavior(
        results_df,
        group_column="group",
        behavior_column="behavior",
        p_column="p",
        alpha=ALPHA,
    )

    return results_df


def create_brain_behavior_matrices(
    results_df,
    group,
):
    """
    Creates rho, uncorrected-p and FDR-p matrices for one group.
    """
    group_results = results_df.loc[
        results_df["group"] == group
    ]

    rho_matrix = group_results.pivot(
        index="behavior",
        columns="brain",
        values="rho",
    )

    p_matrix = group_results.pivot(
        index="behavior",
        columns="brain",
        values="p",
    )

    p_bonf_matrix = group_results.pivot(
        index="behavior",
        columns="brain",
        values="p_bonf",
    )

    rho_matrix = rho_matrix.reindex(
        index=BEHAVIOR_VARS,
        columns=BRAIN_VARS,
    )

    p_matrix = p_matrix.reindex(
        index=BEHAVIOR_VARS,
        columns=BRAIN_VARS,
    )

    p_bonf_matrix = p_bonf_matrix.reindex(
        index=BEHAVIOR_VARS,
        columns=BRAIN_VARS,
    )

    return rho_matrix, p_matrix, p_bonf_matrix


def plot_brain_behavior_heatmap(
    results_df,
    group,
    output_dir,
):
    """
    Plots a brain–behavior heatmap.
    Cell color:
        Spearman rho
    Cell annotation:
        FDR-corrected p-value
    Fully opaque annotation:
        p_fdr < ALPHA
    Faded annotation:
        p_fdr >= ALPHA
    """
    (
        rho_matrix,
        p_matrix,
        p_bonf_matrix,
    ) = create_brain_behavior_matrices(
        results_df=results_df,
        group=group,
    )

    rho_plot = rho_matrix.rename(
        index=BEHAVIOR_LABELS,
        columns=BRAIN_LABELS,
    )

    p_bonf_plot = p_bonf_matrix.rename(
        index=BEHAVIOR_LABELS,
        columns=BRAIN_LABELS,
    )

    plt.figure(figsize=(11, 9))

    ax = sns.heatmap(
        rho_plot,
        cmap="coolwarm",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        annot=False,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Spearman ρ"},
    )

    for row_index in range(p_bonf_plot.shape[0]):
        for column_index in range(p_bonf_plot.shape[1]):
            p_bonf = p_bonf_plot.iloc[
                row_index,
                column_index,
            ]

            if pd.isna(p_bonf):
                continue

            is_significant = p_bonf < ALPHA

            annotation = format_p_value(p_bonf)

            if is_significant:
                annotation = f"{annotation}*"

            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                annotation,
                ha="center",
                va="center",
                color="black",
                alpha=1.0 if is_significant else 0.25,
                fontsize=9,
            )

    ax.set_title(
        f"{group}: brain–behavior correlations\n"
        f"p-values Bonferroni-corrected; "
        f"* p-Bonf < {ALPHA}"
    )

    ax.set_xlabel("Brain variable")
    ax.set_ylabel("Behavior variable")

    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.yticks(rotation=0)

    output_path = os.path.join(
        output_dir,
        f"brain_behavior_heatmap_{group}_bonf.png",
    )

    save_and_show_figure(output_path)

    return rho_matrix, p_matrix, p_bonf_matrix


def plot_brain_behavior_scatter(
    brain_var,
    behavior_var,
    results_df,
    group_data,
    output_dir,
):
    """
    Plots both groups together and adds one robust regression line per group.
    """
    plot_data = []

    for group in GROUPS:
        data = group_data[group][
            [brain_var, behavior_var]
        ].copy()

        data["group"] = group
        plot_data.append(data)

    plot_df = pd.concat(
        plot_data,
        ignore_index=True,
    ).dropna(
        subset=[brain_var, behavior_var]
    )

    if behavior_var == "EOfirm_speed":
        plot_df = plot_df[plot_df["EOfirm_speed"] < 60]

    if plot_df.empty:
        return

    plt.figure(figsize=(9, 7))

    ax = sns.scatterplot(
        data=plot_df,
        x=behavior_var,
        y=brain_var,
        hue="group",
        hue_order=GROUPS,
        palette=PALETTE,
        s=90,
        alpha=0.7,
    )

    statistics_text = []

    for group in GROUPS:
        group_plot_data = plot_df.loc[
            plot_df["group"] == group
        ]

        result_rows = results_df.loc[
            (results_df["group"] == group)
            & (results_df["brain"] == brain_var)
            & (results_df["behavior"] == behavior_var)
        ]

        if result_rows.empty:
            continue

        result = result_rows.iloc[0]

        rho = result["rho"]
        p_value = result["p"]
        p_bonf = result["p_bonf"]
        beta = result["beta"]
        intercept = result["intercept"]
        n = int(result["n"])

        if (
            pd.notna(beta)
            and pd.notna(intercept)
            and len(group_plot_data) > 1
            and group_plot_data[behavior_var].nunique() > 1
        ):
            x_line = np.linspace(
                group_plot_data[behavior_var].min(),
                group_plot_data[behavior_var].max(),
                100,
            )

            y_line = intercept + beta * x_line

            ax.plot(
                x_line,
                y_line,
                color=PALETTE[group],
                linewidth=3,
            )

        statistics_text.append(
            f"{group}: "
            f"ρ = {rho:.2f}, "
            f"p = {format_p_value(p_value)}, "
            f"p-Bonf = {format_p_value(p_bonf)}"
        )

    ax.set_xlabel(
        BEHAVIOR_LABELS.get(
            behavior_var,
            behavior_var,
        )
    )

    ax.set_ylabel(
        BRAIN_LABELS.get(
            brain_var,
            brain_var,
        )
    )

    ax.set_title("\n".join(statistics_text))

    ax.legend(
        title="Group",
        frameon=False,
    )

    sns.despine()

    output_path = os.path.join(
        output_dir,
        f"brain_behavior_{brain_var}_{behavior_var}.png",
    )

    save_and_show_figure(output_path)


# --- Brain–brain correlations
def calculate_brain_brain_correlations(
    group_data,
    brain_vars,
):
    """
    Calculates all unique pairs of brain variables separately by group.
    With nine brain variables, this produces 36 tests per group.
    FDR-BH correction is applied separately within each group.
    """
    results = []

    for group in GROUPS:
        df_group = group_data[group]

        for brain_x, brain_y in combinations(
            brain_vars,
            2,
        ):
            result = calculate_spearman_rlm(
                df=df_group,
                x_var=brain_x,
                y_var=brain_y,
            )

            result.update({
                "group": group,
                "brain_x": brain_x,
                "brain_y": brain_y,
            })

            results.append(result)

    results_df = pd.DataFrame(results)

    return add_fdr_correction(
        results_df=results_df,
        group_column="group",
        p_column="p",
        alpha=ALPHA,
    )


def create_symmetric_brain_matrices(
    results_df,
    group,
):
    """
    Creates symmetric rho, p and p-FDR matrices for brain variables.
    """
    rho_matrix = pd.DataFrame(
        np.eye(len(BRAIN_VARS)),
        index=BRAIN_VARS,
        columns=BRAIN_VARS,
        dtype=float,
    )

    p_matrix = pd.DataFrame(
        np.nan,
        index=BRAIN_VARS,
        columns=BRAIN_VARS,
        dtype=float,
    )

    p_fdr_matrix = pd.DataFrame(
        np.nan,
        index=BRAIN_VARS,
        columns=BRAIN_VARS,
        dtype=float,
    )

    np.fill_diagonal(
        p_matrix.values,
        0.0,
    )

    np.fill_diagonal(
        p_fdr_matrix.values,
        0.0,
    )

    group_results = results_df.loc[
        results_df["group"] == group
    ]

    for _, result in group_results.iterrows():
        brain_x = result["brain_x"]
        brain_y = result["brain_y"]

        rho_matrix.loc[brain_x, brain_y] = result["rho"]
        rho_matrix.loc[brain_y, brain_x] = result["rho"]

        p_matrix.loc[brain_x, brain_y] = result["p"]
        p_matrix.loc[brain_y, brain_x] = result["p"]

        p_fdr_matrix.loc[brain_x, brain_y] = result["p_fdr"]
        p_fdr_matrix.loc[brain_y, brain_x] = result["p_fdr"]

    return rho_matrix, p_matrix, p_fdr_matrix


def plot_brain_brain_heatmap(
    results_df,
    group,
    output_dir,
):
    """
    Plots the lower half of the brain–brain correlation matrix.

    Each visible cell contains:
        Spearman rho
        Significance stars based on FDR-corrected p-values

    Significance levels:
        *   p_fdr < 0.05
        **  p_fdr < 0.01
        *** p_fdr < 0.001

    Returns:
        rho_matrix
        p_matrix
        p_fdr_matrix
        matrix_values_df
    """
    (
        rho_matrix,
        p_matrix,
        p_fdr_matrix,
    ) = create_symmetric_brain_matrices(
        results_df=results_df,
        group=group,
    )

    display_labels = [
        BRAIN_LABELS.get(variable, variable)
        for variable in BRAIN_VARS
    ]

    rho_plot = rho_matrix.copy()
    p_fdr_plot = p_fdr_matrix.copy()

    rho_plot.index = display_labels
    rho_plot.columns = display_labels

    p_fdr_plot.index = display_labels
    p_fdr_plot.columns = display_labels

    annotations = pd.DataFrame(
        "",
        index=display_labels,
        columns=display_labels,
    )

    # Nur unteres Dreieck ohne Diagonale annotieren
    for row_index in range(len(BRAIN_VARS)):
        for column_index in range(len(BRAIN_VARS)):

            if row_index <= column_index:
                continue

            rho = rho_plot.iloc[
                row_index,
                column_index,
            ]

            p_fdr = p_fdr_plot.iloc[
                row_index,
                column_index,
            ]

            if pd.isna(rho):
                continue

            significance_stars = get_significance_stars(
                p_fdr
            )

            annotations.iloc[
                row_index,
                column_index,
            ] = f"{rho:.2f}{significance_stars}"

    # Oberes Dreieck einschließlich Diagonale ausblenden
    mask = np.triu(
        np.ones_like(
            rho_plot,
            dtype=bool,
        ),
        k=0,
    )

    plt.figure(figsize=(12, 10))

    ax = sns.heatmap(
        rho_plot,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 16},
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Spearman ρ"},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    output_path = os.path.join(output_dir, f"brain_brain_heatmap_{group}_fdr.svg")
    save_and_show_figure(output_path)

    # Übersichtlicher DataFrame mit einer Zeile pro Korrelation
    matrix_values_df = (
        results_df.loc[
            results_df["group"] == group,
            [
                "brain_x",
                "brain_y",
                "rho",
                "p",
                "p_fdr",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )

    matrix_values_df["brain_x_label"] = (
        matrix_values_df["brain_x"].map(BRAIN_LABELS)
        .fillna(matrix_values_df["brain_x"])
    )

    matrix_values_df["brain_y_label"] = (
        matrix_values_df["brain_y"].map(BRAIN_LABELS)
        .fillna(matrix_values_df["brain_y"])
    )

    matrix_values_df["significance"] = (
        matrix_values_df["p_fdr"].apply(
            get_significance_stars
        )
    )

    matrix_values_df = matrix_values_df[
        [
            "brain_x",
            "brain_x_label",
            "brain_y",
            "brain_y_label",
            "rho",
            "p",
            "p_fdr",
            "significance",
        ]
    ]

    return (
        rho_matrix,
        p_matrix,
        p_fdr_matrix,
        matrix_values_df,
    )


def plot_brain_brain_scatter(
    brain_x,
    brain_y,
    results_df,
    group_data,
    output_dir,
):
    """
    Plots a significant brain–brain pair for patients and controls.
    """
    plot_data = []

    for group in GROUPS:
        data = group_data[group][
            [brain_x, brain_y]
        ].copy()

        data["group"] = group
        plot_data.append(data)

    plot_df = pd.concat(
        plot_data,
        ignore_index=True,
    ).dropna(
        subset=[brain_x, brain_y]
    )

    if plot_df.empty:
        return

    plt.figure(figsize=(9, 7))

    ax = sns.scatterplot(
        data=plot_df,
        x=brain_x,
        y=brain_y,
        hue="group",
        hue_order=GROUPS,
        palette=PALETTE,
        s=90,
        alpha=0.7,
    )

    statistics_text = []

    for group in GROUPS:
        group_plot_data = plot_df.loc[
            plot_df["group"] == group
        ]

        result_rows = results_df.loc[
            (results_df["group"] == group)
            & (results_df["brain_x"] == brain_x)
            & (results_df["brain_y"] == brain_y)
        ]

        if result_rows.empty:
            continue

        result = result_rows.iloc[0]

        rho = result["rho"]
        p_value = result["p"]
        p_fdr = result["p_fdr"]
        beta = result["beta"]
        intercept = result["intercept"]
        n = int(result["n"])

        if (
            pd.notna(beta)
            and pd.notna(intercept)
            and len(group_plot_data) > 1
            and group_plot_data[brain_x].nunique() > 1
        ):
            x_line = np.linspace(
                group_plot_data[brain_x].min(),
                group_plot_data[brain_x].max(),
                100,
            )

            y_line = intercept + beta * x_line

            ax.plot(
                x_line,
                y_line,
                color=PALETTE[group],
                linewidth=3,
            )

        statistics_text.append(
            f"{group}: "
            f"ρ = {rho:.2f}, "
            f"p = {format_p_value(p_value)}, "
            f"p-FDR = {format_p_value(p_fdr)}, "
        )

    ax.set_xlabel(
        BRAIN_LABELS.get(
            brain_x,
            brain_x,
        )
    )

    ax.set_ylabel(
        BRAIN_LABELS.get(
            brain_y,
            brain_y,
        )
    )

    ax.set_title("\n".join(statistics_text))

    ax.legend(
        title="Group",
        frameon=False,
    )

    sns.despine()

    output_path = os.path.join(
        output_dir,
        f"brain_brain_{brain_x}_{brain_y}.png",
    )

    save_and_show_figure(output_path)


# --- Main analysis
# Load and prepare data
df_full = load_analysis_data()

group_data = {
    group: df_full.loc[
        df_full["group"] == group
    ].copy()
    for group in GROUPS
}

output_root = os.path.join(
    get_connectivity_path(),
    "brain_correlations",
)

brain_behavior_output_dir = os.path.join(
    output_root,
    "brain_behavior",
)

brain_brain_output_dir = os.path.join(
    output_root,
    "brain_brain",
)

os.makedirs(
    brain_behavior_output_dir,
    exist_ok=True,
)

os.makedirs(
    brain_brain_output_dir,
    exist_ok=True,
)

# Calculate all correlations
brain_behavior_results = (
    calculate_brain_behavior_correlations(
        group_data=group_data,
        brain_vars=BRAIN_VARS,
        behavior_vars=BEHAVIOR_VARS,
    )
)

brain_behavior_results.to_csv(
    os.path.join(
        brain_behavior_output_dir,
        "brain_behavior_correlation_results.csv",
    ),
    index=False,
)

brain_brain_results = (
    calculate_brain_brain_correlations(
        group_data=group_data,
        brain_vars=BRAIN_VARS,
    )
)

brain_brain_results.to_csv(
    os.path.join(
        brain_brain_output_dir,
        "brain_brain_correlation_results.csv",
    ),
    index=False,
)

# Plot heatmaps per group
sns.set_style()
sns.set_context()
for group in GROUPS:
    plot_brain_behavior_heatmap(
        results_df=brain_behavior_results,
        group=group,
        output_dir=brain_behavior_output_dir,
    )

for group in GROUPS:
    plot_brain_brain_heatmap(
        results_df=brain_brain_results,
        group=group,
        output_dir=brain_brain_output_dir,
    )


# Plot scatterplots
significant_brain_behavior_pairs = (
    brain_behavior_results.loc[
        brain_behavior_results["significant_bonf"],
        ["brain", "behavior"],
    ]
    .drop_duplicates()
)

sns.set_theme(
    style="ticks",
    context="talk",
)
for _, pair in significant_brain_behavior_pairs.iterrows():
    plot_brain_behavior_scatter(
        brain_var=pair["brain"],
        behavior_var=pair["behavior"],
        results_df=brain_behavior_results,
        group_data=group_data,
        output_dir=brain_behavior_output_dir,
    )

plot_brain_behavior_scatter(
    brain_var="IPLPFcmL__Vermis_8_median",
    behavior_var="EOfirm_speed",
    results_df=brain_behavior_results,
    group_data=group_data,
    output_dir=brain_behavior_output_dir,
)

brain_brain_pairs = (
    brain_brain_results[
        ["brain_x", "brain_y"]
    ]
    .drop_duplicates()
)

for _, pair in brain_brain_pairs.iterrows():
    plot_brain_brain_scatter(
        brain_x=pair["brain_x"],
        brain_y=pair["brain_y"],
        results_df=brain_brain_results,
        group_data=group_data,
        output_dir=brain_brain_output_dir,
    )

# Console summary
print("\nBrain–behavior correlations")
print("---------------------------")

for group in GROUPS:
    group_results = brain_behavior_results.loc[
        brain_behavior_results["group"] == group
    ]

    n_tests = group_results["p"].notna().sum()
    n_significant = group_results["significant_bonf"].sum()

    print(
        f"{group.capitalize()}: "
        f"{n_significant} of {n_tests} valid tests "
        f"significant after FDR-BH correction."
    )

print("\nBrain–brain correlations")
print("------------------------")

for group in GROUPS:
    group_results = brain_brain_results.loc[
        brain_brain_results["group"] == group
    ]

    n_tests = group_results["p"].notna().sum()
    n_significant = group_results["significant_fdr"].sum()

    print(
        f"{group.capitalize()}: "
        f"{n_significant} of {n_tests} valid tests "
        f"significant after FDR-BH correction."
    )