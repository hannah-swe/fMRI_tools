import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PPPD import (get_participants_tsv, get_selected_subject_list, get_main_values_tables_path, get_connectivity_path)
from PPPD.subjects import subs, subjects_to_exclude
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, shapiro
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from itertools import combinations


# Function to run spearman correlation and robust linear model
def run_corr(df, brain_var, behavior_var, min_n=3):
    data = df[[brain_var, behavior_var]].dropna()
    if len(data) < min_n:
        return {
            "brain": brain_var,
            "behavior": behavior_var,
            "rho": np.nan,
            "p": np.nan,
            "beta": np.nan,
            "intercept": np.nan,
            "n": len(data),
            "status": "too_few_valid_cases"
        }
    x = data[behavior_var].astype(float)
    y = data[brain_var].astype(float)
    rho, p = spearmanr(x, y)
    X = sm.add_constant(x, has_constant="add")
    try:
        model = sm.RLM(
            y,
            X,
            M=sm.robust.norms.HuberT()
        )
        fit = model.fit()
        beta = fit.params.iloc[1]
        intercept = fit.params.iloc[0]
    except Exception as e:
        beta = np.nan
        intercept = np.nan

    return {
        "brain": brain_var,
        "behavior": behavior_var,
        "rho": rho,
        "p": p,
        "beta": beta,
        "intercept": intercept,
        "n": len(data),
        "status": "ok"
    }


def plot_corr_heatmap(results_df, group, corr_plot_path):
    corr_plot_dir = os.path.join(corr_plot_path, f"correlation_matrix_{group}")
    df_g = results_df[results_df["group"] == group].copy()
    rho_df = df_g.pivot(index="behavior", columns="brain", values="rho")
    p_df = df_g.pivot(index="behavior", columns="brain", values="p")

    rho_df = rho_df.reindex(index=behavior_vars, columns=brain_vars)
    p_df = p_df.reindex(index=behavior_vars, columns=brain_vars)

    rho_plot = rho_df.rename(index=behavior_labels, columns=brain_labels)
    p_plot = p_df.rename(index=behavior_labels, columns=brain_labels)

    plt.figure(figsize=(9, 8))
    ax = sns.heatmap(
        rho_plot,
        cmap="coolwarm",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        annot=False,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Spearman ρ"}
    )

    # p-Werte manuell einzeichnen mit unterschiedlichem Alpha
    for i in range(p_plot.shape[0]):
        for j in range(p_plot.shape[1]):
            p = p_plot.iloc[i, j]
            if pd.isna(p):
                continue

            alpha = 1.0 if p < 0.05 else 0.25
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{p:.3f}",
                ha="center",
                va="center",
                color="black",
                alpha=alpha,
                fontsize=10
            )
    ax.set_title(f"{group}: Spearman correlations")
    ax.set_xlabel("Functional connectivity")
    ax.set_ylabel("Behavior variable")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(corr_plot_dir, dpi=300, bbox_inches="tight")
    plt.show()

    return rho_df, p_df


# Function to set up plots
def plot_corr_from_results(brain_var, behavior_var, results_df, corr_plot_path):
    corr_plot_dir = os.path.join(corr_plot_path, f"correlation_{brain_var}_{behavior_var}")
    # get raw data for scatterplot
    data_pat = df_pat[[brain_var, behavior_var]].copy()
    data_pat["group"] = "patient"
    data_con = df_con[[brain_var, behavior_var]].copy()
    data_con["group"] = "control"
    data = pd.concat([data_con, data_pat], axis=0).dropna()

    # scatterplot
    plt.figure(figsize=(9, 7))
    ax = sns.scatterplot(
        data=data,
        x=behavior_var,
        y=brain_var,
        hue="group",
        palette=palette,
        s=90,
        alpha=0.7,
        legend=False,
    )
    # get stats and regression line
    stats_text = []
    for group in ["control", "patient"]:
        group_data = data[data["group"] == group]
        row = results_df[
            (results_df["brain"] == brain_var) &
            (results_df["behavior"] == behavior_var) &
            (results_df["group"] == group)
        ]
        if row.empty:
            continue
        row = row.iloc[0]
        beta = row["beta"]
        intercept = row["intercept"]
        rho = row["rho"]
        p = row["p"]
        # if no regression is given
        if pd.notna(beta) and pd.notna(intercept) and len(group_data) > 0:
            x_line = np.linspace(
                group_data[behavior_var].min(),
                group_data[behavior_var].max(),
                100
            )
            y_line = intercept + beta * x_line
            ax.plot(
                x_line,
                y_line,
                color=palette[group],
                linewidth=3
            )
        stats_text.append(f"{group}: ρ = {rho:.2f}, p = {p:.3f}")
    ax.set_title(
        f"\n".join(stats_text)
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(corr_plot_dir, dpi=300, bbox_inches="tight")
    plt.show()


def run_brain_corr_matrix(df, brain_vars, group, min_n=3):
    """
    Berechnet alle eindeutigen paarweisen Spearman-Korrelationen zwischen
    den angegebenen Brain-Variablen.

    Zusätzlich wird ein robustes lineares Modell berechnet:
        brain_var_y ~ brain_var_x
    """
    results = []

    for brain_var_x, brain_var_y in combinations(brain_vars, 2):
        data = df[[brain_var_x, brain_var_y]].dropna()

        result = {
            "brain_x": brain_var_x,
            "brain_y": brain_var_y,
            "group": group,
            "rho": np.nan,
            "p": np.nan,
            "beta": np.nan,
            "intercept": np.nan,
            "n": len(data),
            "status": "too_few_valid_cases"
        }

        if len(data) < min_n:
            results.append(result)
            continue

        x = data[brain_var_x].astype(float)
        y = data[brain_var_y].astype(float)

        rho, p = spearmanr(x, y)

        result["rho"] = rho
        result["p"] = p
        result["status"] = "ok"

        X = sm.add_constant(x, has_constant="add")

        try:
            model = sm.RLM(
                y,
                X,
                M=sm.robust.norms.HuberT()
            )
            fit = model.fit()

            result["intercept"] = fit.params.iloc[0]
            result["beta"] = fit.params.iloc[1]

        except Exception:
            result["status"] = "correlation_ok_rlm_failed"

        results.append(result)

    return pd.DataFrame(results)


def plot_brain_corr_heatmap(
    results_df,
    brain_vars,
    brain_labels,
    group,
    corr_plot_path,
    significance_level=0.05,
    p_column="p_fdr"
):
    """
    Erstellt eine vollständige symmetrische Brain-Korrelationsmatrix.

    p_column:
        "p"     = unkorrigierte p-Werte
        "p_fdr" = Benjamini-Hochberg-korrigierte p-Werte
    """

    os.makedirs(corr_plot_path, exist_ok=True)

    group_results = results_df[
        results_df["group"] == group
    ].copy()

    rho_matrix = pd.DataFrame(
        np.eye(len(brain_vars)),
        index=brain_vars,
        columns=brain_vars,
        dtype=float
    )

    p_matrix = pd.DataFrame(
        np.nan,
        index=brain_vars,
        columns=brain_vars,
        dtype=float
    )

    np.fill_diagonal(p_matrix.values, 0.0)

    for _, row in group_results.iterrows():
        brain_x = row["brain_x"]
        brain_y = row["brain_y"]

        rho_matrix.loc[brain_x, brain_y] = row["rho"]
        rho_matrix.loc[brain_y, brain_x] = row["rho"]

        p_matrix.loc[brain_x, brain_y] = row[p_column]
        p_matrix.loc[brain_y, brain_x] = row[p_column]

    display_labels = [
        brain_labels.get(variable, variable)
        for variable in brain_vars
    ]

    rho_plot = rho_matrix.copy()
    p_plot = p_matrix.copy()

    rho_plot.index = display_labels
    rho_plot.columns = display_labels
    p_plot.index = display_labels
    p_plot.columns = display_labels

    annotations = pd.DataFrame(
        "",
        index=rho_plot.index,
        columns=rho_plot.columns
    )

    for i in range(len(brain_vars)):
        for j in range(len(brain_vars)):
            rho = rho_plot.iloc[i, j]
            p_value = p_plot.iloc[i, j]

            if pd.isna(rho):
                continue

            if i == j:
                annotations.iloc[i, j] = "1.00"
            else:
                star = (
                    "*"
                    if pd.notna(p_value)
                    and p_value < significance_level
                    else ""
                )

                annotations.iloc[i, j] = (
                    f"{rho:.2f}{star}\n"
                    f"p={p_value:.3f}"
                )

    plt.figure(figsize=(12, 10))

    ax = sns.heatmap(
        rho_plot,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Spearman ρ"}
    )

    correction_label = (
        "FDR-BH corrected"
        if p_column == "p_fdr"
        else "uncorrected"
    )

    ax.set_title(
        f"{group}: correlations between brain variables\n"
        f"* {correction_label} p < {significance_level}"
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(
        corr_plot_path,
        f"brain_correlation_matrix_{group}_{p_column}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    return rho_matrix, p_matrix


def plot_brain_corr_from_results(
    brain_var_x,
    brain_var_y,
    results_df,
    df_pat,
    df_con,
    brain_labels,
    corr_plot_path,
    palette
):
    """
    Erstellt einen gemeinsamen Scatterplot für Patienten und Kontrollen.
    Die robusten Regressionslinien werden aus results_df übernommen.
    """

    os.makedirs(corr_plot_path, exist_ok=True)

    data_pat = df_pat[[brain_var_x, brain_var_y]].copy()
    data_pat["group"] = "patient"

    data_con = df_con[[brain_var_x, brain_var_y]].copy()
    data_con["group"] = "control"

    data = pd.concat(
        [data_con, data_pat],
        axis=0,
        ignore_index=True
    ).dropna(subset=[brain_var_x, brain_var_y])

    if data.empty:
        return

    plt.figure(figsize=(9, 7))

    ax = sns.scatterplot(
        data=data,
        x=brain_var_x,
        y=brain_var_y,
        hue="group",
        palette=palette,
        s=90,
        alpha=0.7
    )

    stats_text = []

    for group in ["control", "patient"]:
        group_data = data[data["group"] == group]

        result_row = results_df[
            (results_df["brain_x"] == brain_var_x) &
            (results_df["brain_y"] == brain_var_y) &
            (results_df["group"] == group)
        ]

        if result_row.empty:
            continue

        result_row = result_row.iloc[0]

        beta = result_row["beta"]
        intercept = result_row["intercept"]
        rho = result_row["rho"]
        p = result_row["p"]
        p_fdr = result_row["p_fdr"]
        n = result_row["n"]

        if (
            pd.notna(beta)
            and pd.notna(intercept)
            and len(group_data) > 1
            and group_data[brain_var_x].nunique() > 1
        ):
            x_line = np.linspace(
                group_data[brain_var_x].min(),
                group_data[brain_var_x].max(),
                100
            )

            y_line = intercept + beta * x_line

            ax.plot(
                x_line,
                y_line,
                color=palette[group],
                linewidth=3
            )

        if pd.notna(rho) and pd.notna(p):
            stats_text.append(
                f"{group}: ρ = {rho:.2f}, "
                f"p = {p:.3f}, "
                f"p-FDR = {p_fdr:.3f}, "
                f"n = {n}"
            )

    x_label = brain_labels.get(brain_var_x, brain_var_x)
    y_label = brain_labels.get(brain_var_y, brain_var_y)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("\n".join(stats_text))

    ax.legend(
        title="Group",
        frameon=False
    )

    sns.despine()
    plt.tight_layout()

    output_path = os.path.join(
        corr_plot_path,
        f"brain_correlation_{brain_var_x}_{brain_var_y}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def add_fdr_correction(
    results_df,
    p_column="p",
    group_column="group",
    alpha=0.05
):
    """
    Führt eine Benjamini-Hochberg-FDR-Korrektur getrennt pro Gruppe durch.

    Neue Spalten:
        p_fdr: FDR-korrigierter p-Wert
        significant_fdr: True, wenn p_fdr < alpha
    """
    results_df = results_df.copy()

    results_df["p_fdr"] = np.nan
    results_df["significant_fdr"] = False

    for group, group_df in results_df.groupby(group_column):
        valid_mask = group_df[p_column].notna()
        valid_indices = group_df.index[valid_mask]

        if len(valid_indices) == 0:
            continue

        reject, p_corrected, _, _ = multipletests(
            results_df.loc[valid_indices, p_column],
            alpha=alpha,
            method="fdr_bh"
        )

        results_df.loc[valid_indices, "p_fdr"] = p_corrected
        results_df.loc[valid_indices, "significant_fdr"] = reject

    return results_df


# --- Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seeds = ["InsulaOP3RAnat", "IPLPFcmL"] # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII"
                        # for feature = "falff" or "alff" do seed = None
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
pre_post_diff = False
direction = "negative" # possible directions for pre-post differences:
                        # "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Get connectivity dataframe with subject values for all significant cluster per seed
connectivity_df_path = os.path.join(get_connectivity_path(), "connectivity_pre-data_dataframe_wide_format.csv")
connectivity_df = pd.read_csv(connectivity_df_path)


# --- Get falff dataframe with subject values for significant cluster
falff_df_path = os.path.join(get_connectivity_path(), "falff_pre-data_dataframe_wide_format.csv")
falff_df = pd.read_csv(falff_df_path)


# --- Load full main values table for questionnaire, behavioral and posturography data
main_df_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")
main_df = pd.read_csv(main_df_path)

corr_plot_path = os.path.join(get_connectivity_path(), "brain_behav_corr")


# --- Merge connectivity and main values dataframe
assert main_df["subject_num"].is_unique
assert connectivity_df["subject_num"].is_unique
assert falff_df["subject_num"].is_unique
df_full = main_df.merge(
    connectivity_df,
    on="subject_num",
    how="left",
    suffixes=("_main", "_conn")
)
# quality check: ensure group labels are identical
if not (df_full["group_main"] == df_full["group_conn"]).all():
    mismatches = df_full.loc[
        df_full["group_main"] != df_full["group_conn"],
        ["subject_num", "group_main", "group_conn"]
    ]
    raise ValueError(
        f"Mismatch found in group labels:\n{mismatches}"
    )
# clean up columns
df_full = df_full.drop(columns=["group_conn", "subject_id"])
df_full = df_full.rename(columns={"group_main": "group"})

df_full = df_full.merge(
    falff_df,
    on="subject_num",
    how="left",
    suffixes=("_main", "_falff")
)
# quality check: ensure group labels are identical
if not (df_full["group_main"] == df_full["group_falff"]).all():
    mismatches = df_full.loc[
        df_full["group_main"] != df_full["group_falff"],
        ["subject_num", "group_main", "group_falff"]
    ]
    raise ValueError(
        f"Mismatch found in group labels:\n{mismatches}"
    )
# clean up columns
df_full = df_full.drop(columns=["group_falff", "subject_id"])
df_full = df_full.rename(columns={"group_main": "group"})

# df_full = df_full[df_full["subject_num"] != 40]


# --- Get dataframes split by group
df_pat = df_full[df_full["group"] == "patient"]
df_con = df_full[df_full["group"] == "control"]


# --- Define all correlation analyses
brain_vars = [
    "IPLPFcmL__Vermis_8_median",
    "InsulaIg2L__Lingual_L_median",
    "InsulaIg2L__SupraMarginal_L_median",
    "InsulaOP3RAnat__Cerebellum_Crus2_L_median",
    "OperculumOP1L__Vermis_8_median",
    "OperculumOP1R__Vermis_9_median",
    "V5L__Cerebellum_6_R_median",
    "V5R__Cerebellum_Crus1_L_median",
    "falff__Hippocampus_R_median"
]

behavior_vars = [
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


# --- Correlate all predefined analyses
results = []
for brain_var in brain_vars:
    for behavior_var in behavior_vars:
        # patients
        res_pat = run_corr(
            df_pat,
            brain_var,
            behavior_var
        )
        res_pat["group"] = "patient"
        results.append(res_pat)

        # controls
        res_con = run_corr(
            df_con,
            brain_var,
            behavior_var
        )
        res_con["group"] = "control"
        results.append(res_con)

# one results dataframe
results_df = pd.DataFrame(results)


# --- New labels for all variables
brain_labels = {
    "IPLPFcmL__Vermis_8_median": "IPL–Verm",
    "InsulaIg2L__Lingual_L_median": "Ig2-Ling",
    "InsulaIg2L__SupraMarginal_L_median": "Ig2-Sup",
    "InsulaOP3RAnat__Cerebellum_Crus2_L_median": "OP3–Crus2",
    "OperculumOP1L__Vermis_8_median": "OP1L–Verm",
    "OperculumOP1R__Vermis_9_median": "OP1R–Verm",
    "V5L__Cerebellum_6_R_median": "V5L–Crus1",
    "V5R__Cerebellum_Crus1_L_median": "V5R-Crus1",
    "falff__Hippocampus_R_median": "falff-Hippo"
}
behavior_labels = {
    "age": "Age",
    "disease_duration": "DoD",
    "GVS_threshold_mri": "GVS-thresh",
    "ALQ_total": "ALQ",
    "Niigata_total": "Niigata",
    "MSSQ_raw": "MSSQ",
    "HADS_A_total": "HADS-A",
    "HADS_D_total": "HADS-D",
    "Neo.Skala_n": "Neo_N",
    "EOfirm_speed": "Sway-speed EO",
    "EOfirm_rating": "Sway-rating EO",
    "ECfirm_speed": "Sway-speed EC",
    "ECfirm_rating": "Sway-rating EC",
}


# --- Plot heatmap with all correlations
sns.set_theme()
sns.set_context()
rho_pat, p_pat = plot_corr_heatmap(results_df, "patient", corr_plot_path)
rho_con, p_con = plot_corr_heatmap(results_df, "control", corr_plot_path)


# --- Plot correlation p < 0.1
# filter for interesting results
sns.set_theme(style="ticks")
sns.set_context("talk")
palette = {"control": "teal", "patient": "hotpink"}
sig_results = results_df[results_df["p"] < 0.05].copy()
sig_pairs = sig_results[["brain", "behavior"]].drop_duplicates()
for _, row in sig_pairs.iterrows():
    plot_corr_from_results(
        row["brain"],
        row["behavior"],
        results_df,
        corr_plot_path,
    )


# Brain–Brain correlations
brain_corr_plot_path = os.path.join(
    corr_plot_path,
    "brain_brain_correlations"
)

os.makedirs(brain_corr_plot_path, exist_ok=True)


# --- Alle Brain–Brain-Korrelationen berechnen

brain_results_pat = run_brain_corr_matrix(
    df=df_pat,
    brain_vars=brain_vars,
    group="patient",
    min_n=3
)

brain_results_con = run_brain_corr_matrix(
    df=df_con,
    brain_vars=brain_vars,
    group="control",
    min_n=3
)

brain_results_df = pd.concat(
    [brain_results_pat, brain_results_con],
    axis=0,
    ignore_index=True
)

brain_results_df = add_fdr_correction(
    results_df=brain_results_df,
    p_column="p",
    group_column="group",
    alpha=0.05
)

# --- Ergebnistabelle speichern

brain_results_path = os.path.join(
    brain_corr_plot_path,
    "brain_brain_correlation_results.csv"
)

brain_results_df.to_csv(
    brain_results_path,
    index=False
)


# --- Heatmaps erstellen
sns.set_theme()
sns.set_context()
rho_brain_pat, p_fdr_brain_pat = plot_brain_corr_heatmap(
    results_df=brain_results_df,
    brain_vars=brain_vars,
    brain_labels=brain_labels,
    group="patient",
    corr_plot_path=brain_corr_plot_path,
    significance_level=0.05,
    p_column="p_fdr"
)

rho_brain_con, p_fdr_brain_con = plot_brain_corr_heatmap(
    results_df=brain_results_df,
    brain_vars=brain_vars,
    brain_labels=brain_labels,
    group="control",
    corr_plot_path=brain_corr_plot_path,
    significance_level=0.05,
    p_column="p_fdr"
)


# --- Alle Paare auswählen, die in mindestens einer Gruppe p < 0.05 sind
significant_brain_results = brain_results_df[
    brain_results_df["p_fdr"] < 0.05
].copy()

significant_brain_pairs = significant_brain_results[
    ["brain_x", "brain_y"]
].drop_duplicates()


# --- Scatterplots der signifikanten Paare
sns.set_theme(style="ticks")
sns.set_context("talk")
palette = {"control": "teal", "patient": "hotpink"}
for _, row in significant_brain_pairs.iterrows():
    plot_brain_corr_from_results(
        brain_var_x=row["brain_x"],
        brain_var_y=row["brain_y"],
        results_df=brain_results_df,
        df_pat=df_pat,
        df_con=df_con,
        brain_labels=brain_labels,
        corr_plot_path=brain_corr_plot_path,
        palette=palette
    )


# --- Correlate two brain variables
tmp = df_full[[
    "subject_num",
    "group",
    "age",
    "ALQ_total",
    "Niigata_total",
    "InsulaIg2L__Lingual_L_median",
    "IPLPFcmL__Vermis_8_median",
    "OperculumOP1L__Vermis_8_median",
    "OperculumOP1R__Vermis_9_median",
]]

tmp_pat = tmp[tmp["group"] == "patient"]
tmp_con = tmp[tmp["group"] == "control"]

x = tmp_pat["InsulaIg2L__Lingual_L_median"]
y = tmp_pat["IPLPFcmL__Vermis_8_median"]

rho, p = spearmanr(x, y)
print(f"Spearman rho: {rho:.3f}")
print(f"p-Wert: {p:.4f}")

X = sm.add_constant(x)
model = sm.RLM(
    y,
    X,
    M=sm.robust.norms.HuberT()
)
fit = model.fit()

intercept = fit.params["const"]
beta = fit.params["InsulaIg2L__Lingual_L_median"]
print(f"Intercept: {intercept:.3f}")
print(f"Beta: {beta:.3f}")

plt.figure(figsize=(8, 7))
ax = sns.scatterplot(
    data=df_full,
    x=x,
    y=y,
    hue="group",
    palette=palette,
    s=90,
    alpha=0.7,
    legend=False,
)
x_line = np.linspace(
    x.min(),
    x.max(),
    100
)
y_line = intercept + beta * x_line
ax.plot(
    x_line,
    y_line,
    color="black",
    linewidth=2
)
ax.set_title(f"ρ = {rho:.2f}, p = {p:.4f}")
sns.despine()
plt.tight_layout()
# plt.savefig("/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/brain_corr_4.svg")
plt.show()


model = smf.ols(
    formula="""
    Q("V5L__Cerebellum_6_R_median") ~
    Q("InsulaIg2L__SupraMarginal_L_median") +
    C(group) +
    Q("InsulaIg2L__SupraMarginal_L_median"):C(group)
    """,
    data=tmp
).fit()

print(model.summary())


# --- Look up distributions
dist_vars = [
    "InsulaIg2L__Lingual_L_median",
    "InsulaIg2L__SupraMarginal_L_median",
    "V5L__Cerebellum_6_R_median",
    "V5R__Cerebellum_Crus1_L_median"
]

for d in dist_vars:
    stat, p = shapiro(df_full[d].dropna())
    print(f"shapiro: {stat:.3f}, p = {p:.4f}")
    sns.displot(df_full, x=d, kind="kde")
    plt.show()


# --- Correlate two brain variables
tmp = df_full[[
    "subject_num",
    "group",
    "age",
    "ALQ_total",
    "Niigata_total",
    "InsulaIg2L__SupraMarginal_L_median",
    "IPLPFcmL__Vermis_8_median",
    "OperculumOP1L__Vermis_8_median",
    "OperculumOP1R__Vermis_9_median",
]].copy()

tmp_pat = tmp[tmp["group"] == "patient"].copy()
tmp_con = tmp[tmp["group"] == "control"].copy()

x_col = "InsulaIg2L__SupraMarginal_L_median"
y_col = "OperculumOP1R__Vermis_9_median"

# --- Spearman correlations
rho_all, p_all = spearmanr(tmp[x_col], tmp[y_col], nan_policy="omit")
rho_pat, p_pat = spearmanr(tmp_pat[x_col], tmp_pat[y_col], nan_policy="omit")
rho_con, p_con = spearmanr(tmp_con[x_col], tmp_con[y_col], nan_policy="omit")

print(f"All:     Spearman rho = {rho_all:.3f}, p = {p_all:.4f}")
print(f"Patient: Spearman rho = {rho_pat:.3f}, p = {p_pat:.4f}")
print(f"Control: Spearman rho = {rho_con:.3f}, p = {p_con:.4f}")


# --- Function for robust regression line
def plot_rlm_line(ax, data, x_col, y_col, color, label, linestyle="-"):
    data = data[[x_col, y_col]].dropna()

    x = data[x_col]
    y = data[y_col]

    X = sm.add_constant(x)
    model = sm.RLM(
        y,
        X,
        M=sm.robust.norms.HuberT()
    )
    fit = model.fit()

    intercept = fit.params["const"]
    beta = fit.params[x_col]

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + beta * x_line

    ax.plot(
        x_line,
        y_line,
        color=color,
        linewidth=2,
        linestyle=linestyle,
        label=None,
    )

    return intercept, beta


# --- Plot
plt.figure(figsize=(8, 7))

ax = sns.scatterplot(
    data=tmp,
    x=x_col,
    y=y_col,
    hue="group",
    palette=palette,
    s=90,
    alpha=0.7,
    legend=False,
)

# Robust regression lines
intercept_pat, beta_pat = plot_rlm_line(
    ax, tmp_pat, x_col, y_col,
    color=palette["patient"],
    label="Patients",
    linestyle="-"
)

intercept_con, beta_con = plot_rlm_line(
    ax, tmp_con, x_col, y_col,
    color=palette["control"],
    label="Controls",
    linestyle="-"
)

intercept_all, beta_all = plot_rlm_line(
    ax, tmp, x_col, y_col,
    color="black",
    label="All",
    linestyle="-"
)

print(f"All:     Intercept = {intercept_all:.3f}, Beta = {beta_all:.3f}")
print(f"Patient: Intercept = {intercept_pat:.3f}, Beta = {beta_pat:.3f}")
print(f"Control: Intercept = {intercept_con:.3f}, Beta = {beta_con:.3f}")

# --- Textboxes
text_all = f"All\nρ = {rho_all:.2f}\np = {p_all:.4f}"
text_pat = f"Patients\nρ = {rho_pat:.2f}\np = {p_pat:.4f}"
text_con = f"Controls\nρ = {rho_con:.2f}\np = {p_con:.4f}"

ax.text(
    0.50, 0.98,
    text_all,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax.text(
    0.70, 0.98,
    text_pat,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    color=palette["patient"],
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax.text(
    0.90, 0.98,
    text_con,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    color=palette["control"],
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
sns.despine()
plt.tight_layout()
# plt.savefig("/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/brain_corr_V5r_ling.svg")
plt.show()
