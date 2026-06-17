import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PPPD import (get_participants_tsv, get_selected_subject_list, get_main_values_tables_path, get_connectivity_path)
from PPPD.subjects import subs, subjects_to_exclude
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm


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

    plt.figure(figsize=(8, 7))
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

# df_full = df_full[df_full["subject_num"] != 122]


# --- Get dataframes split by group
df_pat = df_full[df_full["group"] == "patient"]
df_con = df_full[df_full["group"] == "control"]


# --- Define all correlation analyses
brain_vars = [
    "IPLPFcmL--Vermis_8_median",
    "InsulaIg2L--Lingual_L_median",
    "InsulaIg2L--SupraMarginal_L_median",
    "InsulaOP3RAnat--Cerebellum_Crus2_L_median",
    "OperculumOP1L--Vermis_8_median",
    "OperculumOP1R--Vermis_9_median",
    "V5L--Cerebellum_6_R_median",
    "V5R--Cerebellum_Crus1_L_median",
    "falff--Hippocampus_R_median"
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
    "EOfirm_rating",
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
    "IPLPFcmL--Vermis_8_median": "IPL–Verm",
    "InsulaIg2L--Lingual_L_median": "Ig2-Ling",
    "InsulaIg2L--SupraMarginal_L_median": "Ig2-Sup",
    "InsulaOP3RAnat--Cerebellum_Crus2_L_median": "OP3–Crus2",
    "OperculumOP1L--Vermis_8_median": "OP1L–Verm",
    "OperculumOP1R--Vermis_9_median": "OP1R–Verm",
    "V5L--Cerebellum_6_R_median": "V5L–Crus1",
    "V5R--Cerebellum_Crus1_L_median": "V5R-Crus1",
    "falff--Hippocampus_R_median": "falff-Hippo"
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
    "EOfirm_speed": "Sway-speed",
    "EOfirm_rating": "Sway-rating",
}


# --- Plot heatmap with all correlations
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


# --- Correlate two brain variables
tmp = df_full[["V5L--Cerebellum_6_R_median", "InsulaIg2L--Lingual_L_median"]].dropna()
x = tmp["V5L--Cerebellum_6_R_median"]
y = tmp["InsulaIg2L--Lingual_L_median"]

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
beta = fit.params["V5L--Cerebellum_6_R_median"]
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
plt.savefig("/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/brain_corr_4.svg")
plt.show()