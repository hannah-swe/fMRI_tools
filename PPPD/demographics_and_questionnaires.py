import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from PPPD import (get_participants_tsv, get_selected_subject_list, get_main_values_tables_path, get_connectivity_path)
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, rankdata, shapiro, spearmanr


def normality_check(
    df,
    variable,
    group_col="group",
    groups=("patient", "control")
):
    results = []

    for group in groups:
        x = df.loc[df[group_col] == group, variable].dropna()

        W, p = shapiro(x)

        print(f"\n{'='*50}")
        print(f"Normality check: {variable} | group = {group}")
        print(f"{'='*50}")
        print(f"n = {len(x)}")
        print(f"Shapiro-Wilk: W = {W:.3f}, p = {p:.4f}")

        if p < 0.05:
            print("Interpretation: the distribution significantly deviates from normality")
        else:
            print("Interpretation: the distribution does not significantly deviate from normality")

        # Density plot
        plt.figure(figsize=(6, 4))
        x.plot(kind="density")
        plt.title(f"Density plot: {variable} ({group})")
        plt.xlabel(variable)
        plt.ylabel("Density")
        plt.show()

        # Q-Q plot
        plt.figure(figsize=(6, 4))
        stats.probplot(x, dist="norm", plot=plt)
        plt.title(f"Q-Q plot: {variable} ({group})")
        plt.show()

        results.append({
            "variable": variable,
            "group": group,
            "n": len(x),
            "W": W,
            "p": p
        })

    return pd.DataFrame(results)


def mwu_report(
    df,
    variable,
    group_col="group",
    group1="control",
    group2="patient"
):
    # Gruppen extrahieren
    g1 = df.loc[df[group_col] == group1, variable].dropna()
    g2 = df.loc[df[group_col] == group2, variable].dropna()

    n1 = len(g1)
    n2 = len(g2)

    if n1 == 0 or n2 == 0:
        raise ValueError("Mindestens eine der beiden Gruppen enthält keine gültigen Werte.")

    # Deskriptive Statistiken der Originalwerte
    descriptive = pd.DataFrame({
        "n": [n1, n2],
        "mean": [g1.mean(), g2.mean()],
        "SD": [g1.std(ddof=1), g2.std(ddof=1)],
        "median": [g1.median(), g2.median()],
        "Q1": [g1.quantile(0.25), g2.quantile(0.25)],
        "Q3": [g1.quantile(0.75), g2.quantile(0.75)]
    }, index=[group1, group2])

    descriptive["IQR"] = descriptive["Q3"] - descriptive["Q1"]

    # Ränge über beide Gruppen gemeinsam berechnen
    combined = pd.concat([
        pd.DataFrame({
            "value": g1.to_numpy(),
            "group": group1
        }),
        pd.DataFrame({
            "value": g2.to_numpy(),
            "group": group2
        })
    ], ignore_index=True)

    # Gleiche Werte erhalten den mittleren Rang
    combined["rank"] = rankdata(combined["value"], method="average")

    rank_statistics = combined.groupby("group")["rank"].agg(
        n="count",
        mean_rank="mean",
        rank_sum="sum"
    ).reindex([group1, group2])

    # Mann–Whitney-U-Test
    u1, p = mannwhitneyu(
        g1,
        g2,
        method="asymptotic",
        alternative="two-sided"
    )

    u2 = n1 * n2 - u1
    U = min(u1, u2)

    # Einfache z-Berechnung ohne Bindungs- und Kontinuitätskorrektur
    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu_u) / sigma_u

    # Effektgröße
    r = abs(z) / np.sqrt(n1 + n2)

    print(f"\n{'=' * 60}")
    print(variable)
    print(f"{'=' * 60}")

    print("\nDeskriptive Statistiken der Originalwerte")
    print(descriptive.round(3))

    print("\nRangstatistiken")
    print(rank_statistics.round(3))

    print("\nMann–Whitney-U-Test")
    print(f"U = {U:.1f}")
    print(f"z = {z:.3f}")
    print(f"p = {p:.4f}")
    print(f"r = {r:.3f}")

    return {
        "variable": variable,
        "descriptive": descriptive,
        "rank_statistics": rank_statistics,
        "U1": u1,
        "U2": u2,
        "U": U,
        "z": z,
        "p": p,
        "r": r,
        "n1": n1,
        "n2": n2
    }


# --- Load full main values table for questionnaire, behavioral and posturography data
df_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")
df = pd.read_csv(df_path)

# --- Count pat and HC
group = df["group"].value_counts()
print(group)
df['Group'] = df['Group'].astype('category')
Group = df['Group'].value_counts()
print(Group)

# --- Age descriptive stats and t-test
descr_age = df.groupby('group')['age_in_years'].describe()
print(descr_age)

patients = df.loc[df['group'] == 'patient', 'age']
controls = df.loc[df['group'] == 'control', 'age']
t, p = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {t:.3f}, p = {p:.3f}")


# --- Disease duration
descr_dod = df.groupby('group')['disease_duration'].describe()
print(descr_dod)

# --- Gender descriptive stats
df.groupby("group")["gender"].value_counts()


# --- Handedness descriptive stats
df.groupby("group")["Händigkeit"].value_counts()


# --- ALQ descriptive stats and mann-whitney-u-test
_ = normality_check(df, "ALQ_total")
_ = mwu_report(df, "ALQ_total")


# --- NPQ descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Niigata_total")
_ = mwu_report(df, "Niigata_total")


# --- HADS_A descriptive stats and mann-whitney-u-test
_ = normality_check(df, "HADS_A_total")
_ = mwu_report(df, "HADS_A_total")


# --- HADS_D descriptive stats and mann-whitney-u-test
_ = normality_check(df, "HADS_D_total")
_ = mwu_report(df, "HADS_D_total")


# --- MSSQ descriptive stats and mann-whitney-u-test
_ = normality_check(df, "MSSQ_raw")
_ = mwu_report(df, "MSSQ_raw")


# --- Neuroticism descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Neo.Skala_n")
_ = mwu_report(df, "Neo.Skala_n")
patients = df.loc[df['group'] == 'patient', 'Neo.Skala_n'].dropna()
controls = df.loc[df['group'] == 'control', 'Neo.Skala_n'].dropna()
result = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {result.statistic:.3f}")
print(f"df = {result.df:.3f}")
print(f"p = {result.pvalue:.3f}")


# --- Extraversion descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Neo.Skala_e")
_ = mwu_report(df, "Neo.Skala_e")
patients = df.loc[df['group'] == 'patient', 'Neo.Skala_e'].dropna()
controls = df.loc[df['group'] == 'control', 'Neo.Skala_e'].dropna()
result = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {result.statistic:.3f}")
print(f"df = {result.df:.3f}")
print(f"p = {result.pvalue:.3f}")


# --- Openness descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Neo.Skala_o")
_ = mwu_report(df, "Neo.Skala_o")
patients = df.loc[df['group'] == 'patient', 'Neo.Skala_o'].dropna()
controls = df.loc[df['group'] == 'control', 'Neo.Skala_o'].dropna()
result = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {result.statistic:.3f}")
print(f"df = {result.df:.3f}")
print(f"p = {result.pvalue:.3f}")


# --- Agreeableness descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Neo.Skala_v")
_ = mwu_report(df, "Neo.Skala_v")
patients = df.loc[df['group'] == 'patient', 'Neo.Skala_v'].dropna()
controls = df.loc[df['group'] == 'control', 'Neo.Skala_v'].dropna()
result = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {result.statistic:.3f}")
print(f"df = {result.df:.3f}")
print(f"p = {result.pvalue:.3f}")


# --- Conscientiousness descriptive stats and mann-whitney-u-test
_ = normality_check(df, "Neo.Skala_g")
_ = mwu_report(df, "Neo.Skala_g")
patients = df.loc[df['group'] == 'patient', 'Neo.Skala_g'].dropna()
controls = df.loc[df['group'] == 'control', 'Neo.Skala_g'].dropna()
result = ttest_ind(patients, controls, equal_var=False)  # Welch-t-Test
print(f"t = {result.statistic:.3f}")
print(f"df = {result.df:.3f}")
print(f"p = {result.pvalue:.3f}")


# --- GVS threshold normality check
_ = normality_check(df, "GVS_threshold_mri")
# --- GVS threshold descriptive stats and mann-whitney-u-test
_ = mwu_report(df, "GVS_threshold_mri")


# --- EO firm sway speed normality check
_ = normality_check(df, "EOfirm_speed")
# --- EO firm sway speed threshold descriptive stats and mann-whitney-u-test
_ = mwu_report(df, "EOfirm_speed")


# --- EO firm sway rating normality check
_ = normality_check(df, "EOfirm_rating")
# --- EO firm sway rating threshold descriptive stats and mann-whitney-u-test
_ = mwu_report(df, "EOfirm_rating")


# --- EC firm sway speed normality check
_ = normality_check(df, "ECfirm_speed")
# --- EC firm sway speed threshold descriptive stats and mann-whitney-u-test
_ = mwu_report(df, "ECfirm_speed")


# --- EC firm sway rating normality check
_ = normality_check(df, "ECfirm_rating")
# --- EC firm sway rating threshold descriptive stats and mann-whitney-u-test
_ = mwu_report(df, "ECfirm_rating")