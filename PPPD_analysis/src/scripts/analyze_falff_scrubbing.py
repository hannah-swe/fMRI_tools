from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import (
    pearsonr,
    spearmanr,
    ttest_rel,
)


# =========================================================
# CONFIG
# =========================================================

FEATURE_TABLE = (
    Path.cwd()
    / "outputs"
    / "qc_falff_table.csv"
)

OUT_DIR = (
    Path.cwd()
    / "outputs"
    / "plots"
    / "falff"
    / "scrubbing_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TISSUE_COLS = {
    "DMN": "falff_dmn_mean",
    "GM": "falff_gm_mean",
    "WM": "falff_wm_mean",
    "CSF": "falff_csf_mean",
}


# Motion measures from HALFpipe reportvals
FD_VARS = [
    "fd_mean_halfpipe",
    "fd_max_halfpipe",
    "fd_perc_halfpipe",
]


# Actual scrubbing-related measures derived
# from fMRIPrep confounds.
SCRUBBING_VARS = [
    "n_motion_outliers",
    "motion_outlier_perc",
    "n_fd_outliers",
    "n_dvars_outliers",
    "n_fd_only_outliers",
    "n_dvars_only_outliers",
    "n_fd_and_dvars_outliers",
]


sns.set_style("ticks")
sns.set_context()


# =========================================================
# HELPERS
# =========================================================

def create_paired_feature_table(
    df: pd.DataFrame,
    outcome_col: str,
) -> pd.DataFrame:
    """
    Create one row per subject with scrub, no_scrub,
    signed difference, and absolute difference.
    """
    wide = (
        df.pivot(
            index="sub",
            columns="pipeline",
            values=outcome_col,
        )
        .dropna(
            subset=[
                "no_scrub",
                "scrub",
            ]
        )
        .copy()
    )

    wide["difference"] = (
        wide["scrub"]
        - wide["no_scrub"]
    )

    wide["abs_difference"] = (
        wide["difference"].abs()
    )

    wide["relative_difference"] = (
        (
            wide["scrub"]
            - wide["no_scrub"]
        )
        / wide["no_scrub"]
        * 100
    )

    return wide


def add_subject_qc_metrics(
    wide: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add subject-level motion and scrubbing metrics.
    """
    available_vars = [
        col
        for col in (
            FD_VARS
            + SCRUBBING_VARS
        )
        if col in df.columns
    ]

    subject_qc = (
        df.groupby("sub")[
            available_vars
        ]
        .first()
    )

    return wide.join(
        subject_qc
    )


def compute_correlations(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    tissue: str,
) -> list[dict]:
    """
    Compute Pearson and Spearman correlations.
    """
    rows = []

    for predictor in predictors:

        if predictor not in data.columns:
            continue

        tmp = (
            data[
                [
                    outcome,
                    predictor,
                ]
            ]
            .dropna()
        )

        if len(tmp) < 3:
            continue

        if (
            tmp[outcome].nunique() < 2
            or tmp[predictor].nunique() < 2
        ):
            continue

        r, p = pearsonr(
            tmp[outcome],
            tmp[predictor],
        )

        rho, p_s = spearmanr(
            tmp[outcome],
            tmp[predictor],
        )

        rows.append(
            {
                "tissue": tissue,
                "outcome": outcome,
                "predictor": predictor,
                "n": len(tmp),
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": p_s,
            }
        )

    return rows


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # 1. Load fALFF feature table
    # -----------------------------------------------------
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: "
            f"{FEATURE_TABLE}"
        )

    df = pd.read_csv(
        FEATURE_TABLE
    )

    print(
        "\nLoaded:",
        FEATURE_TABLE,
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Subjects:",
        df["sub"].nunique(),
    )


    # -----------------------------------------------------
    # 2. Check required fALFF columns
    # -----------------------------------------------------
    missing_tissue_cols = [
        col
        for col in TISSUE_COLS.values()
        if col not in df.columns
    ]

    if missing_tissue_cols:
        raise KeyError(
            "Missing fALFF tissue columns: "
            f"{missing_tissue_cols}"
        )


    # -----------------------------------------------------
    # 3. Analyze each tissue
    # -----------------------------------------------------
    correlation_results = []
    tissue_diff_tables = []

    predictors = [
        col
        for col in (
            FD_VARS
            + SCRUBBING_VARS
        )
        if col in df.columns
    ]

    print(
        "\nAvailable motion/scrubbing predictors:"
    )

    for predictor in predictors:
        print(
            " ",
            predictor,
        )


    for tissue_name, tissue_col in TISSUE_COLS.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"{tissue_name}: "
            f"{tissue_col}"
        )

        print(
            "=" * 70
        )

        tissue_wide = (
            create_paired_feature_table(
                df=df,
                outcome_col=tissue_col,
            )
        )

        tissue_wide = (
            add_subject_qc_metrics(
                wide=tissue_wide,
                df=df,
            )
        )

        print(
            "Paired subjects:",
            len(tissue_wide),
        )

        print(
            "\nDifference scores:"
        )

        print(
            f"Mean   = "
            f"{tissue_wide['difference'].mean():.6f}"
        )

        print(
            f"Median = "
            f"{tissue_wide['difference'].median():.6f}"
        )

        print(
            f"SD     = "
            f"{tissue_wide['difference'].std():.6f}"
        )

        print(
            f"Min    = "
            f"{tissue_wide['difference'].min():.6f}"
        )

        print(
            f"Max    = "
            f"{tissue_wide['difference'].max():.6f}"
        )

        print(
            f"Mean relative difference = "
            f"{tissue_wide['relative_difference'].mean():.2f}%"
        )

        print(
            f"Median relative difference = "
            f"{tissue_wide['relative_difference'].median():.2f}%"
        )


        # -------------------------------------------------
        # Paired t-test
        # -------------------------------------------------
        t_stat, p_value = ttest_rel(
            tissue_wide["scrub"],
            tissue_wide["no_scrub"],
        )

        print(
            "\nPaired t-test:"
        )

        print(
            f"t = {t_stat:.3f}"
        )

        print(
            f"p = {p_value:.6g}"
        )


        # -------------------------------------------------
        # Correlations
        # -------------------------------------------------
        for outcome in [
            "difference",
            "abs_difference",
            "relative_difference",
        ]:

            results = compute_correlations(
                data=tissue_wide,
                outcome=outcome,
                predictors=predictors,
                tissue=tissue_name,
            )

            correlation_results.extend(
                results
            )


        # -------------------------------------------------
        # Store differences
        # -------------------------------------------------
        tissue_tmp = (
            tissue_wide
            .reset_index()[
                [
                    "sub",
                    "no_scrub",
                    "scrub",
                    "difference",
                    "abs_difference",
                    "relative_difference",
                ]
            ]
            .copy()
        )

        tissue_tmp["tissue"] = (
            tissue_name
        )

        tissue_diff_tables.append(
            tissue_tmp
        )


    # -----------------------------------------------------
    # 4. Combine tissue difference data
    # -----------------------------------------------------
    tissue_diff_df = pd.concat(
        tissue_diff_tables,
        ignore_index=True,
    )


    # -----------------------------------------------------
    # 5. Save correlation table
    # -----------------------------------------------------
    correlation_df = pd.DataFrame(
        correlation_results
    )

    correlation_path = (
        OUT_DIR
        / "falff_scrubbing_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    print(
        "\nSaved correlation results:",
        correlation_path,
    )


    # -----------------------------------------------------
    # 6. Useful console summary:
    #    signed ΔfALFF correlations
    # -----------------------------------------------------
    signed_results = (
        correlation_df[
            correlation_df["outcome"]
            == "difference"
        ]
        .sort_values(
            [
                "tissue",
                "predictor",
            ]
        )
    )

    print(
        "\n=== Signed ΔfALFF correlations ==="
    )

    print(
        signed_results[
            [
                "tissue",
                "predictor",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        ].to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # 7. Signed differences by tissue
    # -----------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.boxplot(
        data=tissue_diff_df,
        x="tissue",
        y="difference",
        showfliers=False,
        fill=False,
        linewidth=1.5,
        ax=ax,
    )

    sns.stripplot(
        data=tissue_diff_df,
        x="tissue",
        y="difference",
        jitter=True,
        alpha=0.4,
        size=4,
        ax=ax,
    )

    ax.axhline(
        0,
        color="grey",
        linewidth=1.5,
        alpha=0.5,
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "Δ fALFF (scrub − no scrub)"
    )

    sns.despine(
        ax=ax
    )

    fig.tight_layout()

    path = (
        OUT_DIR
        / "falff_tissue_signed_difference.svg"
    )

    fig.savefig(
        path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()


    # -----------------------------------------------------
    # 8. Absolute differences by tissue
    # -----------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.boxplot(
        data=tissue_diff_df,
        x="tissue",
        y="abs_difference",
        showfliers=False,
        fill=False,
        linewidth=1.5,
        ax=ax,
    )

    sns.stripplot(
        data=tissue_diff_df,
        x="tissue",
        y="abs_difference",
        jitter=True,
        alpha=0.4,
        size=4,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "|Δ fALFF|"
    )

    sns.despine(
        ax=ax
    )

    fig.tight_layout()

    path = (
        OUT_DIR
        / "falff_tissue_absolute_difference.svg"
    )

    fig.savefig(
        path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()


    # -----------------------------------------------------
    # 9. Relative differences by tissue
    # -----------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.boxplot(
        data=tissue_diff_df,
        x="tissue",
        y="relative_difference",
        showfliers=False,
        fill=False,
        linewidth=1.5,
        ax=ax,
    )

    sns.stripplot(
        data=tissue_diff_df,
        x="tissue",
        y="relative_difference",
        jitter=True,
        alpha=0.4,
        size=4,
        ax=ax,
    )

    ax.axhline(
        0,
        color="grey",
        linewidth=1,
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "Relative Δ fALFF (%)"
    )

    sns.despine(
        ax=ax
    )

    fig.tight_layout()

    path = (
        OUT_DIR
        / "falff_tissue_relative_difference.svg"
    )

    fig.savefig(
        path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()


    # -----------------------------------------------------
    # 10. Correlation of tissue-specific differences
    # -----------------------------------------------------
    tissue_wide_diff = (
        tissue_diff_df
        .pivot(
            index="sub",
            columns="tissue",
            values="difference",
        )
    )

    print(
        "\n=== Spearman correlations between "
        "tissue ΔfALFF values ==="
    )

    print(
        tissue_wide_diff.corr(
            method="spearman"
        )
    )


    # -----------------------------------------------------
    # 11. CSF vs WM difference
    # -----------------------------------------------------
    if {
        "CSF",
        "WM",
    }.issubset(
        tissue_wide_diff.columns
    ):

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        sns.regplot(
            data=tissue_wide_diff,
            x="CSF",
            y="WM",
            ax=ax,
        )

        ax.axhline(
            0,
            color="grey",
            linewidth=1,
        )

        ax.axvline(
            0,
            color="grey",
            linewidth=1,
        )

        ax.set_xlabel(
            "Δ fALFF CSF"
        )

        ax.set_ylabel(
            "Δ fALFF WM"
        )

        sns.despine(
            ax=ax
        )

        fig.tight_layout()

        path = (
            OUT_DIR
            / "falff_delta_csf_vs_wm.svg"
        )

        fig.savefig(
            path,
            bbox_inches="tight",
            transparent=True,
        )

        plt.show()


    # -----------------------------------------------------
    # 12. Save tissue-level difference table
    # -----------------------------------------------------
    tissue_diff_path = (
        OUT_DIR
        / "falff_tissue_differences.csv"
    )

    tissue_diff_df.to_csv(
        tissue_diff_path,
        index=False,
    )

    print(
        "\nSaved tissue differences:",
        tissue_diff_path,
    )


if __name__ == "__main__":
    main()