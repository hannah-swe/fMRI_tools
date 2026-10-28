from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

from pppd.config.pipelines import palette


# =========================================================
# CONFIG
# =========================================================

FEATURE = "falff"   # "falff" oder "seedfc"

FEATURE_TABLE = (
    Path.cwd()
    / "outputs"
    / f"qc_{FEATURE}_table.csv"
)

OUT_DIR = (
    Path.cwd()
    / "outputs"
    / "plots"
    / FEATURE
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Plot style
sns.set_style("ticks")
sns.set_context()


# =========================================================
# HELPERS
# =========================================================

def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=1.5,
        color="black",
    )

    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        color="black",
    )


def get_ylabel(feature: str) -> str:
    if feature == "falff":
        return "Mean fALFF in DMN"

    if feature == "seedfc":
        return "Mean Seed FC in DMN"

    return feature


def get_difference_ylabel(feature: str) -> str:
    if feature == "falff":
        return "Δ fALFF (scrub − no scrub)"

    if feature == "seedfc":
        return "Δ Seed FC (scrub − no scrub)"

    return "Difference (scrub − no scrub)"


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # 1. Load feature table
    # -----------------------------------------------------
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}"
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
        "Unique subjects:",
        df["sub"].nunique(),
    )


    # -----------------------------------------------------
    # 2. Select common DMN outcome
    # -----------------------------------------------------
    outcome_col = (
        f"{FEATURE}_dmn_mean"
    )

    if outcome_col not in df.columns:
        raise KeyError(
            f"Missing expected column: {outcome_col}"
        )


    # -----------------------------------------------------
    # 3. Create paired dataset
    # -----------------------------------------------------
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

    if len(wide) < 3:
        raise RuntimeError(
            "Too few paired subjects for "
            f"statistical test: n={len(wide)}"
        )

    wide["difference"] = (
        wide["scrub"]
        - wide["no_scrub"]
    )

    print(
        "\nPaired subjects:",
        len(wide),
    )

    print(
        "\n=== Difference scores ==="
    )

    print(
        f"Mean difference = "
        f"{wide['difference'].mean():.6f}"
    )

    print(
        f"SD difference   = "
        f"{wide['difference'].std():.6f}"
    )

    print(
        f"Min difference  = "
        f"{wide['difference'].min():.6f}"
    )

    print(
        f"Max difference  = "
        f"{wide['difference'].max():.6f}"
    )


    # -----------------------------------------------------
    # 4. Paired t-test
    # -----------------------------------------------------
    t_stat, p_value = ttest_rel(
        wide["scrub"],
        wide["no_scrub"],
    )

    stars = p_to_stars(
        p_value
    )

    print(
        "\n=== Paired t-test ==="
    )

    print(
        f"Feature: {FEATURE}"
    )

    print(
        f"Outcome: {outcome_col}"
    )

    print(
        f"n = {len(wide)}"
    )

    print(
        f"t = {t_stat:.3f}"
    )

    print(
        f"p = {p_value:.6g}"
    )


    # -----------------------------------------------------
    # 5. Keep paired subjects for plotting
    # -----------------------------------------------------
    paired_subjects = (
        wide.index
    )

    plot_df = df[
        df["sub"].isin(
            paired_subjects
        )
    ].copy()

    pipeline_order = [
        "no_scrub",
        "scrub",
    ]


    # -----------------------------------------------------
    # 6. Boxplot + paired subject lines
    # -----------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(2.5, 4)
    )

    sns.boxplot(
        data=plot_df,
        x="pipeline",
        y=outcome_col,
        order=pipeline_order,
        hue="pipeline",
        palette=palette,
        dodge=False,
        linewidth=1.5,
        linecolor="black",
        showfliers=False,
        fill=True,
        legend=False,
        ax=ax,
    )

    sns.lineplot(
        data=plot_df,
        x="pipeline",
        y=outcome_col,
        units="sub",
        estimator=None,
        sort=False,
        legend=False,
        color="black",
        linewidth=1,
        alpha=0.25,
        ax=ax,
    )


    # -----------------------------------------------------
    # 7. Significance bracket
    # -----------------------------------------------------
    y_max = (
        plot_df[outcome_col]
        .max()
    )

    y_min = (
        plot_df[outcome_col]
        .min()
    )

    y_range = (
        y_max - y_min
    )

    if y_range == 0:
        y_range = 1

    bracket_y = (
        y_max
        + 0.05 * y_range
    )

    bracket_h = (
        0.02 * y_range
    )

    add_sig_bracket(
        ax=ax,
        x1=0,
        x2=1,
        y=bracket_y,
        h=bracket_h,
        text=stars,
    )


    # -----------------------------------------------------
    # 8. Labels
    # -----------------------------------------------------
    sns.despine(
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        get_ylabel(FEATURE)
    )

    # Avoid set_xticklabels warning
    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "No Scrub",
            "Scrub",
        ]
    )

    current_ylim = (
        ax.get_ylim()
    )

    ax.set_ylim(
        current_ylim[0],
        max(
            current_ylim[1],
            bracket_y
            + 0.08 * y_range,
        ),
    )

    fig.tight_layout()

    out_path = (
        OUT_DIR
        / f"{FEATURE}_dmn_scrub_vs_no_scrub.svg"
    )

    fig.savefig(
        out_path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()

    print(
        "\nSaved plot:",
        out_path,
    )


    # -----------------------------------------------------
    # 9. Difference plot
    # -----------------------------------------------------
    diff_df = (
        wide
        .reset_index()[
            [
                "sub",
                "difference",
            ]
        ]
    )

    fig, ax = plt.subplots(
        figsize=(1.75, 4)
    )

    sns.boxplot(
        data=diff_df,
        y="difference",
        linewidth=1.5,
        color="darkblue",
        showfliers=False,
        fill=False,
        ax=ax,
    )

    sns.stripplot(
        data=diff_df,
        y="difference",
        color="darkblue",
        jitter=True,
        alpha=0.6,
        size=6,
        ax=ax,
    )

    ax.axhline(
        0,
        color="grey",
        linewidth=1.5,
        alpha=0.5,
    )

    sns.despine(
        ax=ax
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        get_difference_ylabel(
            FEATURE
        )
    )

    fig.tight_layout()

    diff_out_path = (
        OUT_DIR
        / (
            f"{FEATURE}_dmn_"
            "difference_scrub_minus_no_scrub.svg"
        )
    )

    fig.savefig(
        diff_out_path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()

    print(
        "Saved difference plot:",
        diff_out_path,
    )


if __name__ == "__main__":
    main()