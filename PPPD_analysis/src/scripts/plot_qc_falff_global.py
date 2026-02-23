from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from pppd.config.pipelines import palette

# CONFIG:
IN_TABLE = Path.cwd() / "outputs" / "qc_falff_table.csv"
OUT_DIR = Path.cwd() / "outputs" / "plots" / "falff_global_qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_TABLE)
print("Loaded:", IN_TABLE)
print("Rows:", len(df), "Unique subs:", df["sub"].nunique())

# Plot styles
sns.set_context("talk")
sns.set_style("ticks")
tissue_palette = {"GM": "dodgerblue", "CSF": "darkmagenta"}

# Helper: p -> stars + bracket
def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", color="black")

# define qc metrics to use
metrics = [
    ("falff_gm_mean",  "Global fALFF (GM mask)",  "falff_gm_mean_on_pipelines.png"),
    ("falff_csf_mean", "Global fALFF (CSF mask)", "falff_csf_mean_on_pipelines.png"),
    ("falff_wm_mean",  "Global fALFF (WM mask)",  "falff_wm_mean_on_pipelines.png"),
    ("falff_wb_mean", "Global fALFF (whole brain)", "falff_wb_mean_on_pipelines.png"),
]

# Plot
for metric, ylabel, fname in metrics:
    wide = (
        df.pivot(index="sub", columns="pipeline", values=metric)
        .dropna(subset=["no_scrub", "scrub"])
    )

    # Stats:
    t_stat, p_val = ttest_rel(wide["scrub"], wide["no_scrub"])
    stars = p_to_stars(p_val)
    print(f"\n{metric}: paired t-test scrub vs no_scrub")
    print(f"t = {t_stat:.3f}, p = {p_val:.3g}, n = {len(wide)}")

    # Plot 1: global fALFF (GM, WM, CSF) in different pipelines
    plt.figure(figsize=(7, 9))
    ax = sns.boxplot(
        data=df,
        x="pipeline",
        y=metric,
        hue = "pipeline",
        palette=palette,
        linewidth=1.5,
        linecolor="black",
        showcaps=False,
    )
    sns.lineplot(
        data=df,
        x="pipeline",
        y=metric,
        units="sub",
        estimator=None,
        color="black",
        linewidth=1.5,
        alpha=0.4,
    )
    # set y-axis limits and add significance bracket
    ymin, ymax = -0.6, 0.8
    ax.set_ylim(ymin, ymax)
    y_range = ymax - ymin
    y = ymax - 0.05 * y_range
    h = 0.02 * y_range
    add_sig_bracket(ax, 0, 1, y=y, h=h, text=stars)

    sns.despine()
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.show()


# compute deltas in wide form
w_gm = df.pivot(index="sub", columns="pipeline", values="falff_gm_mean").dropna()
w_csf = df.pivot(index="sub", columns="pipeline", values="falff_csf_mean").dropna()

delta = pd.DataFrame(index=sorted(set(w_gm.index).intersection(w_csf.index)))
delta["delta_gm"] = w_gm.loc[delta.index, "scrub"] - w_gm.loc[delta.index, "no_scrub"]
delta["delta_csf"] = w_csf.loc[delta.index, "scrub"] - w_csf.loc[delta.index, "no_scrub"]
delta = delta.reset_index(names="sub")
pct_scrub_df = (df[df["pipeline"] == "scrub"][["sub", "pct_scrub"]].drop_duplicates(subset="sub"))
delta = delta.merge(pct_scrub_df, on="sub", how="left")

# long format for plotting
delta_long = delta.melt(id_vars=["sub", "pct_scrub"], value_vars=["delta_gm", "delta_csf"],
                        var_name="tissue", value_name="delta_falff")
delta_long["tissue"] = delta_long["tissue"].replace({"delta_gm": "GM", "delta_csf": "CSF"})


# Plot 2: delta fALFF in GM and CSF
plt.figure(figsize=(7, 7))
sns.barplot(
    data=delta_long,
    x="tissue",
    y="delta_falff",
    hue="tissue",
    palette=tissue_palette,
)
sns.stripplot(
    data=delta_long,
    x="tissue",
    y="delta_falff",
    color="black",
    size=5,
    alpha=0.7,
    jitter=True)
plt.axhline(0, color="black", linewidth=1.5)
sns.despine()
plt.xlabel("")
plt.ylabel("Δ fALFF (scrub − no_scrub)")
plt.tight_layout()
plt.savefig(OUT_DIR / "delta_falff_gm_vs_csf.png", dpi=300)
plt.show()


# Plot 3: delta fALFF (scrub - no-scrub) on percentage of scrubbed volumes in GM ans CSF
plt.figure(figsize=(12, 7))
g = sns.lmplot(
    data=delta_long,
    x="pct_scrub",
    y="delta_falff",
    col="tissue",
    hue="tissue",
    ci=95,
    palette=tissue_palette,
    height=7,
    aspect=1,
)

for ax in g.axes.flat:
    ax.axhline(0, color="black", linewidth=1.5)
    sns.despine(ax=ax)

g.set_axis_labels("% scrubbed volumes", "Δ fALFF (scrub − no_scrub)")
plt.tight_layout()
plt.savefig(OUT_DIR / "delta_on_pct_scrub_gm_vs_csf.png", dpi=300)
plt.show()

