import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_output_pre_post_path,
                  get_selected_subject_list, get_posthoc_cluster_mask)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import get_cluster_table_with_aal_labels
from PPPD.config import load_config
from nilearn.masking import apply_mask
import numpy as np
import pandas as pd
import seaborn as sns

# ---- Load script configuration from config.yml --> don't change anything here, make all configurations in config.yml
config = load_config()

task = config["analysis"]["task"]
runs = config["analysis"]["runs"] # "run-01" == pre, "run-02" == post
part = config["analysis"]["part"] # supported: None, 1, 2 (None: all subjects; 1: subjects < 100; 2: subjects >= 100)
feature = config["analysis"]["feature"] # supported features: "falff", "seed_based", "alff"
seed = config["analysis"]["seeds"][0] # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R",
                                    # "OperculumOP4L", "OperculumOP4R", "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII",
                                    #  "HippocampusL", "HippocampusR"
                                    # "PrecuneusL", "PrecuneusR"
group_comparison = config["analysis"]["group_comparison"] # supported comparisons: "pat>HC", "HC>pat"
direction = "negative" # possible directions: "positive", "negative"
effect = config["analysis"]["effect"]
# define color palette for plotting
palette = {"control": "teal", "patient": "hotpink"}
palette_part = {1: "deepskyblue", 2: "darkblue"}
palette_time = {"pre": "teal", "post": "hotpink"}


print(f"Used configuration parameters:\n"
      f"task = {task}\nruns = {runs}\npart = {part}\nfeature = {feature}\nseed = {seed}\n"
      f"group_comparison = {group_comparison}\neffect = {effect}")


# --- Get all directories and participants.tsv:
# path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)

# read participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")
participants_df["part"] = participants_df["participant_id"].apply(lambda x: 1 if x < 100 else 2)

# get output path
output_dir = get_output_pre_post_path(part, feature)
output_dir = os.path.join(output_dir, f"{effect}")


# --- Define the file suffix
if feature == "seed_based":
    base_title = f"{feature} {seed}"
    file_suffix = f"{feature}_{seed}"
else:
    base_title = f"{feature}"
    file_suffix = f"{feature}"
if part is None:
    part_label = "all"
    base_title = f"{base_title}; subjects: {part_label}"
else:
    part_label = f"{part}"
    base_title = f"{base_title}; subjects part: {part_label}"
file_suffix = f"{file_suffix}_{part_label}_{direction}"


# --- Get path for significant cluster masks
mask_path = os.path.join(output_dir, "sig_cluster_masks", f"{direction}",
                         f"{feature}_{seed}_{group_comparison}_{part_label}_{effect}_{direction}_cluster_id_map.nii.gz")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Load data:
# initialize lists for derivatives, subject ids and mask images
included_rows = []

for s in selected_subs:
    # get full subject id
    subject_id = f"sub-{s:03d}"

    # load statistical nifti images of both runs
    run_imgs = {}
    for run in runs:
        filename = get_full_filename(subject_id, task, run, feature, seed)
        img_path = os.path.join(deriv_dir, subject_id, filename)
        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue
        try:
            nib.load(img_path)
            run_imgs[run] = img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    # make sure all subjects do have images for both runs
    if "run-01" not in run_imgs or "run-02" not in run_imgs:
        print(f"Skipping {subject_id}: incomplete pre/post images")
        continue

    included_rows.append({
        "subject_id": subject_id,
        "subject_num": s,
        "pre_img": run_imgs["run-01"],
        "post_img": run_imgs["run-02"],
    })

included_df = pd.DataFrame(included_rows)
included_df = included_df.merge(participants_df[["subject_id", "group", "part"]], on="subject_id", how="left")
print("Loaded subjects:", included_df["subject_id"].nunique())


# --- Load mask with significant clusters:
# get binary positive or negative cluster mask
cluster_mask = nib.load(mask_path)


# --- Load original permutation cluster table
if feature == "seed_based":
    table_file_suffix = f"{feature}_{seed}_{group_comparison}"
else:
    table_file_suffix = f"{feature}_{group_comparison}"
table_file_suffix = f"{table_file_suffix}_{part_label}_{effect}"
cluster_table_path = os.path.join(output_dir, "cluster_tables", f"{table_file_suffix}_cluster_table_perm_mass.csv")
cluster_table_perm_mass = pd.read_csv(cluster_table_path)

# Keep only clusters matching selected direction
if direction == "negative":
    relevant_clusters = cluster_table_perm_mass[cluster_table_perm_mass["Direction"] == "negative"].copy()
elif direction == "positive":
    relevant_clusters = cluster_table_perm_mass[cluster_table_perm_mass["Direction"] == "positive"].copy()
else:
    raise ValueError("direction must be 'positive' or 'negative'")

relevant_clusters = relevant_clusters.reset_index(drop=True)

print("Loaded permutation cluster table:")
print(cluster_table_path)
print("Relevant clusters from table:", len(relevant_clusters))


# --- Split labeled cluster map into single clusters:
labeled_data = cluster_mask.get_fdata().astype(int)

cluster_ids = sorted(np.unique(labeled_data))
cluster_ids = [c for c in cluster_ids if c != 0]

print("Number of clusters in labeled map:", len(cluster_ids))
if len(cluster_ids) == 0:
    raise ValueError("No clusters found in labeled cluster map.")
if len(cluster_ids) != len(relevant_clusters):
    print(
        "WARNING: Number of clusters in labeled map does not match number of clusters "
        "in permutation table for this direction."
    )
    print(f"Map clusters: {len(cluster_ids)}")
    print(f"Table clusters: {len(relevant_clusters)}")
    print("Map cluster IDs:", cluster_ids)
    print("Table cluster IDs:", sorted(relevant_clusters["Cluster"].unique()))

single_cluster_dir = os.path.join(output_dir, "post-hoc", "single_cluster_masks")
os.makedirs(single_cluster_dir, exist_ok=True)

cluster_info = []
cluster_mask_paths = []

for cluster_id in cluster_ids:
    single_cluster_data = (labeled_data == cluster_id).astype(np.uint8)
    n_voxels = int(single_cluster_data.sum())

    # Get matching row by true cluster ID, not by row order
    matching_rows = relevant_clusters.loc[relevant_clusters["Cluster"] == cluster_id]

    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one table row for cluster_id={cluster_id}, "
            f"found {len(matching_rows)}."
        )

    table_row = matching_rows.iloc[0]

    cluster_peak_stat = table_row["Stat"]
    cluster_p = table_row["p-value"]
    table_cluster_id = table_row["Cluster"]

    print(
        f"Cluster {cluster_id}: "
        f"table Cluster ID = {table_cluster_id}, "
        f"Peak Stat = {cluster_peak_stat:.3f}, "
        f"p = {cluster_p:.5f}"
    )

    # Get single cluster image
    single_cluster_img = nib.Nifti1Image(
        single_cluster_data,
        affine=cluster_mask.affine,
        header=cluster_mask.header
    )
    single_cluster_img.set_data_dtype(np.uint8)

    # Get AAL label
    try:
        cluster_table = get_cluster_table_with_aal_labels(
            stat_img=single_cluster_img,
            stat_threshold=0.5,
            cluster_threshold=0,
            two_sided=False
        )
        aal_label = cluster_table.iloc[0]["aal_label"]
        aal_label_clean = (
            aal_label
            .replace(" ", "_")
            .replace("/", "-")
            .replace(",", "")
        )
    except Exception as e:
        print(f"Could not get AAL label for cluster {cluster_id}: {e}")
        aal_label_clean = f"cluster-{cluster_id:02d}"

    # Save cluster mask
    cluster_filename = f"{file_suffix}_cluster-{cluster_id:02d}_{aal_label_clean}.nii.gz"
    cluster_path = os.path.join(single_cluster_dir, cluster_filename)
    single_cluster_img.to_filename(cluster_path)

    cluster_mask_paths.append(cluster_path)
    cluster_info.append({
        "cluster_id": cluster_id,
        "table_cluster_id": table_cluster_id,
        "aal_label": aal_label_clean,
        "direction": direction,
        "n_voxels": n_voxels,
        "peak_stat": cluster_peak_stat,
        "p_value": cluster_p,
        "path": cluster_path,
    })
    print(
        f"Saved cluster {cluster_id}: "
        f"{aal_label_clean} ({n_voxels} voxels)"
    )

cluster_info_df = pd.DataFrame(cluster_info)


# --- Extract connectivity per subject, run, and cluster:
rows = []
for _, cluster_row in cluster_info_df.iterrows():
    cluster_id = cluster_row["cluster_id"]
    cluster_path = cluster_row["path"]
    print(f"Extracting subject-wise connectivity values for cluster {cluster_id}.")

    single_cluster_mask = nib.load(cluster_path)
    for _, row in included_df.iterrows():
        subject_id = row["subject_id"]
        group = row["group"]
        part = row["part"]

        # pre/run-01
        pre_voxels = apply_mask(row["pre_img"], single_cluster_mask)
        pre_mean = np.mean(pre_voxels)
        pre_median = np.median(pre_voxels)
        rows.append({
            "cluster": cluster_id,
            "subject_id": subject_id,
            "group": group,
            "part": part,
            "run": "pre",
            "mean": pre_mean,
            "median": pre_median,
        })

        # post/run-02
        post_voxels = apply_mask(row["post_img"], single_cluster_mask)
        post_mean = np.mean(post_voxels)
        post_median = np.median(post_voxels)
        rows.append({
            "cluster": cluster_id,
            "subject_id": subject_id,
            "group": group,
            "part": part,
            "run": "post",
            "mean": post_mean,
            "median": post_median,
        })

    print (f"Finished extraction for cluster {cluster_id}.")

plot_df = pd.DataFrame(rows)


# --- Get dataframe with difference values
diff_df = (plot_df.pivot(index=["cluster", "subject_id", "group", "part"], columns="run", values="mean").reset_index())
diff_df["post_minus_pre"] = diff_df["post"] - diff_df["pre"]


# --- Plots:
plot_dir = os.path.join(output_dir, "post-hoc", "plots", f"{seed}")
os.makedirs(plot_dir, exist_ok=True)
sns.set_theme(style="ticks")
sns.set_context()

for cluster_id in sorted(plot_df["cluster"].unique()):
    info = cluster_info_df[cluster_info_df["cluster_id"] == cluster_id]
    print("\ncluster_id:", cluster_id)
    print(info[["cluster_id", "aal_label", "p_value"]])
    if len(info) != 1:
        raise ValueError(f"Problem bei cluster_id={cluster_id}: {len(info)} Treffer")
    cluster_label = info["aal_label"].iloc[0]
    cluster_p = info["p_value"].iloc[0]
    print("USED:", cluster_id, cluster_label, cluster_p)

for cluster_id in sorted(plot_df["cluster"].unique()):
    this_plot = plot_df[plot_df["cluster"] == cluster_id].copy()
    this_diff = diff_df[diff_df["cluster"] == cluster_id].copy()
    cluster_label = cluster_info_df.loc[cluster_info_df["cluster_id"] == cluster_id, "aal_label"].iloc[0]

    # get stars for asteriks
    cluster_p = cluster_info_df.loc[cluster_info_df["cluster_id"] == cluster_id, "p_value"].iloc[0]
    if cluster_p < 0.001:
        stars = "***"
    elif cluster_p < 0.01:
        stars = "**"
    elif cluster_p < 0.05:
        stars = "*"
    else:
        stars = "n.s."


    # --- Plot intercept effects (lineplot pre-post)
    if effect == "intercept":
        # --- Plot 1: Pre/Post trajectories
        plt.figure(figsize=(2.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.lineplot(data=this_plot, x="run", y="value", hue="group", units="subject_id", estimator=None, alpha=0.3,
                     linewidth=1, palette=palette, legend=False,)
        sns.pointplot(data=this_plot, x="run", y="value", hue="group", errorbar="se", markers="o", linestyles="-",
                      linewidth=2, palette=palette, legend=False,)
        sns.pointplot(data=this_plot, x="run", y="value", errorbar="se", markers="o", linestyles="-",
                      linewidth=2, color="black", legend=False,)
        xlim = plt.xlim()
        # y-position above data
        y_max = this_plot["value"].max()
        y_min = this_plot["value"].min()
        h = 0.02 * (y_max - y_min)
        y = y_max + h
        # x positions of groups
        x1 = 0
        x2 = 1
        # significance bracket
        plt.plot( [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        # stars
        plt.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=12, weight="bold",)
        plt.xlim(xlim)
        # plt.title(f"{seed}: pre-post values")
        plt.xlabel("")
        plt.ylabel(f"Median {cluster_label} ({cluster_id})")
        plt.tight_layout()
        sns.despine()
        plt.savefig(os.path.join(
            plot_dir, f"{file_suffix}_cluster-{cluster_id:02d}_{cluster_label}_lineplot_pre_post.png"),
            dpi=300)
        plt.show()


    # --- Plot group effects (boxplot with difference per group, lineplot pre-post)
    elif effect == "group":
        # --- Plot 1: Pre/Post trajectories
        plt.figure(figsize=(2.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.lineplot(data=this_plot, x="run", y="mean", hue="group", units="subject_id", estimator=None, alpha=0.3,
                     linewidth=1, palette=palette, legend=False,)
        sns.pointplot(data=this_plot, x="run", y="mean", hue="group", errorbar="se", markers="o", linestyles="-",
                      linewidth=2, palette=palette, legend=False,)
        plt.xlabel("")
        plt.ylabel(f"Mean {cluster_label} ({cluster_id})")
        plt.tight_layout()
        sns.despine()
        plt.savefig(os.path.join(
            plot_dir, f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_lineplot_pre_post_by_group.png"),
            dpi=300)
        plt.show()

        # --- Plot 2: Difference values
        plt.figure(figsize=(2.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.boxplot(data=this_diff, x="group", y="post_minus_pre", hue="group", showfliers=False, palette=palette,
                    linewidth=1.5, fill=False, legend=False,)
        sns.stripplot(data=this_diff, x="group", y="post_minus_pre", hue="group", palette=palette, jitter=True,
                      alpha=0.6, size=6, legend=False)
        # y-position above data
        y_max = this_diff["post_minus_pre"].max()
        y_min = this_diff["post_minus_pre"].min()
        h = 0.02 * (y_max - y_min)
        y = y_max + h
        # x positions of groups
        x1 = 0
        x2 = 1
        # significance bracket
        plt.plot( [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        # stars
        plt.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=12, weight="bold",)
        plt.xlabel("")
        plt.ylabel(f"Mean {cluster_label} ({cluster_id})")
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(
            plot_dir, f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_boxplot_difference_by_group.png"), dpi=300)
        plt.show()


        #---------------------------------------------------------------------------------------------------------------
        # BARPLOTS WITH STRIPPLOT FOR SUBJECT VALUES
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.axhline(0, color="grey", linewidth=1.5, alpha=0.5)

        # Reihenfolge definieren
        group_order = ["control", "patient"]
        run_order = ["pre", "post"]

        # Farben
        colors = {
            ("control", "pre"): "mediumturquoise",
            ("control", "post"): "teal",
            ("patient", "pre"): "lightpink",
            ("patient", "post"): "hotpink"
        }

        # Barplot
        sns.barplot(
            data=this_plot,
            x="group",
            y="mean",
            hue="run",
            order=group_order,
            hue_order=run_order,
            errorbar="ci",
            err_kws={"color": "black", "linewidth": 2},
            dodge=True,
            ax=ax,
            legend=False,
            fill=False,
        )

        # Balken individuell einfärben
        for container, run in zip(ax.containers, run_order):
            for bar, group in zip(container.patches, group_order):
                bar.set_facecolor("none")
                bar.set_edgecolor(colors[(group, run)])
                bar.set_linewidth(1.5)

        # Stripplot
        sns.stripplot(
            data=this_plot,
            x="group",
            y="mean",
            hue="run",
            order=group_order,
            hue_order=run_order,
            dodge=True,
            jitter=0.15,
            alpha=0.6,
            size=6,
            ax=ax,
            legend=False
        )

        # Punkte individuell einfärben
        for collection, (group, run) in zip(
                ax.collections,
                [(g, r) for g in group_order for r in run_order]
        ):
            collection.set_facecolor(colors[(group, run)])

        ax.set_xlabel("")
        ax.set_ylabel(f"Mean {cluster_label} ({cluster_id})")

        sns.despine()
        plt.tight_layout()
        plt.show()
        #---------------------------------------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------------------------------------
        # PAIRED STRIPPLOT WITH POINTPLOT FOR GROUP MEANS
        fig, ax = plt.subplots(figsize=(4, 4))

        # Reihenfolge
        group_order = ["control", "patient"]
        run_order = ["pre", "post"]

        # Farben
        colors = {
            ("control", "pre"): "mediumturquoise",
            ("control", "post"): "teal",
            ("patient", "pre"): "lightpink",
            ("patient", "post"): "hotpink"
        }

        # X-Positionen
        positions = {
            ("control", "pre"): -0.2,
            ("control", "post"): 0.2,
            ("patient", "pre"): 0.8,
            ("patient", "post"): 1.2
        }

        ax.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        # 1. Individuelle Verbindungslinien
        for group in group_order:

            group_data = this_plot[this_plot["group"] == group]

            for subject, subject_data in group_data.groupby("subject_id"):

                pre = subject_data[subject_data["run"] == "pre"]["mean"]
                post = subject_data[subject_data["run"] == "post"]["mean"]

                if len(pre) == 1 and len(post) == 1:
                    ax.plot(
                        [positions[(group, "pre")],
                         positions[(group, "post")]],
                        [pre.iloc[0], post.iloc[0]],
                        color="grey",
                        alpha=0.2,
                        linewidth=0.8,
                        zorder=1
                    )

        # 2. Individuelle Datenpunkte
        rng = np.random.default_rng(42)

        for group in group_order:
            for run in run_order:
                subset = this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]

                x = positions[(group, run)]

                jitter = rng.uniform(-0.04, 0.04, len(subset))

                ax.scatter(
                    np.full(len(subset), x) + jitter,
                    subset["mean"],
                    color=colors[(group, run)],
                    alpha=0.6,
                    s=25,
                    zorder=2
                )

        # 3. Mittelwerte + 95%-Konfidenzintervalle
        for group in group_order:
            for run in run_order:

                subset = this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]

                values = subset["mean"].dropna()

                if len(values) < 2:
                    continue

                mean = values.mean()

                # 95%-Konfidenzintervall (Bootstrap)
                from scipy.stats import bootstrap

                ci = bootstrap(
                    (values.to_numpy(),),
                    np.mean,
                    confidence_level=0.95,
                    random_state=42
                ).confidence_interval

                x = positions[(group, run)]

                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - ci.low], [ci.high - mean]],
                    fmt="o",
                    color="black",
                    markersize=7,
                    capsize=4,
                    linewidth=2,
                    zorder=5
                )

        # 4. Layout
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["control", "patient"])

        ax.set_ylabel(f"Mean {cluster_label} ({cluster_id})")
        ax.set_xlabel("")

        sns.despine()
        plt.tight_layout()
        plt.show()
        #---------------------------------------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(4, 4))

        # Reihenfolge
        group_order = ["control", "patient"]
        run_order = ["pre", "post"]

        # Farben
        colors = {
            ("control", "pre"): "mediumturquoise",
            ("control", "post"): "teal",
            ("patient", "pre"): "lightpink",
            ("patient", "post"): "hotpink"
        }

        # Horizontale Nulllinie
        ax.axhline(0, color="grey", linewidth=1.5, alpha=0.5)

        # 1. Stripplot: Individuelle Datenpunkte
        sns.stripplot(
            data=this_plot,
            x="group",
            y="mean",
            hue="run",
            order=group_order,
            hue_order=run_order,
            palette=["mediumturquoise", "teal"],
            dodge=True,
            jitter=0.15,
            alpha=0.6,
            size=6,
            ax=ax,
            legend=False
        )

        # Punkte individuell einfärben
        for collection, (group, run) in zip(
                ax.collections,
                [(g, r) for g in group_order for r in run_order]
        ):
            collection.set_facecolor(colors[(group, run)])

        # 2. Pointplot: Gruppenmittelwerte + 95%-CI
        sns.pointplot(
            data=this_plot,
            x="group",
            y="mean",
            hue="run",
            order=group_order,
            hue_order=run_order,
            dodge=0.4,
            errorbar=("ci", 95),
            n_boot=1000,
            color="black",
            markers="o",
            markersize=7,
            linestyles="none",
            capsize=0.03,
            err_kws={"linewidth": 1.5},
            ax=ax,
            legend=False
        )

        # 3. Verbindungslinien zwischen Gruppenmittelwerten
        positions = {
            ("control", "pre"): -0.2,
            ("control", "post"): 0.2,
            ("patient", "pre"): 0.8,
            ("patient", "post"): 1.2
        }

        for group in group_order:
            means = [
                this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]["mean"].mean()
                for run in run_order
            ]

            x_positions = [
                positions[(group, run)]
                for run in run_order
            ]

            ax.plot(
                x_positions,
                means,
                color="black",
                linewidth=1.5,
                zorder=4
            )

        # 4. Layout
        ax.set_xlabel("")
        ax.set_ylabel(f"Mean {cluster_label} ({cluster_id})")

        sns.despine()
        plt.tight_layout()
        plt.show()
        #---------------------------------------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(3, 4))

        # Reihenfolge
        group_order = ["control", "patient"]
        run_order = ["pre", "post"]

        # Farben
        colors = {
            ("control", "pre"): "mediumturquoise",
            ("patient", "pre"): "lightpink",
            ("control", "post"): "teal",
            ("patient", "post"): "hotpink"
        }

        # X-Positionen
        positions = {
            ("control", "pre"): -0.08,
            ("patient", "pre"): 0.08,
            ("control", "post"): 0.92,
            ("patient", "post"): 1.08
        }

        # Horizontale Nulllinie
        ax.axhline(
            0,
            color="grey",
            linewidth=1.5,
            alpha=0.5,
            zorder=0
        )

        # --------------------------------------------------
        # 1. Stripplot: Individuelle Datenpunkte
        # --------------------------------------------------

        rng = np.random.default_rng(42)

        for run in run_order:
            for group in group_order:
                subset = this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]

                x = positions[(group, run)]

                jitter = rng.uniform(
                    -0.06, 0.06, len(subset)
                )

                ax.scatter(
                    np.full(len(subset), x) + jitter,
                    subset["mean"],
                    color=colors[(group, run)],
                    alpha=0.5,
                    s=25,
                    edgecolors="none",
                    zorder=2
                )

        # --------------------------------------------------
        # 2. Mittelwerte + 95%-Konfidenzintervalle
        # --------------------------------------------------

        for run in run_order:
            for group in group_order:

                subset = this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]

                values = subset["mean"].dropna().to_numpy()

                if len(values) == 0:
                    continue

                mean = np.mean(values)

                if len(values) >= 2:

                    ci = bootstrap(
                        (values,),
                        np.mean,
                        confidence_level=0.95,
                        n_resamples=1000,
                        random_state=42
                    ).confidence_interval

                    yerr = [
                        [mean - ci.low],
                        [ci.high - mean]
                    ]

                else:
                    yerr = None

                ax.errorbar(
                    positions[(group, run)],
                    mean,
                    yerr=yerr,
                    fmt="o",
                    color="black",
                    markersize=6,
                    capsize=4,
                    linewidth=1.5,
                    zorder=5
                )

        # --------------------------------------------------
        # 3. Verbindungslinien zwischen Gruppenmittelwerten
        # --------------------------------------------------

        for group in group_order:
            means = [
                this_plot[
                    (this_plot["group"] == group) &
                    (this_plot["run"] == run)
                    ]["mean"].mean()
                for run in run_order
            ]

            x_positions = [
                positions[(group, run)]
                for run in run_order
            ]

            ax.plot(
                x_positions,
                means,
                color="black",
                linewidth=1.5,
                zorder=4
            )

        # --------------------------------------------------
        # 4. Layout
        # --------------------------------------------------

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre", "post"])

        ax.set_xlim(-0.4, 1.4)

        ax.set_xlabel("")
        ax.set_ylabel(f"Mean {cluster_label} ({cluster_id})")

        sns.despine()
        plt.tight_layout()
        plt.show()
        #---------------------------------------------------------------------------------------------------------------



    # --- Plot part effects (boxplot with difference per part, lineplot pre-post)
    elif effect == "part":
        # --- Plot 1: Pre/Post trajectories
        plt.figure(figsize=(2.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.lineplot(data=this_plot, x="run", y="mean", hue="part", units="subject_id", estimator=None, alpha=0.3,
                     linewidth=1, palette=palette_part, legend=False,)
        sns.pointplot(data=this_plot, x="run", y="mean", hue="part", errorbar="se", markers="o", linestyles="-",
                      linewidth=2, palette=palette_part, legend=False,)
        plt.xlabel("")
        plt.ylabel(f"Mean {cluster_label} ({cluster_id})")
        plt.tight_layout()
        sns.despine()
        plt.savefig(os.path.join(
            plot_dir, f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_lineplot_pre_post_by_part.png"),
            dpi=300)
        plt.show()

        # --- Plot 2: Difference values
        plt.figure(figsize=(2.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.boxplot(data=this_diff, x="part", y="post_minus_pre", hue="part", showfliers=False, palette=palette_part,
                    linewidth=1.5, fill=False, legend=False)
        sns.stripplot(data=this_diff, x="part", y="post_minus_pre", hue="part", palette=palette_part, jitter=True,
                      alpha=0.6, size=6, legend=False)
        # y-position above data
        y_max = this_diff["post_minus_pre"].max()
        y_min = this_diff["post_minus_pre"].min()
        h = 0.02 * (y_max - y_min)
        y = y_max + h
        # x positions of groups
        x1 = 0
        x2 = 1
        # significance bracket
        plt.plot( [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        # stars
        plt.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=12, weight="bold",)
        # plt.title(f"{seed}: difference (post - pre)")
        plt.xlabel("")
        plt.ylabel(f"Mean {cluster_label} ({cluster_id})")
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(
            plot_dir, f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_boxplot_difference_by_part.png"), dpi=300)
        plt.show()


    # --- Plot group x part interaction (boxplot with difference per group and part)
    elif effect == "groupxpart":
        # --- Plot 1: Difference values
        plt.figure(figsize=(3.5, 4))
        plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
        sns.boxplot(data=this_diff, x="part", y="post_minus_pre", hue="group", showfliers=False, palette=palette,
                    linewidth=1.5, fill=False, legend=False)
        sns.stripplot(data=this_diff, x="part", y="post_minus_pre", hue="group", palette=palette, jitter=True,
                      dodge=True, alpha=0.6, size=6, legend=False)
        # y-position above data
        y_max = this_diff["post_minus_pre"].max()
        y_min = this_diff["post_minus_pre"].min()
        h = 0.02 * (y_max - y_min)
        y = y_max + h
        # x positions of groups
        x1 = 0
        x2 = 1
        # significance bracket
        plt.plot( [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        # stars
        plt.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=12, weight="bold",)
        # plt.title(f"{seed}: difference (post - pre)")
        plt.xlabel("")
        plt.ylabel(f"Mean {cluster_label} ({cluster_id})")
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(
            plot_dir,
            f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_boxplot_difference_by_group_x_part.png"),
            dpi=300
        )
        plt.show()

        for part in this_plot["part"].unique():
            part_plot = this_plot[this_plot["part"] == part]

            # --- Plot 1: Pre/Post trajectories
            plt.figure(figsize=(2.5, 4))
            plt.axhline(0, color="grey", linewidth=1.5, alpha=0.5)
            sns.lineplot(data=part_plot, x="run", y="mean", hue="group", units="subject_id", estimator=None, alpha=0.3,
                         linewidth=1, palette=palette, legend=False,)
            sns.pointplot(data=part_plot, x="run", y="mean", hue="group", errorbar="se", markers="o", linestyles="-",
                          linewidth=2, palette=palette, legend=False,)
            plt.xlabel("")
            plt.ylabel(f"Mean {cluster_label} ({cluster_id}); Part {part}")
            plt.ylim(-0.2, 0.3)
            plt.tight_layout()
            sns.despine()
            plt.savefig(os.path.join(
                plot_dir,
                f"{file_suffix}_mean_cluster-{cluster_id:02d}_{cluster_label}_lineplot_pre_post_by_group_part_{part}.png"),
                dpi=300)
            plt.show()

    else:
        raise ValueError("Invalid effect")
