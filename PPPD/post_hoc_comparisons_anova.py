import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_output_path,
                  get_selected_subject_list, get_posthoc_cluster_mask)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import get_cluster_table_with_aal_labels
from nilearn.masking import apply_mask
import numpy as np
import pandas as pd
import seaborn as sns


# --- Script configuration:
task = "rest"
runs = ["run-01", "run-02"] # pre, post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "OperculumOP4R" # List of supported seeds:
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
# define which cluster mask from mixed anova to use
direction = "negative" # possible directions: "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)
# define color palette for plotting
palette = {"control": "teal", "patient": "hotpink"}


# --- Get all directories and participants.tsv:
# path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)

# read participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get output path
output_dir = get_output_path(part, feature, seed)
output_dir = os.path.join(output_dir, "pre_post_diff")
os.makedirs(output_dir, exist_ok=True)


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


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Load data:
# initialize lists for derivatives, subject ids and mask images
included_rows = []
mask_path = get_posthoc_cluster_mask(feature=feature, seed=seed, group_comparison=group_comparison, part=part,
                                      direction=direction)
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
included_df = included_df.merge(participants_df[["subject_id", "group"]], on="subject_id", how="left")
print("Loaded subjects:", included_df["subject_id"].nunique())


# --- Load mask with significant clusters:
# get binary positive or negative cluster mask
cluster_mask = nib.load(mask_path)


# --- Load original permutation cluster table
if feature == "seed_based":
    table_file_suffix = f"{feature}_{seed}_{group_comparison}"
else:
    table_file_suffix = f"{feature}_{group_comparison}"
part_label = "all" if part is None else f"{part}"
table_file_suffix = f"{table_file_suffix}_{part_label}"
cluster_table_path = os.path.join(output_dir, "cluster_tables", f"{table_file_suffix}_cluster_table_perm_mass.csv")
cluster_table_perm_mass = pd.read_csv(cluster_table_path)

# Keep only clusters matching selected direction
if direction == "negative":
    relevant_clusters = cluster_table_perm_mass[
        cluster_table_perm_mass["Stat"] < 0
    ].copy()
elif direction == "positive":
    relevant_clusters = cluster_table_perm_mass[
        cluster_table_perm_mass["Stat"] > 0
    ].copy()
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

    single_cluster_mask = nib.load(cluster_path)
    for _, row in included_df.iterrows():
        subject_id = row["subject_id"]
        group = row["group"]

        # pre/run-01
        pre_voxels = apply_mask(row["pre_img"], single_cluster_mask)
        pre_mean = np.mean(pre_voxels)
        rows.append({
            "cluster": cluster_id,
            "subject_id": subject_id,
            "group": group,
            "run": "pre",
            "value": pre_mean
        })

        # post/run-02
        post_voxels = apply_mask(row["post_img"], single_cluster_mask)
        post_mean = np.mean(post_voxels)
        rows.append({
            "cluster": cluster_id,
            "subject_id": subject_id,
            "group": group,
            "run": "post",
            "value": post_mean
        })

plot_df = pd.DataFrame(rows)


# --- Get dataframe with difference values
diff_df = (plot_df.pivot(index=["cluster", "subject_id", "group"], columns="run", values="value").reset_index())
diff_df["post_minus_pre"] = diff_df["post"] - diff_df["pre"]


# --- Plots:
plot_dir = os.path.join(output_dir, "post-hoc", "plots")
os.makedirs(plot_dir, exist_ok=True)
sns.set_theme(style="ticks")
sns.set_context("talk")

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


    # --- Plot 1: Pre/Post trajectories
    plt.figure(figsize=(7, 8))
    plt.axhline(0, color="grey", linewidth=2, alpha=0.5)
    sns.lineplot(data=this_plot, x="run", y="value", hue="group", units="subject_id", estimator=None, alpha=0.35,
                 linewidth=1.75, palette=palette, legend=False,)
    sns.pointplot(data=this_plot, x="run", y="value", hue="group", errorbar="se", markers="o", linestyles="-",
                  linewidth=3.5, palette=palette, legend=False,)
    plt.title(f"{seed}: pre-post values")
    plt.xlabel("")
    plt.ylabel(f"Mean value in {cluster_label}")
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(
        plot_dir, f"{file_suffix}_cluster-{cluster_id:02d}_{cluster_label}_lineplot_pre_post_by_group.png"), dpi=300)
    plt.show()


    # --- Plot 2: Difference values
    plt.figure(figsize=(7, 8))
    plt.axhline(0, color="grey", linewidth=2, alpha=0.5)
    sns.boxplot(data=this_diff, x="group", y="post_minus_pre", hue="group", showfliers=False, palette=palette,
                linewidth=2.5,)
    sns.stripplot(data=this_diff, x="group", y="post_minus_pre", jitter=True, alpha=0.5, color="black",)
    # y-position above data
    y_max = this_diff["post_minus_pre"].max()
    y_min = this_diff["post_minus_pre"].min()
    h = 0.02 * (y_max - y_min)
    y = y_max + h
    # x positions of groups
    x1 = 0
    x2 = 1
    # significance bracket
    plt.plot( [x1, x1, x2, x2], [y, y + h, y + h, y], lw=2, c="black")
    # stars
    plt.text((x1 + x2) * 0.5, y + h, stars, ha="center", va="bottom", fontsize=16, weight="bold",)
    plt.title(f"{seed}: difference (post - pre)")
    plt.xlabel("")
    plt.ylabel(f"Mean value in {cluster_label}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir, f"{file_suffix}_cluster-{cluster_id:02d}_{cluster_label}_boxplot_difference_by_group.png"), dpi=300)
    plt.show()
