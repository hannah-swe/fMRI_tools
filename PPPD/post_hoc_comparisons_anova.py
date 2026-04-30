import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename,
                  _get_output_path, _get_posthoc_cluster_mask, _get_cluster_table_with_aal_labels)
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.masking import apply_mask
import numpy as np
import pandas as pd
from scipy import ndimage
import seaborn as sns


# --- Script configuration:
task = "rest"
runs = ["run-01", "run-02"] # pre, post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "InsulaId1L" # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus"
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
# define which cluster mask from mixed anova to use
direction = "negative" # possible directions: "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)
# define color palette for plotting
palette = {"control": "teal", "patient": "hotpink"}


# --- Get all directories and participants.tsv:
# path to halfpipe derivatives directory
base_dir = _get_data_path(feature)

# read participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get derivatives path
deriv_dir = _get_derivatives_path(feature)

# get output path
output_dir = _get_output_path(feature)
output_dir = os.path.join(output_dir, "pre_post_diff", "post-hoc")
os.makedirs(output_dir, exist_ok=True)


# --- Choose subjects depending on experimental part:
if part is None:
    selected_subs = list(subs)
elif part == 1:
    selected_subs = [s for s in subs if s < 100]
elif part == 2:
    selected_subs = [s for s in subs if s >= 100]
else:
    raise ValueError("part must be None, 1, or 2")
# exclude subjects
selected_subs = [s for s in selected_subs if s not in subjects_to_exclude]
print(f"Selected part: {part if part is not None else 'all'}")
print(f"Selected subjects before loading: {len(selected_subs)}")


# --- Load data:
# initialize lists for derivatives, subject ids and mask images
included_rows = []

for s in selected_subs:
    # get full subject id
    subject_id = f"sub-{s:03d}"

    # load statistical nifti images of both runs
    run_imgs = {}
    for run in runs:
        filename = _get_full_filename(subject_id, task, run, feature, seed)
        img_path = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", filename)
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
mask_path = _get_posthoc_cluster_mask(feature=feature, seed=seed, group_comparison=group_comparison, part=part,
                                      direction=direction)
cluster_mask = nib.load(mask_path)

# --- Split significant mask into single clusters:
mask_data = cluster_mask.get_fdata() != 0
structure = ndimage.generate_binary_structure(3, 3)  # 26-neighbor connectivity
labeled_data, n_clusters = ndimage.label(mask_data, structure=structure)
print("Number of clusters:", n_clusters)
if n_clusters == 0:
    raise ValueError("No clusters found in mask.")

single_cluster_dir = os.path.join(output_dir, "single_cluster_masks")
os.makedirs(single_cluster_dir, exist_ok=True)

# get cluster mask with aal label
cluster_info = []
cluster_mask_paths = []
for cluster_id in range(1, n_clusters + 1):
    single_cluster_data = (labeled_data == cluster_id).astype(np.uint8)
    n_voxels = int(single_cluster_data.sum())
    single_cluster_img = nib.Nifti1Image(
        single_cluster_data,
        affine=cluster_mask.affine,
        header=cluster_mask.header
    )
    single_cluster_img.set_data_dtype(np.uint8)

    # get AAL label
    try:
        cluster_table = _get_cluster_table_with_aal_labels(stat_img=single_cluster_img, stat_threshold=0.5,
                                                           cluster_threshold=0, two_sided=False,)
        # first/main AAL label
        aal_label = cluster_table.iloc[0]["aal_label"]
        # sanitize for filenames
        aal_label_clean = (
            aal_label
            .replace(" ", "_")
            .replace("/", "-")
            .replace(",", "")
        )
    except Exception as e:
        print(f"Could not get AAL label for cluster {cluster_id}: {e}")
        aal_label_clean = f"cluster-{cluster_id:02d}"

    # save cluster mask
    cluster_filename = (
        f"{feature}_{seed}_{group_comparison}_"
        f"{'all' if part is None else part}_"
        f"{direction}_cluster-{cluster_id:02d}_"
        f"{aal_label_clean}.nii.gz"
    )
    cluster_path = os.path.join(single_cluster_dir, cluster_filename)
    single_cluster_img.to_filename(cluster_path)
    cluster_mask_paths.append(cluster_path)

    cluster_info.append({
        "cluster_id": cluster_id,
        "aal_label": aal_label_clean,
        "n_voxels": n_voxels,
        "path": cluster_path,
    })

    print(
        f"Saved cluster {cluster_id}: "
        f"{aal_label_clean} ({n_voxels} voxels)"
    )

cluster_info_df = pd.DataFrame(cluster_info)


# --- Extract connectivity per subject, run, and cluster:
rows = []
for cluster_id, cluster_path in enumerate(cluster_mask_paths, start=1):
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
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
sns.set_theme(style="ticks")
sns.set_context("talk")

for cluster_id in sorted(plot_df["cluster"].unique()):
    this_plot = plot_df[plot_df["cluster"] == cluster_id].copy()
    this_diff = diff_df[diff_df["cluster"] == cluster_id].copy()
    cluster_label = cluster_info_df.loc[cluster_info_df["cluster_id"] == cluster_id, "aal_label"].iloc[0]

    # --- Plot 1: Pre/Post trajectories
    plt.figure(figsize=(7, 8))
    plt.axhline(0, color="grey", linewidth=2, alpha=0.5)
    sns.lineplot(data=this_plot, x="run", y="value", hue="group", units="subject_id", estimator=None, alpha=0.4,
                 linewidth=1.5, palette=palette, legend=False,)
    sns.pointplot(data=this_plot, x="run", y="value", hue="group", errorbar="se", markers="o", linestyles="-",
                  linewidth=2.75, palette=palette, legend=False,)
    plt.title(f"{seed}: pre-post values")
    plt.xlabel("")
    plt.ylabel(f"Mean value in {cluster_label}")
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(plot_dir, f"{seed}_{cluster_label}_lineplot_pre_post_by_group.png"), dpi=300)
    plt.show()


    # --- Plot 2: Difference values
    plt.figure(figsize=(7, 8))
    plt.axhline(0, color="grey", linewidth=2, alpha=0.5)
    sns.boxplot(data=this_diff, x="group", y="post_minus_pre", hue="group", showfliers=False, palette=palette,
                linewidth=2.5,)
    sns.stripplot(data=this_diff, x="group", y="post_minus_pre", jitter=True, alpha=0.5, color="black",)
    plt.title(f"Cluster {cluster_id}: post - pre")
    plt.xlabel("")
    plt.ylabel("Post - pre mean value")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{seed}_{cluster_label}_boxplot_difference_by_group.png"), dpi=300)
    plt.show()

