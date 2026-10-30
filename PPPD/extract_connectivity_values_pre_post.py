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

# ---- Load script configuration from config.yml --> don't change anything here, make all configurations in config.yml
config = load_config()

task = config["analysis"]["task"]
runs = config["analysis"]["runs"] # "run-01" == pre, "run-02" == post
part = config["analysis"]["part"] # supported: None, 1, 2 (None: all subjects; 1: subjects < 100; 2: subjects >= 100)
feature = config["analysis"]["feature"] # supported features: "falff", "seed_based", "alff"
seeds = config["analysis"]["seeds"] # List of supported seeds:
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
pre_post_diff = True


print(f"Used configuration parameters:\n"
      f"task = {task}\nruns = {runs}\npart = {part}\nfeature = {feature}\nseeds = {seeds}\n"
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


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)



all_rows = []

for seed in seeds:
    print(f"\n === Running seed: {seed} ===")


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
    # get sig cluster mask directory
    mask_path = get_posthoc_cluster_mask(
        feature=feature,
        group_comparison=group_comparison,
        pre_post_diff=pre_post_diff,
        direction=direction,
        part=part,
        seed=seed,
        effect=effect
    )

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
            subject_part = row["part"]

            # pre/run-01
            pre_voxels = apply_mask(row["pre_img"], single_cluster_mask)
            pre_mean = np.mean(pre_voxels)
            pre_median = np.median(pre_voxels)
            rows.append({
                "seed": seed,
                "cluster": cluster_id,
                "subject_id": subject_id,
                "group": group,
                "part": subject_part,
                "run": "pre",
                "mean": pre_mean,
                "median": pre_median,
                "aal_label": cluster_row["aal_label"],
                "n_voxels": cluster_row["n_voxels"],
                "peak_stat": cluster_row["peak_stat"],
                "p_value": cluster_row["p_value"],
            })

            # post/run-02
            post_voxels = apply_mask(row["post_img"], single_cluster_mask)
            post_mean = np.mean(post_voxels)
            post_median = np.median(post_voxels)
            rows.append({
                "seed": seed,
                "cluster": cluster_id,
                "subject_id": subject_id,
                "group": group,
                "part": subject_part,
                "run": "post",
                "mean": post_mean,
                "median": post_median,
                "aal_label": cluster_row["aal_label"],
                "n_voxels": cluster_row["n_voxels"],
                "peak_stat": cluster_row["peak_stat"],
                "p_value": cluster_row["p_value"],
            })

        print(f"Finished extraction for cluster {cluster_id}.")

    # Collect rows from all seeds
    all_rows.extend(rows)

# Create one DataFrame containing all seeds
connectivity_df = pd.DataFrame(all_rows)


# --- Get dataframe with difference values
diff_df = (connectivity_df.pivot(index=["seed", "cluster", "subject_id", "group", "part", "aal_label", "n_voxels",
                                        "peak_stat", "p_value"], columns="run", values="mean").reset_index())
diff_df["post_minus_pre"] = diff_df["post"] - diff_df["pre"]


# --- Save both dataframes
filename_connectivity = f"connectivity_dataframe_cluster_for_effect_{effect}.csv"
connectivity_df.to_csv(os.path.join(output_dir, filename_connectivity), index=False)
print("saved connectivity df.")

filename_difference = f"difference_dataframe_cluster_for_effect_{effect}.csv"
diff_df.to_csv(os.path.join(output_dir, filename_difference), index=False)
print("saved difference df.")