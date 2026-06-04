import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_output_path,
                  get_selected_subject_list, get_posthoc_cluster_mask)
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.masking import apply_mask
import numpy as np
import pandas as pd


def get_dataframe_for_connectivity_values(
    feature,
    group_comparison,
    pre_post_diff,
    selected_subs,
    participants_df,
    task="rest",
    seed=None,
    run=None,
    part=None,
    direction=None,
):
    # get derivatives path
    deriv_dir = get_derivatives_path(feature)

    all_rows = []



    # --- Get all directories:
    # get output path
    output_dir = get_output_path(part, feature, seed)

    # get sig cluster mask directory and load mask
    mask_path = get_posthoc_cluster_mask(
        feature=feature,
        group_comparison=group_comparison,
        pre_post_diff=pre_post_diff,
        direction=direction,
        part=part,
        seed=seed,
        )
    cluster_mask = nib.load(mask_path)


    # --- Define the file suffix
    base_title = f"{feature}"
    file_suffix = f"{feature}"
    if part is None:
        part_label = "all"
        base_title = f"{base_title}; subjects: {part_label}"
    else:
        part_label = f"{part}"
        base_title = f"{base_title}; subjects part: {part_label}"
    file_suffix = f"{file_suffix}_{group_comparison}_{part_label}"


    # --- Load original permutation cluster table
    table_file_suffix = f"{feature}_{group_comparison}"
    # part_label = "all" if part is None else f"{part}"
    # table_file_suffix = f"{table_file_suffix}_{part_label}"
    cluster_table_path = os.path.join(output_dir, "cluster_tables", f"{file_suffix}_twosided_cluster_table_perm_mass.csv")
    cluster_table_perm_mass = pd.read_csv(cluster_table_path)

    print("mask_path:", mask_path)

    labeled_data = cluster_mask.get_fdata().astype(int)
    cluster_ids = sorted(np.unique(labeled_data))
    cluster_ids = [c for c in cluster_ids if c != 0]

    print("Map cluster IDs:", cluster_ids)
    print("Table cluster IDs:", sorted(cluster_table_perm_mass["Cluster"].unique()))


    # --- Load data:
    # initialize lists for derivatives, subject ids and mask images
    included_subjects = []

    # subject loop to load original stat map for connectivity extraction
    for s in selected_subs:
        # get full subject id
        subject_id = f"sub-{s:03d}"

        # load statistical nifti maps
        filename = get_full_filename(subject_id, task, run, feature, seed)
        img = os.path.join(deriv_dir, subject_id, filename)
        if not os.path.exists(img):
            print(f"Missing file: {img}")
            continue
        try:
            nib.load(img)
            included_subjects.append({
                "subject_id": subject_id,
                "subject_num": s,
                "img": img,
            })
        except Exception as e:
            print(f"Error loading {img}: {e}")
            continue

    print("Loaded images:", len(included_subjects))
    included_subjects_df = pd.DataFrame(included_subjects)
    included_subjects_df = included_subjects_df.merge(participants_df[["subject_id", "group"]], on="subject_id", how="left")


    # --- Split labeled cluster map into single clusters:
    labeled_data = cluster_mask.get_fdata().astype(int)

    cluster_ids = sorted(np.unique(labeled_data))
    cluster_ids = [c for c in cluster_ids if c != 0]

    print("Number of clusters in labeled map:", len(cluster_ids))
    if len(cluster_ids) == 0:
        raise ValueError("No clusters found in labeled cluster map.")
    if len(cluster_ids) != len(cluster_table_perm_mass):
        print(
            "WARNING: Number of clusters in labeled map does not match number of clusters "
            "in permutation table for this direction."
        )
        print(f"Map clusters: {len(cluster_ids)}")
        print(f"Table clusters: {len(cluster_table_perm_mass)}")
        print("Map cluster IDs:", cluster_ids)
        print("Table cluster IDs:", sorted(cluster_table_perm_mass["Cluster"].unique()))

    single_cluster_dir = os.path.join(output_dir, "correlations", "single_cluster_masks")
    os.makedirs(single_cluster_dir, exist_ok=True)

    cluster_info = []

    for cluster_id in cluster_ids:
        single_cluster_data = (labeled_data == cluster_id).astype(np.uint8)
        n_voxels = int(single_cluster_data.sum())

        # Get matching row by true cluster ID, not by row order
        matching_rows = cluster_table_perm_mass.loc[cluster_table_perm_mass["Cluster"] == cluster_id]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one table row for cluster_id={cluster_id}, "
                f"found {len(matching_rows)}."
            )

        table_row = matching_rows.iloc[0]

        cluster_peak_stat = table_row["Stat"]
        cluster_p = table_row["p-value"]
        table_cluster_id = table_row["Cluster"]
        aal_label = table_row["aal_label"]

        print(
            f"Cluster {cluster_id}: "
            f"table Cluster ID = {table_cluster_id}, "
            f"Peak Stat = {cluster_peak_stat:.3f}, "
            f"p = {cluster_p:.5f}, "
            f"aal = {aal_label}"
        )

        # Get single cluster image
        single_cluster_img = nib.Nifti1Image(
            single_cluster_data,
            affine=cluster_mask.affine,
            header=cluster_mask.header
        )
        single_cluster_img.set_data_dtype(np.uint8)

        # Save cluster mask
        cluster_filename = f"{file_suffix}_cluster-{cluster_id:02d}_{aal_label}.nii.gz"
        cluster_path = os.path.join(single_cluster_dir, cluster_filename)
        single_cluster_img.to_filename(cluster_path)

        cluster_info.append({
            "cluster_id": cluster_id,
            "table_cluster_id": table_cluster_id,
            "aal_label": aal_label,
            "n_voxels": n_voxels,
            "peak_stat": cluster_peak_stat,
            "p_value": cluster_p,
            "path": cluster_path,
        })
        print(
            f"Saved cluster {cluster_id}: "
            f"{aal_label} ({n_voxels} voxels)"
        )

    cluster_info_df = pd.DataFrame(cluster_info)

    # --- Extract connectivity per subject and cluster:
    for _, cluster_row in cluster_info_df.iterrows():
        cluster_id = cluster_row["cluster_id"]
        cluster_path = cluster_row["path"]

        single_cluster_mask = nib.load(cluster_path)

        for _, row in included_subjects_df.iterrows():
            subject_id = row["subject_id"]
            group = row["group"]

            voxels = apply_mask(row["img"], single_cluster_mask)
            mean = np.mean(voxels)
            median = np.median(voxels)

            all_rows.append({
                "cluster": cluster_id,
                "subject_id": subject_id,
                "subject_num": row["subject_num"],
                "group": group,
                "mean": mean,
                "median": median,
                "aal_label": cluster_row["aal_label"],
                "n_voxels": cluster_row["n_voxels"],
                "peak_stat": cluster_row["peak_stat"],
                "p_value": cluster_row["p_value"],
            })

    connectivity_df = pd.DataFrame(all_rows)
    return connectivity_df


# --- Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "falff" # supported features: "falff", "seed_based", "alff"
seed = None
                                    # List of supported seeds:
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


# --- Get output path to save dataframes as csv
output_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/"
if pre_post_diff is True:
    file_suffix = "pre-post-diff"
else:
    file_suffix = "pre-data"
filename_long = f"falff_{file_suffix}_dataframe_long_format.csv"
filename_wide = f"falff_{file_suffix}_dataframe_wide_format.csv"


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Get connectivity dataframe with subject values for all significant cluster per seed
df = get_dataframe_for_connectivity_values(feature, group_comparison, pre_post_diff, selected_subs,
                                            participants_df, task, seed, run, part, direction)

# --- Get connectivity dataframe in wide format
# make a unique cluster label for column names
df["cluster_label"] = ("falff--"+ df["aal_label"].astype(str))

# convert long to wide
connectivity_wide_df = df.pivot_table(index="subject_id", columns="cluster_label", values=["mean", "median"],
                                      aggfunc="first")

# flatten multi-level columns
connectivity_wide_df.columns = [f"{cluster}_{value_type}" for value_type, cluster in connectivity_wide_df.columns]

# make SubjID a normal column again
connectivity_wide_df = connectivity_wide_df.reset_index()

# extract subject-level variables
subject_info_df = df[["subject_id", "subject_num", "group"]].drop_duplicates()

# merge with wide dataframe
connectivity_wide_df = connectivity_wide_df.merge(subject_info_df, on="subject_id", how="left")

# reorder columns
front_cols = ["subject_id", "subject_num", "group"]
other_cols = [
    col for col in connectivity_wide_df.columns
    if col not in front_cols
]
connectivity_wide_df = connectivity_wide_df[front_cols + other_cols]


# --- Save both dataframes
df.to_csv(os.path.join(output_dir, filename_long), index=False)
print("saved connectivity df in long format.")
connectivity_wide_df.to_csv(os.path.join(output_dir, filename_wide), index=False)
print("saved connectivity df in wide format.")