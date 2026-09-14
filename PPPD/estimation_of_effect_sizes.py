import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_output_path,
                  get_selected_subject_list)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.config import load_config
import numpy as np
import pandas as pd
import gc


def cohens_d_independent(x_patient, x_control):
    """
    Cohen's d for two independent groups using pooled SD.
    Positive d: patient > control
    """
    x_patient = np.asarray(x_patient, dtype=float)
    x_control = np.asarray(x_control, dtype=float)

    x_patient = x_patient[np.isfinite(x_patient)]
    x_control = x_control[np.isfinite(x_control)]

    n_pat = len(x_patient)
    n_con = len(x_control)

    if n_pat < 2 or n_con < 2:
        return np.nan

    var_pat = np.var(x_patient, ddof=1)
    var_con = np.var(x_control, ddof=1)

    pooled_sd = np.sqrt(
        ((n_pat - 1) * var_pat + (n_con - 1) * var_con)
        / (n_pat + n_con - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(x_patient) - np.mean(x_control)) / pooled_sd



# ---- Load script configuration from config.yml --> don't change anything here, make all configurations in config.yml
config = load_config()

task = config["analysis"]["task"]
run = "run-01" # "run-01" == pre, "run-02" == post
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
# define color palette for plotting
palette = {"control": "black", "patient": "teal"}
t_test_strategy = config["statistics"]["t_test_strategy"] # either two-sided or one-sided

# Definition of plot parameters depending on t-test strategy
if t_test_strategy == "two_sided":
    two_sided = True
    test_label = "twosided"
    cmap = "RdBu_r"
    symmetric_cbar = True
elif t_test_strategy == "one_sided":
    two_sided = False
    test_label = "onesided"
    cmap = "inferno"
    symmetric_cbar = False
else:
    raise ValueError(f"Unknown t_test_strategy: {t_test_strategy}. Use 'two_sided' or 'one_sided'.")

print(f"Used configuration parameters:\n"
      f"task = {task}\nrun = {run}\npart = {part}\nfeature = {feature}\nseeds = {seeds}\n"
      )


# --- Path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)


# --- Get participants.tsv:
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"\n=== Running seed: {seed} ===")

    # get output path
    data_dir = get_output_path(part, feature, seed)


    # --- Define the file suffix
    if feature == "seed_based":
        base_title = f"{feature} {seed}; {group_comparison}"
        file_suffix = f"{feature}_{seed}_{group_comparison}"
    else:
        base_title = f"{feature}; {group_comparison}"
        file_suffix = f"{feature}_{group_comparison}"
    if part is None:
        part_label = "all"
        base_title = f"{base_title}; subjects: {part_label}"
    else:
        part_label = f"{part}"
        base_title = f"{base_title}; subjects part: {part_label}"
    file_suffix = f"{file_suffix}_{part_label}_{test_label}"


    # --- Load maps with voxelwise t-values
    t_map_path = os.path.join(data_dir, "t_maps", f"{file_suffix}_tmap.nii.gz")
    # load t-value map
    if not os.path.exists(t_map_path):
        raise FileNotFoundError(f"t-map not found: {t_map_path}")
    t_img = nib.load(t_map_path)
    t_data = t_img.get_fdata()
    print("Loaded t-map:")
    print(t_map_path)


    # --- Get path for significant cluster masks
    sig_cluster_path = os.path.join(data_dir, "sig_cluster_masks", f"{file_suffix}_cluster_id_map.nii.gz")
    # load cluster map
    if not os.path.exists(sig_cluster_path):
        raise FileNotFoundError(f"Cluster map not found: {sig_cluster_path}")
    cluster_img = nib.load(sig_cluster_path)
    cluster_data = cluster_img.get_fdata()
    print("Loaded cluster map:")
    print(sig_cluster_path)


    # --- Load analysis mask
    mask_path = os.path.join(data_dir, "masks", f"group_mask_{file_suffix}.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Analysis mask not found: {mask_path}")
    analysis_mask_img = nib.load(mask_path)
    analysis_mask_data = analysis_mask_img.get_fdata().astype(bool)
    print("Loaded analysis mask:")
    print(mask_path)

    if analysis_mask_data.shape != cluster_data.shape:
        raise ValueError(
            f"Shape mismatch: analysis mask {analysis_mask_data.shape} "
            f"vs cluster map {cluster_data.shape}"
        )
    if not np.allclose(analysis_mask_img.affine, cluster_img.affine):
        raise ValueError("Affine mismatch between analysis mask and cluster map")


    # --- Load original permutation cluster table
    cluster_table_path = os.path.join(data_dir, "cluster_tables", f"{file_suffix}_cluster_table_perm_mass.csv")
    cluster_table_perm_mass = pd.read_csv(cluster_table_path)
    print("Loaded permutation cluster table:")
    print(cluster_table_path)
    print("Clusters from table:", len(cluster_table_perm_mass))


    # --- Split labeled cluster map into single clusters:
    cluster_ids = sorted(np.unique(cluster_data))
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


    # --- Load subject data:
    # initialize lists for derivatives, subject ids and mask images
    included_subjects = []

    # subject loop to load original stat map for connectivity extraction
    for s in selected_subs:
        # get full subject id
        subject_id = f"sub-{s:03d}"

        # load statistical nifti maps
        filename = get_full_filename(subject_id, task, run, feature, seed)
        img_path = os.path.join(deriv_dir, subject_id, filename)
        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue

        included_subjects.append({
            "subject_id": subject_id,
            "subject_num": s,
            "img": img_path,
        })

    print("Images to load:", len(included_subjects))
    included_subjects_df = pd.DataFrame(included_subjects)
    included_subjects_df = included_subjects_df.merge(
        participants_df[["subject_id", "group"]],
        on="subject_id",
        how="left"
    )
    n_control = (included_subjects_df["group"] == "control").sum()
    n_patient = (included_subjects_df["group"] == "patient").sum()

    print(f"Included controls: {n_control}")
    print(f"Included patients: {n_patient}")


    # --- Calculate voxelwise Cohen's d from voxelwise t-values
    conversion_factor = np.sqrt((1 / n_control) + (1 / n_patient))
    cohens_d_data = t_data * conversion_factor

    # handle invalid voxels
    cohens_d_data[~np.isfinite(t_data)] = np.nan

    # create NIfTI image
    cohens_d_img = nib.Nifti1Image(
        cohens_d_data,
        affine=t_img.affine,
        header=t_img.header.copy()
    )

    # save Cohen's d map
    cohens_d_dir = os.path.join(data_dir, "cohens_d_maps")
    os.makedirs(cohens_d_dir, exist_ok=True)
    cohens_d_path = os.path.join(cohens_d_dir, f"{file_suffix}_cohens-d.nii.gz")
    cohens_d_img.to_filename(cohens_d_path)
    print(f"Saved Cohen's d map:\n{cohens_d_path}")


    # ------------------------------------------------------------------------------------------------------------------
    # Meyer et al. Option 1:
    # Calculate subject-level mean signal within each significant cluster
    # ------------------------------------------------------------------------------------------------------------------

    subject_cluster_means = []

    for _, row in included_subjects_df.iterrows():

        subject_id = row["subject_id"]
        group = row["group"]
        img_path = row["img"]

        # load subject only once
        subject_img = nib.load(img_path)
        subject_data = subject_img.get_fdata()

        if subject_data.shape != cluster_data.shape:
            raise ValueError(
                f"Shape mismatch for {subject_id}: "
                f"subject image {subject_data.shape}, "
                f"cluster image {cluster_data.shape}"
            )

        # now process all clusters
        for cluster_id in cluster_ids:

            cluster_mask = cluster_data == cluster_id
            cluster_values = subject_data[cluster_mask]

            cluster_values = cluster_values[np.isfinite(cluster_values)]

            if len(cluster_values) == 0:
                subject_mean = np.nan
            else:
                subject_mean = np.mean(cluster_values)

            subject_cluster_means.append({
                "Cluster": int(cluster_id),
                "subject_id": subject_id,
                "group": group,
                "cluster_mean": subject_mean,
                "n_cluster_voxels": int(np.sum(cluster_mask)),
            })

    subject_cluster_means_df = pd.DataFrame(subject_cluster_means)


    option1_results = []

    for cluster_id in cluster_ids:
        cluster_df = subject_cluster_means_df[subject_cluster_means_df["Cluster"] == cluster_id]

        patient_values = cluster_df.loc[cluster_df["group"] == "patient", "cluster_mean"].values
        control_values = cluster_df.loc[cluster_df["group"] == "control", "cluster_mean"].values

        cluster_average_cohens_d = cohens_d_independent(patient_values, control_values)

        option1_results.append({
            "Cluster": int(cluster_id),
            "cluster_average_cohens_d": cluster_average_cohens_d,
            "mean_patient": np.nanmean(patient_values),
            "mean_control": np.nanmean(control_values),
            "n_patient": np.sum(np.isfinite(patient_values)),
            "n_control": np.sum(np.isfinite(control_values)),
        })

    option1_df = pd.DataFrame(option1_results)

    print("\nMeyer Option 1 results:")
    print(option1_df)


    # ------------------------------------------------------------------------------------------------------------------
    # Meyer et al. Option 2:
    # Maximum voxelwise Cohen's d within each significant cluster
    # ------------------------------------------------------------------------------------------------------------------

    option2_results = []

    for cluster_id in cluster_ids:

        cluster_mask = cluster_data == cluster_id

        # Cohen's d values within current cluster
        cluster_d_values = cohens_d_data[cluster_mask]

        # remove invalid values
        valid_mask = np.isfinite(cluster_d_values)
        cluster_d_values_valid = cluster_d_values[valid_mask]

        if len(cluster_d_values_valid) == 0:
            peak_cohens_d = np.nan
            peak_x = np.nan
            peak_y = np.nan
            peak_z = np.nan

        else:
            # strongest effect by absolute magnitude
            peak_idx_valid = np.argmax(np.abs(cluster_d_values_valid))
            peak_cohens_d = cluster_d_values_valid[peak_idx_valid]

            # voxel coordinates of all voxels in cluster
            cluster_voxels = np.argwhere(cluster_mask)

            # apply same validity mask
            valid_cluster_voxels = cluster_voxels[valid_mask]

            # voxel index of peak
            peak_voxel = valid_cluster_voxels[peak_idx_valid]

            # corresponding peak t-value
            peak_t_value = t_data[tuple(peak_voxel)]

            # Plausibility check:
            # calculate Cohen's d directly from subject values at peak voxel
            peak_patient_values = []
            peak_control_values = []

            for _, row in included_subjects_df.iterrows():

                subject_img = nib.load(row["img"])
                subject_data = subject_img.get_fdata()

                peak_value = subject_data[tuple(peak_voxel)]

                if not np.isfinite(peak_value):
                    continue

                if row["group"] == "patient":
                    peak_patient_values.append(peak_value)

                elif row["group"] == "control":
                    peak_control_values.append(peak_value)

            peak_cohens_d_direct = cohens_d_independent(
                peak_patient_values,
                peak_control_values
            )

            # convert voxel coordinates to world/MNI coordinates
            peak_xyz = nib.affines.apply_affine(
                t_img.affine,
                peak_voxel
            )

            peak_x, peak_y, peak_z = peak_xyz

            print(
                f"Cluster {cluster_id}: "
                f"d from t-map = {peak_cohens_d:.6f}, "
                f"d directly from subject data = {peak_cohens_d_direct:.6f}"
            )

        option2_results.append({
            "Cluster": int(cluster_id),
            "peak_t_value": peak_t_value,
            "option2_peak_cohens_d": peak_cohens_d,
            "option2_peak_cohens_d_direct": peak_cohens_d_direct,
            "peak_x": peak_x,
            "peak_y": peak_y,
            "peak_z": peak_z,
        })

    option2_df = pd.DataFrame(option2_results)

    print("\nMeyer Option 2 results:")
    print(option2_df)


    # ------------------------------------------------------------------------------------------------------------------
    # Meyer et al. Option 3:
    # Cohen's d based on a rectangular bounding box around each significant cluster
    # ------------------------------------------------------------------------------------------------------------------

    subject_bbox_means = []
    option3_results = []

    for cluster_id in cluster_ids:

        print(f"\nProcessing bounding box for cluster {cluster_id}")

        # binary mask of current cluster
        cluster_mask = cluster_data == cluster_id

        # voxel coordinates belonging to cluster
        cluster_voxels = np.argwhere(cluster_mask)

        if len(cluster_voxels) == 0:
            continue

        # minimum and maximum voxel coordinates along x, y, z
        mins = cluster_voxels.min(axis=0)
        maxs = cluster_voxels.max(axis=0)

        # create rectangular 3D bounding box
        bbox_mask = np.zeros(cluster_data.shape, dtype=bool)

        bbox_mask[
        mins[0]:maxs[0] + 1,
        mins[1]:maxs[1] + 1,
        mins[2]:maxs[2] + 1
        ] = True

        # restrict bounding box to voxels included in the original analysis
        bbox_mask &= analysis_mask_data

        n_cluster_voxels = int(cluster_mask.sum())
        n_bbox_voxels = int(bbox_mask.sum())

        print(
            f"Cluster {cluster_id}: "
            f"{n_cluster_voxels} cluster voxels, "
            f"{n_bbox_voxels} bounding-box voxels"
        )

        # --------------------------------------------------------------
        # Subject-wise mean within bounding box
        # --------------------------------------------------------------

        for _, row in included_subjects_df.iterrows():

            subject_id = row["subject_id"]
            group = row["group"]

            subject_img = nib.load(row["img"])
            subject_data = subject_img.get_fdata()

            if subject_data.shape != cluster_data.shape:
                raise ValueError(
                    f"Shape mismatch for {subject_id}: "
                    f"{subject_data.shape} vs {cluster_data.shape}"
                )

            if not np.allclose(subject_img.affine, cluster_img.affine):
                raise ValueError(
                    f"Affine mismatch for {subject_id}"
                )

            bbox_values = subject_data[bbox_mask]

            # remove NaN / inf
            bbox_values = bbox_values[np.isfinite(bbox_values)]

            if len(bbox_values) == 0:
                bbox_mean = np.nan
            else:
                bbox_mean = np.mean(bbox_values)

            subject_bbox_means.append({
                "Cluster": int(cluster_id),
                "subject_id": subject_id,
                "group": group,
                "bbox_mean": bbox_mean,
                "n_cluster_voxels": n_cluster_voxels,
                "n_bbox_voxels": n_bbox_voxels,
            })

    subject_bbox_means_df = pd.DataFrame(subject_bbox_means)

    for cluster_id in cluster_ids:
        cluster_bbox_df = subject_bbox_means_df[subject_bbox_means_df["Cluster"] == cluster_id]

        patient_values = cluster_bbox_df.loc[cluster_bbox_df["group"] == "patient", "bbox_mean"].values
        control_values = cluster_bbox_df.loc[cluster_bbox_df["group"] == "control", "bbox_mean"].values

        bbox_cohens_d = cohens_d_independent(patient_values, control_values)

        # voxel counts for current cluster
        n_cluster_voxels = int(cluster_bbox_df["n_cluster_voxels"].iloc[0])
        n_bbox_voxels = int(cluster_bbox_df["n_bbox_voxels"].iloc[0])

        option3_results.append({
            "Cluster": int(cluster_id),
            "option3_bounding_box_cohens_d": bbox_cohens_d,
            "n_cluster_voxels": n_cluster_voxels,
            "n_bbox_voxels": n_bbox_voxels,
            "bbox_mean_patient": np.nanmean(patient_values),
            "bbox_mean_control": np.nanmean(control_values),
            "bbox_n_patient": np.sum(np.isfinite(patient_values)),
            "bbox_n_control": np.sum(np.isfinite(control_values)),
        })

    option3_df = pd.DataFrame(option3_results)

    print("\nMeyer Option 3 results:")
    print(option3_df)


    # ------------------------------------------------------------------------------------------------------------------
    # Merge and save Meyer effect size results
    # ------------------------------------------------------------------------------------------------------------------
    effect_size_df = (
        option1_df
        .merge(option2_df, on="Cluster", how="outer", validate="one_to_one")
        .merge(option3_df, on="Cluster", how="outer", validate="one_to_one")
    )

    # save effect size results
    effect_size_table_dir = os.path.join(data_dir, "effect_size_tables")
    os.makedirs(effect_size_table_dir, exist_ok=True)
    effect_size_table_path = os.path.join(effect_size_table_dir, f"{file_suffix}_cohens_d_meyer_options.csv")
    effect_size_df.to_csv(effect_size_table_path, index=False)
    print(f"Saved effect size table:\n{effect_size_table_path}")

    # Merge effect sizes with permutation cluster table
    cluster_table_with_effect_sizes = cluster_table_perm_mass.merge(
        effect_size_df,
        on="Cluster",
        how="left",
        validate="one_to_one"
    )

    # save merged table
    final_cluster_table_path = os.path.join(data_dir, "cluster_tables",
                                            f"{file_suffix}_cluster_table_perm_mass_with_effect_sizes.csv")
    cluster_table_with_effect_sizes.to_csv(final_cluster_table_path, index=False)
    print(f"Saved permutation cluster table with effect sizes:\n{final_cluster_table_path}")


    # --- Clean up memory after each seed
    plt.close("all")

    for name in [
        "t_img",
        "t_data",
        "cluster_img",
        "cluster_data",
        "analysis_mask_img",
        "analysis_mask_data",
        "cohens_d_data",
        "cohens_d_img",
        "subject_img",
        "subject_data",
        "subject_cluster_means_df",
        "subject_bbox_means_df",
        "option1_df",
        "option2_df",
        "option3_df",
        "effect_size_df",
        "cluster_table_with_effect_sizes",
    ]:
        if name in globals():
            del globals()[name]

    gc.collect()