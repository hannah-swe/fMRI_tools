import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_mask_filename,
                  get_output_path, define_group_comparison, get_selected_subject_list, get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import (get_cluster_table_with_aal_labels, get_cluster_table_with_juelich_prob_labels)
from PPPD.config import load_config
from PPPD.plotting import get_colorbar_limits
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img, new_img_like
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings
import gc


# ---- Load script configuration from config.yml --> don't change anything here, make all configurations in config.yml
config = load_config()

task = config["analysis"]["task"]
runs = config["analysis"]["runs"] # "run-01" == pre, "run-02" == post
run = "run-01"
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
                                    # "HippocampusL", "HippocampusR"
                                    # "PrecuneusL", "PrecuneusR"
group_comparison = config["analysis"]["group_comparison"] # supported comparisons: "pat>HC", "HC>pat"
mask_strategy = config["mask"]["strategy"] # supported strategies: "subject_based", "predefined"
predefined_mask = config["mask"]["predefined_mask"]
threshold_mask = config["mask"]["threshold"] # only used if mask_strategy == "subject_based"
n_perm = config["statistics"]["n_perm"] # number of permutations for non-parametric cluster-based permutation test
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
      f"group_comparison = {group_comparison}\nmask_strategy = {mask_strategy}\nn_perm = {n_perm}\n"
      f"t_test_strategy = {t_test_strategy}\ntwo_sided = {two_sided}\n")


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")


# --- Path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)


# --- Group mapping for contrast via predefined comparison strategy:
group_mapping = define_group_comparison(group_comparison)


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"\n=== Running seed: {seed} ===")

    # get output path
    output_dir = get_output_path(part, feature, seed)
    os.makedirs(output_dir, exist_ok=True)


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


    # --- Load data:
    # initialize lists for derivatives, subject ids and mask images
    derivative_nii = []
    included_subjects = []
    sub_mask_imgs = []

    # read subjects' derivatives data
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
            derivative_nii.append(img)
            included_subjects.append(subject_id)
        except Exception as e:
            print(f"Error loading {img}: {e}")
            continue

        # load subject masks only if subject-based mask strategy is used
        if mask_strategy == "subject_based":
            sub_mask_filename = get_mask_filename(subject_id, task, run, feature, seed)
            sub_mask = os.path.join(deriv_dir, subject_id, sub_mask_filename)
            if not os.path.exists(sub_mask):
                print(f"Missing mask: {sub_mask}")
                continue
            try:
                nib.load(sub_mask)
                sub_mask_imgs.append(sub_mask)
            except Exception as e:
                print(f"Error loading mask {sub_mask}: {e}")
                continue

    print("Loaded images:", len(derivative_nii))


    # --- Get analysis mask:
    # compute data based group mask with predefined threshold
    if mask_strategy == "subject_based":
         print("Loaded subjects' masks:", len(sub_mask_imgs))
         if len(sub_mask_imgs) == 0:
             raise ValueError("No subjects mask found.")
         analysis_mask = intersect_masks(sub_mask_imgs, threshold=threshold_mask)
         # save_mask_path = os.path.join(output_dir, "masks", f"group_mask_{file_suffix}.nii.gz")
         # analysis_mask.to_filename(save_mask_path)
    # use predefined mask; cave: resample before to same MNI space HALFpipe is using
    elif mask_strategy == "predefined":
        if predefined_mask is None:
            raise ValueError("No predefined mask was loaded.")
        analysis_mask = get_mask_file(predefined_mask)
    else:
        raise ValueError(f"Unknown mask strategy {mask_strategy}")


    # --- Compute two sample t-test unpaired:
    # get design matrix and plot it
    design_df = participants_df[participants_df["subject_id"].isin(included_subjects)].copy()
    design_df = design_df.set_index("subject_id").loc[included_subjects].reset_index()
    group_contrast = design_df["group"].map(group_mapping).values
    unpaired_design_matrix = pd.DataFrame({
            "intercept": np.ones(len(group_contrast)),
            "group": group_contrast
    })
    print(len(derivative_nii))
    print(unpaired_design_matrix.shape)
    # plot_design_matrix(unpaired_design_matrix)
    # plt.show()



    # _________________________________________________________________________________________________________________
    # PARAMETRIC TESTS with different versions of threshold and correction for multiple comparisons:
    # _________________________________________________________________________________________________________________

    # fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
    second_level_model_unpaired = SecondLevelModel(mask_img=analysis_mask)
    second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
    z_map = second_level_model_unpaired.compute_contrast("group", output_type='z_score')

    # save statistical output map
    # save_z_map = os.path.join(output_dir, "stat_maps", f"z_map_{file_suffix}.nii.gz")
    # z_map.to_filename(save_z_map)

    # Version 1: abs(z) > 3.09 (equivalent to p < 0.001 one-sided test), cluster extent threshold = 10 voxels
    # z(threshold)=3.09 for p=0.001 when testing one-sided; 3.29 for two-sided
    thresholded_map1, z_threshold_p001 = threshold_stats_img(
        z_map,
        alpha=0.001,
        height_control="fpr", # meaning voxelwise uncorrected thresholding
        cluster_threshold=10,
        two_sided=two_sided,
    )
    thr1_data = thresholded_map1.get_fdata()

    if two_sided:
        threshold_label = f"|z| > {z_threshold_p001:.2f}"
    else:
        threshold_label = f"z > {z_threshold_p001:.2f}"

    if two_sided:
        print(f"Uncorrected voxel-wise threshold: |z| > {z_threshold_p001:.4f}")
    else:
        print(f"Uncorrected voxel-wise threshold: z > {z_threshold_p001:.4f}")

    if np.any(thr1_data != 0):
        # save nifti of uncorrected t-test
        save_thresh_map_dir = os.path.join(output_dir, "uncorrected_maps")
        os.makedirs(save_thresh_map_dir, exist_ok=True)

        # Split thresholded image into positive and negative image and save
        # positive image
        if np.any(thr1_data > 0):
            positive_map = new_img_like(thresholded_map1, np.where(thr1_data > 0, thr1_data, 0))
            save_pos_thresh_map = os.path.join(save_thresh_map_dir, f"{file_suffix}_positive_thresh_map.nii.gz")
            positive_map.to_filename(save_pos_thresh_map)
            print("Saved positive cluster map.")
        else:
            print("No positive significant clusters.")

        # negative image
        if np.any(thr1_data < 0):
            negative_map = new_img_like(thresholded_map1, np.where(thr1_data < 0, thr1_data, 0))
            save_neg_thresh_map = os.path.join(save_thresh_map_dir, f"{file_suffix}_negative_thresh_map.nii.gz")
            negative_map.to_filename(save_neg_thresh_map)

            # save additionally an image with the absolute values of negative clusters for visualization only
            negative_display_map = new_img_like(thresholded_map1, np.where(thr1_data < 0, np.abs(thr1_data), 0))
            save_neg_thresh_display_map = os.path.join(save_thresh_map_dir,
                                                       f"{file_suffix}_negative_thresh_display_map.nii.gz")
            negative_display_map.to_filename(save_neg_thresh_display_map)
            print("Saved negative cluster map.")
        else:
            print("No negative significant clusters.")

        # plot thresholded maps if there are any voxels/clusters left
        # get vmin and vmax
        vmin, vmax = get_colorbar_limits(data=thr1_data, threshold=z_threshold_p001, two_sided=two_sided)

        # plot
        plot_stat_map(thresholded_map1, display_mode='mosaic', cmap=cmap, threshold=z_threshold_p001,
                      symmetric_cbar=symmetric_cbar, vmin=vmin, vmax=vmax,
                      title=f"z map {base_title}; {threshold_label}; clusters >= 10 voxels")
        plt.show()
        fig = plt.figure(figsize=(9,5))
        display = plot_glass_brain(thresholded_map1, cmap=cmap, threshold=z_threshold_p001,
                                   symmetric_cbar=symmetric_cbar, vmin=vmin, vmax=vmax,
                                   plot_abs=False, figure=fig, title=None)
        display.frame_axes.figure.suptitle(f"z map {base_title}; {threshold_label}; clusters >= 10 voxels")
        plt.show()
        uncorrected_plot_dir = os.path.join(output_dir, "01_uncorrected")
        os.makedirs(uncorrected_plot_dir, exist_ok=True)
        display.savefig(os.path.join(uncorrected_plot_dir, f"{file_suffix}_uncorrected_p001_cluster10.png"))

        # get cluster table with anatomical labels
        cluster_table1 = get_cluster_table_with_aal_labels(
            stat_img=z_map,
            stat_threshold=z_threshold_p001,
            cluster_threshold=10,
            two_sided=two_sided,
        )
        # get voxel size and add cluster size in number of voxels to the table
        voxel_sizes = z_map.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_sizes)

        cluster_sizes_mm3 = pd.to_numeric(cluster_table1["Cluster Size (mm3)"], errors="coerce")
        cluster_size_voxels = (cluster_sizes_mm3 / voxel_volume_mm3).round().astype("Int64")
        insert_position = cluster_table1.columns.get_loc("Cluster Size (mm3)") + 1
        cluster_table1.insert(insert_position, "Cluster Size (voxels)", cluster_size_voxels)

        # save uncorrected cluster tables
        uncorrected_table_dir = os.path.join(output_dir, "uncorrected_tables")
        os.makedirs(uncorrected_table_dir, exist_ok=True)
        uncorrected_table_path = os.path.join(uncorrected_table_dir, f"{file_suffix}_cluster_table_uncorrected.csv")
        cluster_table1.to_csv(uncorrected_table_path, index=False)

    else:
        print("No suprathreshold clusters; skipping plots.")



    # ----------------------------------------------------------------------------------------------------------------
    # NON-PARAMETRIC TESTS: permutation inference with cluster-level correction:
    # ----------------------------------------------------------------------------------------------------------------
    # threshold is in p-scale, not z-scale; threshold=0.001 corresponds to a cluster-forming threshold of p < .001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm_out = non_parametric_inference(
            second_level_input=derivative_nii,
            design_matrix=unpaired_design_matrix,
            second_level_contrast="group",
            mask=analysis_mask,
            model_intercept=False,   # intercept is already in the design matrix
            n_perm=n_perm,
            two_sided_test=two_sided,
            threshold=0.001,         # cluster-forming threshold in p-scale
            random_state=42,
            n_jobs=8,                # adapt to your system
            verbose=1,
        )

    # Convert corrected -log10(p) maps into thresholded views (corrected p < .05  <=>  -log10(p) > 1.30103)
    neglog_alpha_05 = -np.log10(0.05)
    logp_mass_thr = threshold_img(perm_out["logp_max_mass"], threshold=neglog_alpha_05, two_sided=False)

    perm_plot_dir = os.path.join(output_dir, "02_nonparametric")
    os.makedirs(perm_plot_dir, exist_ok=True)
    perm_mask_dir = os.path.join(output_dir, "tresh_cluster_masks")
    os.makedirs(perm_mask_dir, exist_ok=True)

    perm_cluster_img = None
    has_sig_clusters = False

    # plots for two-sided tests
    if two_sided:
        # For two-sided tests, use a signed -log10(p) map so that clusters can be assigned
        # to the positive or negative direction of the contrast.
        signed_logp_mass = math_img("np.sign(t) * logp", t=perm_out["t"], logp=perm_out["logp_max_mass"])
        signed_logp_mass_thr = threshold_img(signed_logp_mass, threshold=neglog_alpha_05, two_sided=True)

        data = signed_logp_mass_thr.get_fdata()
        has_sig_clusters = np.any(np.abs(data) > neglog_alpha_05)

        if not has_sig_clusters:
            print("No clusters survive permutation cluster-mass FWER correction.")
        else:
            signed_logp_mass_path = os.path.join(
                perm_mask_dir,
                f"{file_suffix}_signed_logp_clustermass_fwer05.nii.gz"
            )
            signed_logp_mass_thr.to_filename(signed_logp_mass_path)

            vmin, vmax = get_colorbar_limits(data=data, threshold=neglog_alpha_05, two_sided=True)

            fig = plt.figure(figsize=(9, 5))
            display = plot_glass_brain(signed_logp_mass_thr, cmap=cmap, threshold=neglog_alpha_05, vmin=vmin, vmax=vmax,
                                       symmetric_cbar=True, plot_abs=False, figure=fig, title=None, colorbar=True)
            display.frame_axes.figure.suptitle(f"Permutation test cluster-mass FWER\n"
                                               f"{base_title} | corrected p < .05")
            display.savefig(os.path.join(perm_plot_dir,
                                         f"{file_suffix}_perm_clustermass_fwer05.png"))

        perm_cluster_img = signed_logp_mass_thr

    # plots for one-sided tests
    else:
        # For one-sided tests, the direction is already determined by the contrast.
        # Therefore, plot the unsigned corrected -log10(p) map.
        logp_mass_thr = threshold_img(perm_out["logp_max_mass"], threshold=neglog_alpha_05, two_sided=False)

        data = logp_mass_thr.get_fdata()
        has_sig_clusters = np.any(data > neglog_alpha_05)

        if not has_sig_clusters:
            print("No clusters survive permutation cluster-mass FWER correction.")
        else:
            logp_mass_path = os.path.join(
                perm_mask_dir,
                f"{file_suffix}_logp_clustermass_fwer05.nii.gz"
            )
            logp_mass_thr.to_filename(logp_mass_path)

            vmin, vmax = get_colorbar_limits(data=data, threshold=neglog_alpha_05, two_sided=False)

            fig = plt.figure(figsize=(9, 5))
            display = plot_glass_brain(logp_mass_thr, cmap=cmap, threshold=neglog_alpha_05, vmin=vmin, vmax=vmax,
                                       figure=fig, title=None, colorbar=True)
            display.frame_axes.figure.suptitle(f"Permutation test cluster-mass FWER\n"
                                               f"{base_title} | corrected p < .05")
            display.savefig(os.path.join(perm_plot_dir,
                                         f"{file_suffix}_perm_clustermass_fwer05.png"))

        perm_cluster_img = logp_mass_thr


    # --- Get cluster table with aal and julich brain atlas
    if perm_cluster_img is None:
        print("No significant clusters found. Skipping cluster tables.")
    else:
        perm_cluster_img_float = math_img("img.astype(float)", img=perm_cluster_img)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cluster_table_perm_mass, label_maps = (
                get_cluster_table_with_aal_labels(
                    stat_img=perm_cluster_img_float,
                    stat_threshold=neglog_alpha_05,
                    cluster_threshold=0,
                    two_sided=two_sided,
                    return_label_maps=True,
                )
            )

        if cluster_table_perm_mass.empty:
            print("Cluster table is empty despite significant voxels. Skipping saves.")
        else:
            print(f"Found {len(label_maps)} label map(s).")

            cluster_table_perm_mass["p-value"] = (10 ** (-np.abs(cluster_table_perm_mass["Peak Stat"])))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                cluster_table_perm_mass_juelich = get_cluster_table_with_juelich_prob_labels(
                    stat_img=perm_cluster_img_float,
                    stat_threshold=neglog_alpha_05,
                    cluster_threshold=0,
                    two_sided=two_sided,
                    atlas_name="prob-2mm",
                    top_n=5,
                    min_prob=5.0
                )

            # combine tables with aal and juelich label
            cols_aal = [
                "Cluster ID",
                "X", "Y", "Z",
                "Peak Stat", "p-value",
                "Cluster Size (mm3)",
                "aal_label", "distance_mm"
            ]
            cols_juelich = ["Cluster ID", "juelich_top_probs"]

            cluster_table_combined = (
                cluster_table_perm_mass[cols_aal]
                .merge(
                    cluster_table_perm_mass_juelich[cols_juelich],
                    on="Cluster ID",
                    how="left"
                )
                .rename(columns={
                    "Cluster ID": "Cluster",
                    "Peak Stat": "Stat",
                    "Cluster Size (mm3)": "Size (mm3)",
                    "juelich_top_probs": "juelich_label"
                })
            )

            # save cluster table
            cluster_table_dir = os.path.join(output_dir, "cluster_tables")
            os.makedirs(cluster_table_dir, exist_ok=True)

            cluster_table_path = os.path.join(cluster_table_dir, f"{file_suffix}_cluster_table_perm_mass.csv")
            cluster_table_combined.to_csv(cluster_table_path, index=False)

            # save significant cluster mask as true cluster-ID map
            posthoc_mask_dir = os.path.join(output_dir, "sig_cluster_masks")
            os.makedirs(posthoc_mask_dir, exist_ok=True)

            if cluster_table_perm_mass.empty:
                print("Cluster table is empty despite significant voxels. Skipping saves.")

            elif len(label_maps) == 0:
                print("No label map returned despite non-empty cluster table.")

            else:
                # one label map containing all clusters as integer IDs
                lm = label_maps[0]

                posthoc_mask_dir = os.path.join(output_dir, "sig_cluster_masks")
                os.makedirs(posthoc_mask_dir, exist_ok=True)

                lm_path = os.path.join(posthoc_mask_dir, f"{file_suffix}_cluster_id_map.nii.gz")

                lm.to_filename(lm_path)
                print(f"Saved cluster ID map: {lm_path}")



    # --- Clean up memory after each seed
    # Close nilearn/matplotlib plots first
    try:
        display.close()
    except (NameError, AttributeError):
        pass
    plt.close("all")

    # Large numpy / nifti / nilearn objects
    for name in [
        "data",
        "thr1_data",
        "perm_out",
        "second_level_model_unpaired",
        "z_map",
        "thresholded_map1",
        "logp_mass_thr",
        "signed_logp_mass",
        "signed_logp_mass_thr",
        "perm_cluster_img",
        "perm_cluster_img_float",
        "analysis_mask",
        "label_maps",
        "lm",
    ]:
        if name in globals():
            del globals()[name]

    gc.collect()