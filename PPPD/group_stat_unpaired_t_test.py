import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_mask_filename,
                  get_output_path, define_group_comparison, get_selected_subject_list, get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import (get_cluster_table_with_aal_labels, get_cluster_table_with_juelich_prob_labels)
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings
import gc


# --- Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seeds = ["OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R"] # List of supported seeds:
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
# mask settings
mask_strategy = "subject_based" # supported strategies: "subject_based", "predefined"
predefined_mask = "vvn" # supported masks: "dmn", "vvn"
threshold_mask = 0.8 # only used if mask_strategy == "subject_based"
# number of permutations for non-parametric cluster-based permutation test
n_perm = 10000


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)


# --- Group mapping for contrast via predefined comparison strategy:
group_mapping = define_group_comparison(group_comparison)


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"=== Running seed: {seed} ===")

    # get output path
    output_dir = get_output_path(part, feature, seed)


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
    file_suffix = f"{file_suffix}_{part_label}"


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


    # --- PARAMETRIC TESTS with different versions of threshold and correction for multiple comparisons:
    # fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
    second_level_model_unpaired = SecondLevelModel(mask_img=analysis_mask)
    second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
    z_map = second_level_model_unpaired.compute_contrast("group", output_type='z_score')

    # save statistical output map
    # save_z_map = os.path.join(output_dir, "stat_maps", f"z_map_{file_suffix}.nii.gz")
    # z_map.to_filename(save_z_map)

    # Version 1: abs(z) > 3.09 (equivalent to p < 0.001 one-sided test), cluster size > 10 voxels
    # z(threshold)=3.09 for p=0.001 when testing one-sided; 3.29 for two-sided
    thresholded_map1 = threshold_img(z_map, threshold=3.09, cluster_threshold=10, two_sided=False)
    thr1_data = thresholded_map1.get_fdata()
    # plot thresholded maps if there are any voxels/clusters left
    if np.any(thr1_data != 0):
        plot_stat_map(thresholded_map1, display_mode='mosaic', cmap="inferno", threshold=3.09, vmin=3.09,
                      title=f"z map {base_title}; z > 3.09; clusters > 10 voxels")
        plt.show()
        fig = plt.figure(figsize=(9,5))
        display = plot_glass_brain(thresholded_map1, cmap="inferno", threshold=3.09, vmin=3.09,
                                   figure=fig, title=None)
        display.frame_axes.figure.suptitle(f"z map {base_title}; z > 3.09; clusters > 10 voxels")
        display.savefig(os.path.join(output_dir, "01_uncorrected", f"{file_suffix}_uncorrected_p001_cluster10.png"))
    else:
        print("No suprathreshold clusters; skipping plots.")
    # get cluster table with anatomical labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster_table1 = get_cluster_table_with_aal_labels(
            stat_img=thresholded_map1,
            stat_threshold=3.09,
            cluster_threshold=10,
            two_sided=False
        )
    # create model report as html regardless if suprathreshold clusters are left or not
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_v1 = second_level_model_unpaired.generate_report(
            contrasts="group",
            title=f"GLM report | {base_title} | z > 3.09, cluster > 10",
            height_control=None,
            threshold=3.09,
            cluster_threshold=10,
            two_sided=False,
            plot_type="glass",
        )
    report_v1.save_as_html(os.path.join(output_dir, "01_uncorrected", f"glm_report_{file_suffix}_uncorrected_p001_cluster10.html"))


    # --- NON-PARAMETRIC TESTS: permutation inference with cluster-level correction:
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
            two_sided_test=False,
            threshold=0.001,         # cluster-forming threshold in p-scale
            random_state=42,
            n_jobs=8,                # adapt to your system
            verbose=1,
        )

    # Convert corrected -log10(p) maps into thresholded views (corrected p < .05  <=>  -log10(p) > 1.30103)
    neglog_alpha_05 = -np.log10(0.05)
    logp_mass_thr = threshold_img(perm_out["logp_max_mass"], threshold=neglog_alpha_05, two_sided=False)

    # Check if there are any significant cluster
    has_sig_clusters = np.any(logp_mass_thr.get_fdata() > neglog_alpha_05)

    if not has_sig_clusters:
        print("No clusters survive permutation cluster-mass FWER correction.")
    else:
        # Save significant cluster mask
        perm_mask_dir = os.path.join(output_dir, "tresh_cluster_masks")
        os.makedirs(perm_mask_dir, exist_ok=True)
        logp_mass_path = os.path.join(perm_mask_dir, f"{file_suffix}_logp_clustermass_fwer05.nii.gz")
        logp_mass_thr.to_filename(logp_mass_path)

        # Plot cluster-mass corrected map
        data = logp_mass_thr.get_fdata()
        visible = data[data > neglog_alpha_05]
        if len(visible) > 0:
            vmax = np.ceil(np.max(visible) * 10) / 10
        else:
            vmax = neglog_alpha_05 + 0.1
        fig = plt.figure(figsize=(9, 5))
        display = plot_glass_brain(logp_mass_thr, cmap="inferno", threshold=neglog_alpha_05, vmin=neglog_alpha_05, vmax=vmax,
                                   figure=fig, title=None, colorbar=True)
        display.frame_axes.figure.suptitle(f"Permutation test cluster-mass FWER \n {base_title} | corrected p < .05")
        display.savefig(os.path.join(output_dir, "04_nonparametric", f"{file_suffix}_perm_clustermass_fwer05.png"))


        # --- Get cluster table with aal and julich brain atlas
        # create a binary/significant map from cluster-size corrected output
        logp_mass_thr_float = math_img("img.astype(float)", img=perm_out["logp_max_mass"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cluster_table_perm_mass, label_maps = get_cluster_table_with_aal_labels(
                stat_img=logp_mass_thr_float,
                stat_threshold=neglog_alpha_05,   # binary image after math_img
                cluster_threshold=0,
                two_sided=False,
                return_label_maps=True,
            )
        if cluster_table_perm_mass.empty:
            print("Cluster table is empty despite significant voxels. Skipping saves.")
        else:
            label_maps = label_maps[0]
            cluster_table_perm_mass["p-value"] = (10 ** (-cluster_table_perm_mass["Peak Stat"]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_table_perm_mass_juelich = get_cluster_table_with_juelich_prob_labels(
                    stat_img=logp_mass_thr_float,
                    stat_threshold=neglog_alpha_05,
                    cluster_threshold=0,
                    two_sided=False,
                    atlas_name="prob-2mm",
                    top_n=5,
                    min_prob=5.0
                )

            # combine tables with aal and juelich label
            cols_aal = ["Cluster ID", "X", "Y", "Z", "Peak Stat", "p-value", "Cluster Size (mm3)", "aal_label", "distance_mm"]
            cols_juelich = ["Cluster ID", "juelich_top_probs"]
            cluster_table_combined = (cluster_table_perm_mass[cols_aal]
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
            for lm in label_maps:
                lm_path = os.path.join(posthoc_mask_dir, f"{file_suffix}_cluster_id_map.nii.gz")
                lm.to_filename(lm_path)
                print(f"Saved cluster ID map: {lm_path}")



    # --- Clean up memory after each seed
    # plt.close("all")
    vars_to_delete = ["derivative_nii", "included_subjects", "sub_mask_imgs", "analysis_mask",
                      "second_level_model_unpaired", "z_map", "thresholded_map1", "thr1_data", "perm_out",
                      "logp_mass_thr", "logp_mass_thr_float", "cluster_table1", "report_v1", "cluster_table_perm_mass",
                      "cluster_table_perm_mass_juelich", "cluster_table_combined", "label_maps", "lm"]
    for var in vars_to_delete:
        if var in locals():
            del locals()[var]
    gc.collect()