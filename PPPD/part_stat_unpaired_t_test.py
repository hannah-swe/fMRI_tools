import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_full_filename, get_derivatives_path, get_mask_filename, get_participants_tsv,
                  define_group_comparison, get_selected_subject_list, get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import get_cluster_table_with_aal_labels
from PPPD.plotting import get_colorbar_limits
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
run = "run-01" # pre, post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "falff" # supported features: "falff", "seed_based", "alff"
seeds = ["InsulaId1L"] # List of supported seeds:
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

output_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/part1_vs_part2/"

# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")
participants_df["part"] = participants_df["participant_id"].apply(lambda x: 1 if x < 100 else 2)

# --- Derivatives path used for subject based mask strategy
deriv_dir = get_derivatives_path(feature)

# --- Group mapping for contrast via predefined comparison strategy:
group_mapping = define_group_comparison(group_comparison)


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"\n === Running seed: {seed} === \n")

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


    # --- Load existing diff images and subject masks
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

        # load subject masks from original derivative folders
        run_masks = {}

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


    # --- Get analysis mask
    if mask_strategy == "subject_based":
        print("Loaded subjects' masks:", len(sub_mask_imgs))
        if len(sub_mask_imgs) == 0:
            raise ValueError("No subjects mask found.")
        analysis_mask = intersect_masks(sub_mask_imgs, threshold=threshold_mask, connected=False)
    elif mask_strategy == "predefined":
        if predefined_mask is None:
            raise ValueError("No predefined mask was loaded.")
        analysis_mask = get_mask_file(predefined_mask)
    else:
        raise ValueError(f"Unknown mask strategy {mask_strategy}")


    # --- Get design matrix for pre-post model and plot it
    design_df = participants_df[participants_df["subject_id"].isin(included_subjects)].copy()
    design_df = design_df.set_index("subject_id").loc[included_subjects].reset_index()
    design_df["group_code"] = design_df["group"].map(group_mapping)
    if design_df["group_code"].isna().any():
        print(design_df[design_df["group_code"].isna()][["subject_id", "group"]])
        raise ValueError("Some subjects have missing or unmapped group labels.")
    design_df["part_code"] = design_df["part"].map({
        1: -1,
        2: 1
    })
    design_df["interaction"] = (design_df["group_code"] * design_df["part_code"])
    second_level_design = pd.DataFrame({
        "intercept": np.ones(len(design_df)),
        "group": design_df["group_code"].astype(float),
        "part": design_df["part_code"].astype(float),
        "interaction": design_df["interaction"].astype(float),
    })
    print("Number of images:", len(derivative_nii))
    print("Design matrix shape:", second_level_design.shape)
    # plot_design_matrix(second_level_design)
    # plt.show()


    # --- PARAMETRIC TEST (voxel-wise two sample t-test unpaired)
    # fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
    second_level_model = SecondLevelModel(mask_img=analysis_mask)
    second_level_model = second_level_model.fit(derivative_nii, design_matrix=second_level_design)
    # get interaction group x part
    interaction_z_map = second_level_model.compute_contrast(second_level_contrast="interaction", output_type="z_score")
    # get part main effect
    part_z_map = second_level_model.compute_contrast(second_level_contrast="part", output_type="z_score")

    # Significance test:
    # Version 1: abs(z) > 3.09 (equivalent to p < 0.001 one-sided test), cluster size > 10 voxels
    # z(threshold)=3.09 for p=0.001 when testing one-sided; 3.29 for two-sided
    thresholded_map1 = threshold_img(part_z_map, cluster_threshold=10, threshold=3.09, two_sided=True)
    thr1_data = thresholded_map1.get_fdata()

    # plot thresholded maps if there are any voxels/clusters left
    uncorr_plot_dir = os.path.join(output_dir, "01_uncorrected")
    os.makedirs(uncorr_plot_dir, exist_ok=True)

    if np.any(thr1_data != 0):
        fig = plt.figure(figsize=(9,5))
        display = plot_glass_brain(thresholded_map1, cmap="RdBu_r",
                                   figure=fig, title=None, plot_abs=False, symmetric_cbar=True)
        display.frame_axes.figure.suptitle(f"z map main effect: part \n {base_title}; z > 3.09; clusters > 10 voxels")
        display.savefig(os.path.join(uncorr_plot_dir, f"{file_suffix}_uncorrected_p001_cluster10.png"))
    else:
        print("No suprathreshold clusters; skipping plots.")


    # --- NON-PARAMETRIC TESTS: permutation inference with cluster-level correction:
    # threshold is in p-scale, not z-scale; threshold=0.001 corresponds to a cluster-forming threshold of p < .001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm_out = non_parametric_inference(
            second_level_input=derivative_nii,
            design_matrix=second_level_design,
            second_level_contrast="part",
            mask=analysis_mask,
            model_intercept=False,  # intercept is already in the design matrix
            n_perm=n_perm,
            two_sided_test=True,
            threshold=0.001,  # cluster-forming threshold in p-scale
            random_state=42,  # pseudo randomization to get reproducible results
            n_jobs=8,  # adapt to the system
            verbose=1,
        )

    # convert corrected -log10(p) maps into thresholded views (corrected p < .05  <=>  -log10(p) > 1.30103)
    neglog_alpha_05 = -np.log10(0.05)
    logp_mass_thr = threshold_img(perm_out["logp_max_mass"], threshold=neglog_alpha_05, two_sided=False)
    # use parametric z-map only for the sign/direction
    sign_z_map = part_z_map

    # For two-sided tests, use a signed -log10(p) map so that clusters can be assigned
    # to the positive or negative direction of the contrast.
    signed_logp_mass = math_img("np.sign(t) * logp", t=perm_out["t"], logp=perm_out["logp_max_mass"])
    signed_logp_mass_thr = threshold_img(signed_logp_mass, threshold=neglog_alpha_05, two_sided=True)

    perm_cluster_img = None
    data = signed_logp_mass_thr.get_fdata()
    has_sig_clusters = np.any(np.abs(data) > neglog_alpha_05)

    perm_plot_dir = os.path.join(output_dir, "02_nonparametric")
    os.makedirs(perm_plot_dir, exist_ok=True)

    if not has_sig_clusters:
        print("No clusters survive permutation cluster-mass FWER correction.")
    else:
        print("Clusters survive permutation cluster-mass FWER correction.")
        vmin, vmax = get_colorbar_limits(data=data, threshold=neglog_alpha_05, two_sided=True)
        fig = plt.figure(figsize=(9, 5))
        display = plot_glass_brain(signed_logp_mass_thr, cmap="RdBu_r", threshold=neglog_alpha_05, vmin=vmin, vmax=vmax,
                                   symmetric_cbar=True, plot_abs=False, figure=fig, title=None, colorbar=True)
        display.frame_axes.figure.suptitle(f"Permutation test cluster-mass FWER\nmain effect: part\n"
                                           f"{base_title} | corrected p < .05")
        display.savefig(os.path.join(perm_plot_dir, f"{file_suffix}_perm_clustermass_fwer05.png"))

    perm_cluster_img = signed_logp_mass_thr

    if perm_cluster_img is None:
        print("No significant clusters found. Skipping cluster tables.")
    else:
        perm_cluster_img_float = math_img("img.astype(float)", img=perm_cluster_img)

        # --- Get cluster table with aal brain atlas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cluster_table_perm_mass, label_maps = get_cluster_table_with_aal_labels(
                stat_img=perm_cluster_img_float,
                stat_threshold=neglog_alpha_05,
                cluster_threshold=0,
                two_sided=True,
                return_label_maps=True
            )

        label_maps = label_maps[0]
        print(f"Found {len(label_maps)} label map(s)")

        cluster_table_perm_mass["p-value"] = (10 ** (-np.abs(cluster_table_perm_mass["Peak Stat"])))


        cluster_table_perm_mass = cluster_table_perm_mass.rename(columns={
            "Cluster ID": "Cluster",
            "Peak Stat": "Stat",
            "Cluster Size (mm3)": "Size (mm3)",
        })

    # --- Clean up memory after each seed
    plt.close("all")
    vars_to_delete = [
        "derivative_nii", "included_subjects", "sub_mask_imgs", "analysis_mask", "interaction_z_map", "part_z_map",
        "thresholded_map1", "logp_mass_thr", "signed_logp_mass", "signed_logp_mass_thr", "perm_cluster_img",
        "perm_cluster_img_float", "thr1_data", "data", "perm_out", "second_level_model", "cluster_table_perm_mass",
        "label_maps"
    ]
    for var in vars_to_delete:
        if var in locals():
            del locals()[var]
    gc.collect()