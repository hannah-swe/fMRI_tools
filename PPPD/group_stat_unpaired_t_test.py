import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_mask_filename,
                  _get_output_path, _get_mask_file, _get_cluster_table_with_aal_labels, _get_cluster_table_with_juelich_prob_labels)
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings


# --- Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "InsulaOP3RAnat" # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus"
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
# mask settings
mask_strategy = "subject_based" # supported strategies: "subject_based", "predefined"
predefined_mask = "vvn" # supported masks: "dmn", "vvn"
threshold_mask = 0.8 # only used if mask_strategy == "subject_based"
# number of permutations for non-parametric cluster-based permutation test
n_perm = 1000


# --- Define the file suffix
if feature == "seed_based":
    base_title = f"{feature} {seed}; {group_comparison}"
    file_suffix = f"{feature}_{seed}_{group_comparison}"
else:
    base_title = f"{feature}; {group_comparison}"
    file_suffix = f"{feature}_{group_comparison}"
if mask_strategy == "subject_based":
    mask_label = f"submask-{threshold_mask}"
    base_title = f"{base_title}; subject mask {threshold_mask}"
else:
    mask_label = predefined_mask
    base_title = f"{base_title}; mask {predefined_mask}"
file_suffix = f"{file_suffix}_{mask_label}"


# --- Get all directories and participants.tsv:
# path to halfpipe derivatives directory
base_dir = _get_data_path(feature)

# read participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get derivatives path
deriv_dir = _get_derivatives_path(feature)

# get output path
output_dir = _get_output_path(part, feature)


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
derivative_nii = []
included_subjects = []
sub_mask_imgs = []

# read subjects' derivatives data
for s in subs:
    # exclude subjects
    if s in subjects_to_exclude:
        continue

    # get full subject id
    subject_id = f"sub-{s:03d}"

    # load statistical nifti maps
    filename = _get_full_filename(subject_id, task, run, feature, seed)
    img = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", filename)
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
        sub_mask_filename = _get_mask_filename(subject_id, task, run, feature, seed)
        sub_mask = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", sub_mask_filename)
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
    analysis_mask = _get_mask_file(predefined_mask)
else:
    raise ValueError(f"Unknown mask strategy {mask_strategy}")


# --- Group mapping for contrast via predefined comparison strategy:
if group_comparison == "pat>HC":
    group_mapping = {
        "patient": 1,
        "control": -1}
elif group_comparison == "HC>pat":
    group_mapping = {
        "control": 1,
        "patient": -1}
else:
    raise ValueError(f"Unknown group comparison: {group_comparison}")


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
    cluster_table1 = _get_cluster_table_with_aal_labels(
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

'''
# Version 2: thresholding z-statistic image with a false discovery rate < .05, no cluster-level threshold
thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=0.05, height_control="fdr", two_sided=False)
print(f"The FDR=.05 threshold is {threshold2:.3g}")
thr2_data = thresholded_map2.get_fdata()
if np.any(thr2_data != 0):
    plot_stat_map(thresholded_map2, display_mode='mosaic', cmap="inferno", threshold=threshold2, vmin=threshold2,
                  title=f"z map {base_title}; fdr < .05")
    plt.show()
    fig = plt.figure(figsize=(9, 5))
    display = plot_glass_brain(thresholded_map2, cmap="inferno", threshold=threshold2, vmin=threshold2,
                               figure=fig, title=None)
    display.frame_axes.figure.suptitle(f"z map {base_title}; fdr < .05")
    display.savefig(os.path.join(output_dir, "02_fdr", f"{file_suffix}_fdr05.png"))
else:
    print("No suprathreshold clusters; skipping plots.")
# get cluster table with anatomical labels
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cluster_table2 = _get_cluster_table_with_aal_labels(
        stat_img=thresholded_map2,
        stat_threshold=threshold2,
        two_sided=False
    )
# create model report as html regardless if suprathreshold clusters are left or not
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report_v2 = second_level_model_unpaired.generate_report(
        contrasts="group",
        title=f"GLM report | {base_title} | fdr < .05",
        height_control="fdr",
        two_sided=False,
        plot_type="glass",
    )
report_v2.save_as_html(os.path.join(output_dir, "02_fdr", f"glm_report_{file_suffix}_fdr05.html"))
'''

'''
# Version 3: FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold
thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=0.05, height_control="bonferroni")
print(f"The p<.05 Bonferroni-corrected threshold is {threshold3:.3g}")
thr3_data = thresholded_map3.get_fdata()
# plot thresholded maps if there are any voxels/clusters left
if np.any(thr3_data != 0):
    plot_stat_map(thresholded_map3, display_mode='mosaic', cmap="inferno", threshold=threshold3, vmin=threshold3,
                  title=f"z map {base_title}; fwer < .05")
    plt.show()
    display = plot_glass_brain(thresholded_map3, cmap="inferno", threshold=threshold3, vmin=threshold3,
                               title=f"z map {base_title}; fwer < .05")
    display.savefig(os.path.join(output_dir, "03_bonferroni", f"{file_suffix}_bonferroni_fwer05.png"))
else:
    print("No voxels survive Bonferroni correction; skipping plots.")
# get cluster table with anatomical labels
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cluster_table3 = _get_cluster_table_with_aal_labels(
        stat_img=thresholded_map3,
        stat_threshold=threshold3,
        two_sided=False
    )
# create model report as html regardless if suprathreshold clusters are left or not
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report_v3 = second_level_model_unpaired.generate_report(
        contrasts="group",
        title=f"GLM report | {base_title} | Bonferroni FWER < .05",
        height_control="bonferroni",
        alpha=0.05,
        two_sided=False,
    )
report_v3.save_as_html(os.path.join(output_dir, "03_bonferroni", f"glm_report_{file_suffix}_bonferroni_fwer05.html"))
'''


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
        n_jobs=40,                # adapt to your system
        verbose=1,
    )
# Save raw permutation outputs
# perm_out["t"].to_filename(os.path.join(output_dir, "04_nonparametric", f"t_map_{file_suffix}_perm.nii.gz"))
# perm_out["logp_max_t"].to_filename(os.path.join(output_dir, "04_nonparametric", f"logp_max_t_{file_suffix}_perm.nii.gz"))
# perm_out["logp_max_size"].to_filename(os.path.join(output_dir, "04_nonparametric", f"logp_max_size_{file_suffix}_perm.nii.gz"))
# perm_out["logp_max_mass"].to_filename(os.path.join(output_dir, "04_nonparametric", f"logp_max_mass_{file_suffix}_perm.nii.gz"))

# Convert corrected -log10(p) maps into thresholded views (corrected p < .05  <=>  -log10(p) > 1.30103)
neglog_alpha_05 = -np.log10(0.05)
logp_size_thr = threshold_img(perm_out["logp_max_size"], threshold=neglog_alpha_05, two_sided=False)
logp_mass_thr = threshold_img(perm_out["logp_max_mass"], threshold=neglog_alpha_05, two_sided=False)
logp_voxel_thr = threshold_img(perm_out["logp_max_t"], threshold=neglog_alpha_05, two_sided=False)

# Plot voxel-level corrected map
if np.any(logp_voxel_thr.get_fdata() != 0):
    fig = plt.figure(figsize=(9, 5))
    display = plot_glass_brain(logp_voxel_thr, cmap="inferno", threshold=neglog_alpha_05, vmin=neglog_alpha_05,
                               figure=fig, title=None, colorbar=True)
    display.frame_axes.figure.suptitle(f"Permutation test voxel-level FWER \n {base_title} | corrected p < .05")
    # display.savefig(os.path.join(output_dir, "04_nonparametric", f"{file_suffix}_perm_voxel_fwer05.png"))
else:
    print("No voxels survive permutation voxel-level FWER correction.")

# Plot cluster-size corrected map
if np.any(logp_size_thr.get_fdata() != 0):
    fig = plt.figure(figsize=(9, 5))
    display = plot_glass_brain(logp_size_thr, cmap="inferno", threshold=neglog_alpha_05, vmin=neglog_alpha_05,
                               figure=fig, title=None, colorbar=True)
    display.frame_axes.figure.suptitle(f"Permutation test cluster-size FWER \n {base_title} | corrected p < .05")
    display.savefig(os.path.join(output_dir, "04_nonparametric", f"{file_suffix}_perm_clustersize_fwer05.png"))
else:
    print("No clusters survive permutation cluster-size FWER correction.")

# --- Plot cluster-mass corrected map
if np.any(logp_mass_thr.get_fdata() != 0):
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
else:
    print("No clusters survive permutation cluster-mass FWER correction.")


# --- Get cluster table with aal and julich brain atlas
# create a binary/significant map from cluster-size corrected output
sig_cluster_size_map = math_img(f"img > {neglog_alpha_05}", img=perm_out["logp_max_mass"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cluster_table_perm_size = _get_cluster_table_with_aal_labels(
        stat_img=sig_cluster_size_map,
        stat_threshold=0.5,   # binary image after math_img
        cluster_threshold=0,
        two_sided=False,
    )
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cluster_table_perm_size_juelich = _get_cluster_table_with_juelich_prob_labels(
        stat_img=sig_cluster_size_map,
        stat_threshold=0.5,
        cluster_threshold=0,
        two_sided=False,
        atlas_name="prob-2mm",
        top_n=5,
        min_prob=0.0
    )
cols_aal = ["Cluster ID", "X", "Y", "Z", "Cluster Size (mm3)", "aal_label", "distance_mm"]
cols_juelich = ["Cluster ID", "juelich_top_probs"]
cluster_table_combined = (cluster_table_perm_size[cols_aal]
    .merge(
        cluster_table_perm_size_juelich[cols_juelich],
        on="Cluster ID",
        how="left"
    )
    .rename(columns={
        "Cluster ID": "Cluster",
        "Cluster Size (mm3)": "Size (mm3)",
        "juelich_top_probs": "juelich_label"
    })
)
