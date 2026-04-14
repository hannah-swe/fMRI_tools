import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_mask_filename,
                  _get_output_path, _get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import threshold_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
from nilearn.reporting import make_glm_report
import numpy as np
import pandas as pd
import warnings


# Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
feature = "seed_based" # supported features: "falff", "seed_based"
seed = "Precuneus" # List of supported seeds: "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus"
group_comparison = "HC>pat" # supported comparisons: "pat>HC", "HC>pat"
# Mask settings:
mask_strategy = "subject_based" # supported strategies: "subject_based", "predefined"
predefined_mask = "dmn" # supported masks: "dmn"
threshold_mask = 0.8 # only used if mask_strategy == "subject_based"


# define the file suffix
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


# path to halfpipe derivatives directory
base_dir = _get_data_path(feature)

# read participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get derivatives path
deriv_dir = _get_derivatives_path(feature)

# get output path
output_dir = _get_output_path(feature)


# read subjects' derivatives data
derivative_nii = []
included_subjects = []
sub_mask_imgs = []

for s in subs:
    if s in subjects_to_exclude:
        continue

    subject_id = f"sub-{s:03d}"

    # get statistical nifti maps
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

    # get subject masks only if subject-based mask strategy is used
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

# get group mask
if mask_strategy == "subject_based":
     print("Loaded subjects' masks:", len(sub_mask_imgs))
     if len(sub_mask_imgs) == 0:
         raise ValueError("No subjects mask found.")
     analysis_mask = intersect_masks(sub_mask_imgs, threshold=threshold_mask)
     save_mask_path = os.path.join(output_dir, "masks", f"group_mask_{file_suffix}.nii.gz")
     analysis_mask.to_filename(save_mask_path)
elif mask_strategy == "predefined":
    if predefined_mask is None:
        raise ValueError("No predefined mask was loaded.")
    analysis_mask = _get_mask_file(predefined_mask)
else:
    raise ValueError(f"Unknown mask strategy {mask_strategy}")


# group mapping for contrast
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

# two sample t-test unpaired (control vs. patient)
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

# fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
second_level_model_unpaired = SecondLevelModel(mask_img=analysis_mask)
second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
z_map = second_level_model_unpaired.compute_contrast("group", output_type='z_score')
save_z_map = os.path.join(output_dir, "stat_maps", f"z_map_{file_suffix}.nii.gz")
z_map.to_filename(save_z_map)


# Version 1: abs(z) > 3.29 (equivalent to p < 0.001), cluster size > 10 voxels
thresholded_map1 = threshold_img(z_map, threshold=3.29, cluster_threshold=10, two_sided=False)
thr1_data = thresholded_map1.get_fdata()
if np.any(thr1_data != 0):
    plot_stat_map(thresholded_map1, display_mode='mosaic', cmap="inferno", threshold=3.29, vmin=3.29,
                  title=f"z map {base_title}; z > 3.29; clusters > 10 voxels")
    plt.show()
    display = plot_glass_brain(thresholded_map1, cmap="inferno", threshold=3.29, vmin=3.29,
                               title=f"z map {base_title}; z > 3.29; clusters > 10 voxels")
    display.savefig(os.path.join(output_dir, f"{file_suffix}_uncorrected_p001_cluster10.png"))
else:
    print("No suprathreshold clusters; skipping plots.")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report_v1 = second_level_model_unpaired.generate_report(
        contrasts="group",
        title=f"GLM report | {base_title} | z > 3.29, cluster > 10",
        height_control=None,
        threshold=3.29,
        cluster_threshold=10,
        two_sided=False,
        plot_type="glass",
    )
report_v1.save_as_html(os.path.join(output_dir, f"glm_report_{file_suffix}_uncorrected_p001_cluster10.html"))


'''# Version 2: thresholding z-statistic image with a false positive rate < .001, cluster size > 10 voxels
thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=0.001, height_control="fpr", cluster_threshold=10, two_sided=False)
plot_stat_map(thresholded_map2, display_mode='mosaic', cmap="inferno", threshold=threshold2, vmin=threshold2,
              title=f"z map {base_title}; fpr < .001; clusters > 10 voxels")
plt.show()
plot_glass_brain(thresholded_map2, cmap="inferno", threshold=threshold2, vmin=threshold2,
                 title=f"z map {base_title}; fpr < .001; clusters > 10 voxels")
plt.show()'''


'''# Version 3: FDR <.05 (False Discovery Rate) and no cluster-level threshold
thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=0.05, height_control="fdr")
print(f"The FDR=.05 threshold is {threshold3:.3g}")
plot_stat_map(thresholded_map3, display_mode='mosaic', cmap="inferno", threshold=threshold3, vmin=threshold3,
              title=f"z map {base_title}; fdr < .05")
plt.show()
plot_glass_brain(thresholded_map3, cmap="inferno", threshold=threshold3, vmin=threshold3,
                 title=f"z map {base_title}; fdr < .05")
plt.show()'''


# Version 4: FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold
thresholded_map4, threshold4 = threshold_stats_img(z_map, alpha=0.05, height_control="bonferroni")
print(f"The p<.05 Bonferroni-corrected threshold is {threshold4:.3g}")
thr4_data = thresholded_map4.get_fdata()
if np.any(thr4_data != 0):
    plot_stat_map(thresholded_map4, display_mode='mosaic', cmap="inferno", threshold=threshold4, vmin=threshold4,
                  title=f"z map {base_title}; fwer < .05")
    plt.show()
    display = plot_glass_brain(thresholded_map4, cmap="inferno", threshold=threshold4, vmin=threshold4,
                               title=f"z map {base_title}; fwer < .05")
    display.savefig(os.path.join(output_dir, f"{file_suffix}_bonferroni_fwer05.png"))
else:
    print("No voxels survive Bonferroni correction; skipping plots.")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report_v4 = second_level_model_unpaired.generate_report(
        contrasts="group",
        title=f"GLM report | {base_title} | Bonferroni FWER < .05",
        height_control="bonferroni",
        alpha=0.05,
        two_sided=False,
    )
report_v4.save_as_html(os.path.join(output_dir, f"glm_report_{file_suffix}_bonferroni_fwer05.html"))
