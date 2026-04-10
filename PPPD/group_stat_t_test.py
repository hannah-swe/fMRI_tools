import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import _get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_mask_filename, _get_output_path
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import threshold_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
from nilearn.reporting import make_glm_report
import numpy as np
import pandas as pd


# CONFIG:
task = "rest"
run = "run-01"
feature = "falff" # supported features: "falff", "seed_based"
'''List of supported seeds: "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                            "InsulaOP3RAnat", "InsulaOP3Sphere",
                            "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                            "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                            "Precuneus" '''
seed = None
threshold_mask = 0.8

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
mask_imgs = []

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

    # get mask images for the group mask
    mask_filename = _get_mask_filename(subject_id, task, run, feature, seed)
    mask = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", mask_filename)
    if not os.path.exists(mask):
        print(f"Missing mask: {mask}")
        continue
    try:
        nib.load(mask)
        mask_imgs.append(mask)
    except Exception as e:
        print(f"Error loading mask {mask}: {e}")
        continue

print("Loaded images:", len(derivative_nii))
print("Loaded masks:", len(mask_imgs))

# get group mask
group_mask = intersect_masks(mask_imgs, threshold=threshold_mask)


# two sample t-test unpaired (control vs. patient)
# get design matrix and plot it
design_df = participants_df[participants_df["subject_id"].isin(included_subjects)].copy()
design_df = design_df.set_index("subject_id").loc[included_subjects].reset_index()
group_contrast = design_df["group"].map({
    "patient": 1,
    "control": -1
}).values
unpaired_design_matrix = pd.DataFrame({
        "intercept": np.ones(len(group_contrast)),
        "group": group_contrast
})
print(len(derivative_nii))
print(unpaired_design_matrix.shape)
plot_design_matrix(unpaired_design_matrix)
plt.show()

# fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
second_level_model_unpaired = SecondLevelModel(mask_img=group_mask)
second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
z_map = second_level_model_unpaired.compute_contrast("group", output_type='z_score')

# define the file suffix
if feature == "seed_based":
    base_title = f"{feature} {seed}"
    file_suffix = f"{feature}_{seed}"
else:
    base_title = f"{feature}"
    file_suffix = f"{feature}"


# Version 1: abs(z) > 3.29 (equivalent to p < 0.001), cluster size > 10 voxels
thresholded_map1 = threshold_img(z_map, threshold=3.29, cluster_threshold=10, two_sided=False)
plot_stat_map(thresholded_map1, display_mode='mosaic', cmap="inferno", threshold=3.29, vmin=3.29,
              title=f"z map {base_title}; z > 3.29; clusters > 10 voxels")
plt.show()
plot_glass_brain(thresholded_map1, cmap="inferno", threshold=3.29, vmin=3.29,
                 title=f"z map {base_title}; z > 3.29; clusters > 10 voxels")
plt.show()
report_v1 = make_glm_report(model=second_level_model_unpaired, contrasts="group",
                            title=f"GLM report | {base_title} | z > 3.29, cluster > 10", height_control=None,
                            threshold=3.29, cluster_threshold=10, two_sided=False, plot_type="glass")
report_v1.save_as_html(os.path.join(output_dir, f"glm_report_{file_suffix}_uncorrected_p001_cluster10.html"))



# Version 2: thresholding z-statistic image with a false positive rate < .001, cluster size > 10 voxels
thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=0.001, height_control="fpr", cluster_threshold=10, two_sided=False)
plot_stat_map(thresholded_map2, display_mode='mosaic', cmap="inferno", threshold=threshold2, vmin=threshold2,
              title=f"z map {base_title}; fpr < .001; clusters > 10 voxels")
plt.show()
plot_glass_brain(thresholded_map2, cmap="inferno", threshold=threshold2, vmin=threshold2,
                 title=f"z map {base_title}; fpr < .001; clusters > 10 voxels")
plt.show()


# Version 3: FDR <.05 (False Discovery Rate) and no cluster-level threshold
thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=0.05, height_control="fdr")
print(f"The FDR=.05 threshold is {threshold3:.3g}")
plot_stat_map(thresholded_map3, display_mode='mosaic', cmap="inferno", threshold=threshold3, vmin=threshold3,
              title=f"z map {base_title}; fdr < .05")
plt.show()
plot_glass_brain(thresholded_map3, cmap="inferno", threshold=threshold3, vmin=threshold3,
                 title=f"z map {base_title}; fdr < .05")
plt.show()


# Version 4: FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold
thresholded_map4, threshold4 = threshold_stats_img(z_map, alpha=0.05, height_control="bonferroni")
print(f"The p<.05 Bonferroni-corrected threshold is {threshold4:.3g}")
plot_stat_map(thresholded_map4, display_mode='mosaic', cmap="inferno", threshold=threshold4, vmin=threshold4,
              title=f"z map {base_title}; fwer < .05")
plt.show()
plot_glass_brain(thresholded_map4, cmap="inferno", threshold=threshold4, vmin=threshold4,
                 title=f"z map {base_title}; fwer < .05")
plt.show()
report_v4 = make_glm_report(model=second_level_model_unpaired, contrasts="group",
                            title=f"GLM report | {base_title} | Bonferroni FWER < .05",
                            height_control="bonferroni", alpha=0.05, two_sided=False)
report_v4.save_as_html(os.path.join(output_dir, f"glm_report_{file_suffix}_bonferroni_fwer05.html"))



# TODO: save outputs
# save group statistic map
# out_file = "/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/group_level/group_statmap.nii.gz"
# nib.save(map_group, out_file)
