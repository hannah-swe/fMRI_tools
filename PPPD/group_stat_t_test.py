import os
import sys
import nibabel as nib
from PPPD import _get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename
from PPPD.subjects import subs, subjects_to_exclude
from nibabel import load
from nilearn import plotting
from nilearn import datasets
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, plot_design_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG:
task = "rest"
run = "run-01"
feature = "seed_based"
seed = "Precuneus"

# path to halfpipe derivatives directory
base_dir = _get_data_path(feature)

# read participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get derivatives path
deriv_dir = _get_derivatives_path(feature)

# read subjects' derivatives data
derivative_nii = []
included_subjects = []

for s in subs:
    subject_id = f"sub-{s:03d}"
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

print("Loaded images:", len(derivative_nii))

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
# fit model and plot output
second_level_model_unpaired = SecondLevelModel()
second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
stat_map_unpaired = second_level_model_unpaired.compute_contrast("group", output_type='stat')
plot_stat_map(stat_map_unpaired, display_mode='mosaic', cmap="inferno", threshold=2)
plt.show()


# two sample t-test paired

# one-sample t-test
# Design matrix for second-level analysis: 1 for each subject (single-group design)
# design_matrix = np.ones((n_subjects, 1))  # All subjects contribute to the same condition
design_matrix = pd.DataFrame(np.ones((n_subjects, 1)), columns=["intercept"])
print(f'Design matrix shape: {design_matrix.shape}')
plot_design_matrix(design_matrix)
plt.show()

# Second-level GLM
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(derivative_nii, design_matrix=design_matrix)

# Contrast: Testing the mean effect across subjects
contrast = np.array([1])  # One-sample t-test (testing the constant regressor)
map_group = second_level_model.compute_contrast(contrast,
                                                output_type='stat')  # ['z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all']

# save group statistic map
out_file = "/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/group_level/group_statmap.nii.gz"
nib.save(map_group, out_file)
# plots
plot_stat_map(map_group, title="Second-level analysis", display_mode='mosaic', cmap="inferno", threshold=3)
plt.show()
