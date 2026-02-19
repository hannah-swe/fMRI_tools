#!/usr/bin/env python3
# perform group level analysis of rs-fMRI derivatives
import argparse
import os
import sys
from bids import BIDSLayout
import nibabel as nib
from nibabel import load
from nilearn import plotting
from nilearn import datasets
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, plot_design_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# path to halfpipe derivatives directory
base_dir = "/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/derivatives/halfpipe"

# read participants.tsv
participants_tsv = "/data_wgs04/ag-sensomotorik/PPPD/data/part2_pre/participants.tsv"
if not os.path.exists(participants_tsv):
    print("File does not exist: participants.tsv.")
    sys.exit(1)
df = pd.read_csv(participants_tsv, sep="\t")
df = df.drop(df[df["participant_id"] == 144].index)

# read subjects' derivatives data
sub = sorted(os.listdir(base_dir))
derivative_nii = []
for s in sub:
    img = os.path.join(base_dir, s, "func", "task-rest",
                       f"{s}_task-rest_run-01_feature-seedPrecNoscrub_seed-Precuneus_stat-effect_statmap.nii.gz")
    if not os.path.exists(img):
        print(f"Missing file: {img}")
        continue
    try:
        nib.load(img)
        derivative_nii.append(img)
    except Exception as e:
        print(f"Error loading {img}: {e}")
        continue
n_subjects = len(derivative_nii)
print("Loaded images:", n_subjects)


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


# two sample t-test unpaired (control vs. patient)
# two-sample t-test
group_contrast = df["group"].map({
    "patient": 1,
    "control": -1
}).values
unpaired_design_matrix = pd.DataFrame(
    {
        "intercept": np.ones(len(group_contrast)),
        "group": group_contrast
    }
)
plot_design_matrix(unpaired_design_matrix)
plt.show()

second_level_model_unpaired = SecondLevelModel()
second_level_model_unpaired = second_level_model_unpaired.fit(derivative_nii, design_matrix=unpaired_design_matrix)
stat_maps_unpaired = second_level_model_unpaired.compute_contrast("group", output_type='stat')
plot_stat_map(stat_maps_unpaired, display_mode='mosaic', cmap="inferno", threshold=2)
plt.show()


# two sample t-test paired
