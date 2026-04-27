import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_mask_filename,
                  _get_output_path, _get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings
import tempfile

# ---Script configuration:
task = "rest"
runs = ["run-01", "run-02"] # pre, post
part = 1 # supported: 1, 2 (part 1: subjects < 100; part 2: subjects >= 100)
feature = "falff" # supported features: "falff", "seed_based", "alff"
seed = None # List of supported seeds:
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


# --- Get all directories and participants.tsv:
# path to halfpipe derivatives directory
base_dir = _get_data_path(feature)

# read participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

# get derivatives path
deriv_dir = _get_derivatives_path(feature)

# get output path
output_dir = _get_output_path(feature)


# -- Choose subjects depending on experimental part:
if part == 1:
    selected_subs = [s for s in subs if s < 100]
elif part == 2:
    selected_subs = [s for s in subs if s >= 100]
else:
    raise ValueError("part must be either 1 or 2")
selected_subs = [s for s in selected_subs if s not in subjects_to_exclude]


# --- Load data:
# initialize lists for derivatives, subject ids and mask images
diff_imgs = []
included_subjects = []
sub_mask_imgs = []

diff_output_dir = os.path.join(output_dir, "pre_post_diff")
os.makedirs(diff_output_dir, exist_ok=True)

# read subjects' derivatives data
for s in selected_subs:

    subject_id = f"sub-{s:03d}"

    run_imgs = {}
    run_masks = {}

    for run in runs:

        filename = _get_full_filename(subject_id, task, run, feature, seed)
        img_path = os.path.join(
            deriv_dir,
            subject_id,
            "func",
            f"task-{task}",
            filename
        )

        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue

        try:
            nib.load(img_path)
            run_imgs[run] = img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        if mask_strategy == "subject_based":
            mask_filename = _get_mask_filename(subject_id, task, run, feature, seed)
            mask_path = os.path.join(
                deriv_dir,
                subject_id,
                "func",
                f"task-{task}",
                mask_filename
            )

            if not os.path.exists(mask_path):
                print(f"Missing mask: {mask_path}")
                continue

            try:
                nib.load(mask_path)
                run_masks[run] = mask_path
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                continue

    # only keep subjects with both runs
    if "run-01" not in run_imgs or "run-02" not in run_imgs:
        print(f"Skipping {subject_id}: incomplete pre/post data")
        continue

    # post - pre
    diff_img = math_img(
        "post - pre",
        post=run_imgs["run-02"],
        pre=run_imgs["run-01"]
    )

    diff_filename = f"{subject_id}_task-{task}_{feature}_post-minus-pre.nii.gz"
    diff_path = os.path.join(diff_output_dir, diff_filename)
    diff_img.to_filename(diff_path)

    diff_imgs.append(diff_path)
    included_subjects.append(subject_id)

    if mask_strategy == "subject_based":
        if "run-01" in run_masks and "run-02" in run_masks:
            sub_mask_imgs.extend([run_masks["run-01"], run_masks["run-02"]])
        else:
            print(f"Warning: incomplete masks for {subject_id}")

included_df = pd.DataFrame(included_rows)
print("Loaded images:", len(derivative_nii))
print("Loaded subjects:", included_df["subject_id"].nunique())
print(included_df.groupby("run")["subject_id"].nunique())


# --- Get analysis mask:
# compute data based group mask with predefined threshold
if mask_strategy == "subject_based":
     print("Loaded subjects' masks:", len(sub_mask_imgs))
     if len(sub_mask_imgs) == 0:
         raise ValueError("No subjects mask found.")
     analysis_mask = intersect_masks(sub_mask_imgs, threshold=threshold_mask, connected=False)
     # save_mask_path = os.path.join(output_dir, "masks", f"group_mask_{file_suffix}.nii.gz")
     # analysis_mask.to_filename(save_mask_path)
# use predefined mask; cave: resample before to same MNI space HALFpipe is using
elif mask_strategy == "predefined":
    if predefined_mask is None:
        raise ValueError("No predefined mask was loaded.")
    analysis_mask = _get_mask_file(predefined_mask)
else:
    raise ValueError(f"Unknown mask strategy {mask_strategy}")


# --- Get design matrix for pre-post / group model
design_df = participants_df[
    participants_df["subject_id"].isin(included_subjects)
].copy()

design_df = design_df.set_index("subject_id").loc[included_subjects].reset_index()

design_df["group_code"] = design_df["group"].map(group_mapping)

second_level_design = pd.DataFrame({
    "intercept": np.ones(len(design_df)),
    "group": design_df["group_code"].astype(float)
})

print("Number of difference images:", len(diff_imgs))
print("Design matrix shape:", second_level_design.shape)

plot_design_matrix(second_level_design)
plt.show()


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


# --- Get design matrix for pre-post / group model

# Add group info to included_df
design_df = included_df.merge(
    participants_df[["subject_id", "group"]],
    on="subject_id",
    how="left"
)

# Keep image order exactly as in design_df
derivative_nii = design_df["img"].tolist()

# Code variables
# time: run-01 = pre, run-02 = post
design_df["time"] = design_df["run"].map({
    "run-01": -1,
    "run-02": 1
})

# group: adapt this to your existing group_mapping
# e.g. {"HC": -1, "pat": 1}
design_df["group_code"] = design_df["group"].map(group_mapping)

# interaction
design_df["group_x_time"] = design_df["group_code"] * design_df["time"]

# subject effects: one column per subject
subject_dummies = pd.get_dummies(
    design_df["subject_id"],
    prefix="sub",
    drop_first=True
).astype(float)

mixed_design_matrix = pd.concat(
    [
        pd.DataFrame({
            "intercept": 1,
            "group": design_df["group_code"].astype(float),
            "time": design_df["time"].astype(float),
            "group_x_time": design_df["group_x_time"].astype(float),
        }),
        subject_dummies
    ],
    axis=1
)

print(len(derivative_nii))
print(mixed_design_matrix.shape)

plot_design_matrix(mixed_design_matrix)
plt.show()