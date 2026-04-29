import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_mask_filename,
                  _get_output_path, _get_mask_file, _get_cluster_table_with_aal_labels)
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
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "InsulaId1L" # List of supported seeds:
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
n_perm = 5000


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
output_dir = os.path.join(output_dir, "pre_post_diff")


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


# -- Choose subjects depending on experimental part:
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
diff_imgs = []
included_rows = []
sub_mask_imgs = []

for s in selected_subs:
    # get full subject id
    subject_id = f"sub-{s:03d}"

    # load statistical nifti images of both runs
    run_imgs = {}
    run_masks = {}
    for run in runs:
        filename = _get_full_filename(subject_id, task, run, feature, seed)
        img_path = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", filename)
        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue
        try:
            nib.load(img_path)
            run_imgs[run] = img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # load subject masks only if subject-based mask strategy is used
        if mask_strategy == "subject_based":
            mask_filename = _get_mask_filename(subject_id, task, run, feature, seed)
            mask_path = os.path.join(deriv_dir, subject_id, "func", f"task-{task}", mask_filename)
            if not os.path.exists(mask_path):
                print(f"Missing mask: {mask_path}")
                continue
            try:
                nib.load(mask_path)
                run_masks[run] = mask_path
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                continue

    # make sure all subjects do have images for both runs
    if "run-01" not in run_imgs or "run-02" not in run_imgs:
        print(f"Skipping {subject_id}: incomplete pre/post images")
        continue

    # require both masks if subject-based mask is used
    if mask_strategy == "subject_based":
        if "run-01" not in run_masks or "run-02" not in run_masks:
            print(f"Skipping {subject_id}: incomplete pre/post masks")
            continue

    # compute difference image between post/run-02 and pre/run-01
    diff_img = math_img("post - pre", post=run_imgs["run-02"], pre=run_imgs["run-01"])
    diff_filename = f"{subject_id}_task-{file_suffix}_post-minus-pre.nii.gz"
    diff_path = os.path.join(output_dir, "diff_images", diff_filename)
    diff_img.to_filename(diff_path)

    diff_imgs.append(diff_path)

    included_rows.append({
        "subject_id": subject_id,
        "subject_num": s,
        "diff_img": diff_path,
        "pre_img": run_imgs["run-01"],
        "post_img": run_imgs["run-02"],
    })

    if mask_strategy == "subject_based":
        sub_mask_imgs.extend([
            run_masks["run-01"],
            run_masks["run-02"]
        ])

included_df = pd.DataFrame(included_rows)
print("Loaded difference images:", len(diff_imgs))
print("Loaded subjects:", included_df["subject_id"].nunique())


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


# --- Get design matrix for pre-post model and plot it
design_df = included_df.merge(participants_df[["subject_id", "group"]], on="subject_id", how="left")
design_df["group_code"] = design_df["group"].map(group_mapping)
if design_df["group_code"].isna().any():
    print(design_df[design_df["group_code"].isna()][["subject_id", "group"]])
    raise ValueError("Some subjects have missing or unmapped group labels.")
# keep exact image order
diff_imgs = design_df["diff_img"].tolist()
second_level_design = pd.DataFrame({
    "intercept": np.ones(len(design_df)),
    "group": design_df["group_code"].astype(float)
})
print("Number of difference images:", len(diff_imgs))
print("Design matrix shape:", second_level_design.shape)
# plot_design_matrix(second_level_design)
# plt.show()


# --- PARAMETRIC TEST (voxel-wise two sample t-test unpaired)
# fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
second_level_model = SecondLevelModel(mask_img=analysis_mask)
second_level_model = second_level_model.fit(diff_imgs, design_matrix=second_level_design)
z_map = second_level_model.compute_contrast("group", output_type='z_score')

# Significance test:
# Version 1: abs(z) > 3.09 (equivalent to p < 0.001 one-sided test), cluster size > 10 voxels
# z(threshold)=3.09 for p=0.001 when testing one-sided; 3.29 for two-sided
thresholded_map1 = threshold_img(z_map, cluster_threshold=10, threshold=3.09, two_sided=True)
thr1_data = thresholded_map1.get_fdata()
# plot thresholded maps if there are any voxels/clusters left
if np.any(thr1_data != 0):
    plot_stat_map(thresholded_map1, display_mode='mosaic', cmap="RdBu_r",
                  title=f"difference z map (post - pre) \n {base_title}; z > 3.09; clusters > 10 voxels")
    plt.show()
    fig = plt.figure(figsize=(9,5))
    display = plot_glass_brain(thresholded_map1, cmap="RdBu_r",
                               figure=fig, title=None, plot_abs=False, symmetric_cbar=True)
    display.frame_axes.figure.suptitle(f"difference z map (post - pre) \n {base_title}; z > 3.09; clusters > 10 voxels")
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
        two_sided=True
    )


# --- NON-PARAMETRIC TESTS: permutation inference with cluster-level correction:
# threshold is in p-scale, not z-scale; threshold=0.001 corresponds to a cluster-forming threshold of p < .001
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    perm_out = non_parametric_inference(
        second_level_input=diff_imgs,
        design_matrix=second_level_design,
        second_level_contrast="group",
        mask=analysis_mask,
        model_intercept=False,   # intercept is already in the design matrix
        n_perm=n_perm,
        two_sided_test=True,
        threshold=0.001,         # cluster-forming threshold in p-scale
        n_jobs=40,                # adapt to your system
        verbose=1,
    )

# convert corrected -log10(p) maps into thresholded views (corrected p < .05  <=>  -log10(p) > 1.30103)
neglog_alpha_05 = -np.log10(0.05)
# use parametric z-map only for the sign/direction
sign_z_map = second_level_model.compute_contrast("group", output_type="z_score")
# signed cluster-mass corrected -log10(p) map
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    signed_logp_mass_thr = math_img(
        f"np.where(logp > {neglog_alpha_05}, np.sign(z) * logp, 0)",
        logp=perm_out["logp_max_mass"],
        z=sign_z_map
    )

# plot cluster-mass corrected map
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = signed_logp_mass_thr.get_fdata()
    visible = np.abs(data[data != 0])
    if len(visible) > 0:
        vmax = np.ceil(np.max(visible) * 10) / 10
        fig = plt.figure(figsize=(9, 5))
        display = plot_glass_brain(signed_logp_mass_thr, cmap="RdBu_r", vmax=vmax, threshold=neglog_alpha_05,
                                   plot_abs=False, symmetric_cbar=True, figure=fig, title=None, colorbar=True) #
        display.frame_axes.figure.suptitle(f"difference map permutation test cluster-mass FWER\n {base_title} | corrected p < .05")
        display.savefig(os.path.join(output_dir, "04_nonparametric", f"{file_suffix}_perm_clustermass_fwer05.png"))
    else:
        print("No clusters survive permutation cluster-mass FWER correction.")

# Get cluster table with aal brain atlas
logp_mass_thr_float = math_img("img.astype(float)", img=signed_logp_mass_thr)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cluster_table_perm_mass = _get_cluster_table_with_aal_labels(
        stat_img=logp_mass_thr_float,
        stat_threshold=neglog_alpha_05,
        cluster_threshold=0,
        two_sided=True,
    )