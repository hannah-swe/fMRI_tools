import os
import pandas as pd
import nibabel as nib
from .config import load_config


SUPPORTED_TASKS = ["rest"]
SUPPORTED_FEATURES = ["seed_based", "falff", "alff"]
SUPPORTED_RUNS = ["run-01", "run-02"]
SUPPORTED_SEEDS_1 = ["InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                   "InsulaOP3RAnat", "InsulaOP3Sphere",
                   "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                   "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                   "Precuneus"]
SUPPORTED_SEEDS_2 = ["CSv", "CSvR",
                     "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                     "VermisUvulaL", "VermisVII"]
SUPPORTED_SEEDS_3 = ["HippocampusL", "HippocampusR"]
SUPPORTED_SEEDS_4 = ["PrecuneusL", "PrecuneusR"]
SUPPORTED_SEEDS = SUPPORTED_SEEDS_1 + SUPPORTED_SEEDS_2 + SUPPORTED_SEEDS_3 + SUPPORTED_SEEDS_4
SUPPORTED_MASKS = ["dmn", "vvn"]


# Load config.yml
config = load_config()
analysis_path = config["paths"]["analysis_path"]
output_path = config["paths"]["output_path"]
mask_path = config["paths"]["mask_path"]
aal_path = config["paths"]["aal_path"]
suit_path = config["paths"]["suit_path"]


# Gets the path for analysis folder based on the selected feature
def get_derivatives_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    return os.path.join(analysis_path, "HALFpipe_output")


# Gets participants.tsv from data folder
def get_participants_tsv():
    tsv_path = os.path.join(analysis_path, "HALFpipe_output", "participants.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"No participants.tsv found in {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    return df


# Gets full filename for statistical maps via run, feature and seed
def get_full_filename(subject_id, task, run, feature, seed=None):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if run not in SUPPORTED_RUNS:
        raise ValueError(f"Unsupported run: {run}")
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")

    if feature == 'seed_based':
        if seed not in SUPPORTED_SEEDS:
            raise ValueError(f"Unsupported seed: {seed}")
        filename = f"{subject_id}_task-{task}_{run}_feature-seedbased_seed-{seed}_stat-effect_statmap.nii.gz"
    elif feature == 'falff':
        filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_falff.nii.gz"
    elif feature == 'alff':
        filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_alff.nii.gz"
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    return filename


# Gets the filename of subject-wise brain masks to create a group wide brain mask
def get_mask_filename(subject_id, task, run, feature, seed=None):
    if feature == 'seed_based':
        if seed not in SUPPORTED_SEEDS:
            raise ValueError(f"Unsupported seed: {seed}")
        mask_filename = f"{subject_id}_task-{task}_{run}_feature-seedbased_seed-{seed}_mask.nii.gz"
    elif feature == 'falff':
        mask_filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_mask.nii.gz"
    elif feature == 'alff':
        mask_filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_mask.nii.gz"
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    return mask_filename


# Gets output path
def get_output_path(part, feature, seed=None):
    if feature == "seed_based":
        if seed is None:
            raise ValueError("For feature='seed_based', you must provide a seed.")
        if seed in SUPPORTED_SEEDS_1:
            seed_folder = "seed1"
        elif seed in SUPPORTED_SEEDS_2:
            seed_folder = "seed2"
        elif seed in SUPPORTED_SEEDS_3:
            seed_folder = "seed3"
        elif seed in SUPPORTED_SEEDS_4:
            seed_folder = "seed4"
        else:
            raise ValueError(f"Unsupported seed: {seed}")

    if part is None:
        if feature == "seed_based":
            folder = f"both_parts_{seed_folder}"
        elif feature == "falff":
            folder = "both_parts_falff"
        elif feature == "alff":
            folder = "both_parts_alff"
        else:
            raise ValueError(f"Unsupported feature: {feature}")

    elif part == 1:
        if feature == "seed_based":
            folder = f"part1_{seed_folder}"
        elif feature == "falff":
            folder = "part1_falff"
        elif feature == "alff":
            folder = "part1_alff"
        else:
            raise ValueError(f"Unsupported feature: {feature}")

    elif part == 2:
        if feature == "seed_based":
            folder = f"part2_{seed_folder}"
        elif feature == "falff":
            folder = "part2_falff"
        elif feature == "alff":
            folder = "part2_alff"
        else:
            raise ValueError(f"Unsupported feature: {feature}")

    else:
        raise ValueError(f"Unsupported part: {part}")
    full_output_path = os.path.join(output_path, folder)
    os.makedirs(full_output_path, exist_ok=True)
    return full_output_path


# Gets mask path and load predefined mask file
def get_mask_file(predefined_mask):
    if predefined_mask not in SUPPORTED_MASKS:
        raise ValueError(f"Unsupported mask: {predefined_mask}")
    mask_dir = os.path.join(mask_path, f"{predefined_mask}_mask_resampled.nii.gz")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"No resampled mask file found in {mask_dir}")
    mask_file = nib.load(mask_dir)
    print(f"Used mask file: {mask_dir}")

    return mask_file


# Define croup comparison
def define_group_comparison(group_comparison):
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

    return group_mapping


# List of selected subjects based on subjects.py
def get_selected_subject_list(part, subs, subjects_to_exclude):
    if part is None:
        selected_subs = [s for s in subs if s not in subjects_to_exclude]
    elif part == 1:
        selected_subs = [s for s in subs if s < 100]
    elif part == 2:
        selected_subs = [s for s in subs if s >= 100]
    else:
        raise ValueError("part must be None, 1, or 2")
    print(f"Selected part: {part if part is not None else 'all'}")
    print(f"Selected subjects before loading: {len(selected_subs)}")

    return selected_subs


# Get path to significant post-hoc cluster mask
def get_posthoc_cluster_mask(feature, group_comparison, pre_post_diff=True, direction=None, part=None, seed=None,):
    if feature == "seed_based" and seed is None:
        raise ValueError("seed must be provided for seed_based feature.")
    if pre_post_diff and direction not in ["positive", "negative"]:
        raise ValueError("direction must be 'positive' or 'negative'")
    if part is None:
        part_label = "all"
    else:
        part_label = str(part)

    # build filename and directory
    if pre_post_diff:
        if feature == "seed_based":
            filename = f"{feature}_{seed}_{group_comparison}_{part_label}_{direction}_twosided_cluster_id_map.nii.gz"
        else:
            filename = f"{feature}_{group_comparison}_{part_label}_{direction}_twosided_cluster_id_map.nii.gz"
    else:
        if feature == "seed_based":
            filename = f"{feature}_{seed}_{group_comparison}_{part_label}_twosided_cluster_id_map.nii.gz"
        else:
            filename = f"{feature}_{group_comparison}_{part_label}_twosided_cluster_id_map.nii.gz"

    # build directory
    if pre_post_diff:
        mask_dir = os.path.join(get_output_path(part, feature, seed), "pre_post_diff", "sig_cluster_masks", direction)
    else:
        mask_dir = os.path.join(get_output_path(part, feature, seed), "sig_cluster_masks")
    os.makedirs(mask_dir, exist_ok=True)

    return os.path.join(mask_dir, filename)


# Get signed cluster-mass corrected permutation map
def get_signed_posthoc_map(feature, group_comparison, part=None, seed=None,):
    if feature == "seed_based" and seed is None:
        raise ValueError("seed must be provided for seed_based feature.")
    if part is None:
        part_label = "all"
    else:
        part_label = str(part)
    if feature == "seed_based":
        filename = f"{feature}_{seed}_{group_comparison}_{part_label}_signed_logp_clustermass_fwer05.nii.gz"
    else:
        filename = f"{feature}_{group_comparison}_{part_label}_signed_logp_clustermass_fwer05.nii.gz"

    map_dir = os.path.join(get_output_path(part, feature, seed), "pre_post_diff", "sig_cluster_masks", "signed")
    os.makedirs(map_dir, exist_ok=True)

    return os.path.join(map_dir, filename)


# Get SUIT atlas image and lut image
def get_suit_atlas():
    lut_file = os.path.join(suit_path, "atl-Anatom.lut")
    atlas_img = nib.load(os.path.join(suit_path, "atl-Anatom_space-MNI_dseg.nii"))

    return lut_file, atlas_img


def get_main_values_tables_path():
    main_values_tables_path = config["paths"]["main_values_tables_path"]
    return main_values_tables_path


def get_posturography_path():
    posturography_path = config["paths"]["posturography_path"]
    return posturography_path


def get_connectivity_path():
    connectivity_path = config["paths"]["connectivity_path"]
    return connectivity_path