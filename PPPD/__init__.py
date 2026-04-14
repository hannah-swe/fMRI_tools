import os
import pandas as pd
import nibabel as nib


SUPPORTED_TASKS = ["rest"]
SUPPORTED_FEATURES = ["seed_based", "falff"]
SUPPORTED_RUNS = ["run-01", "run-02"]
SUPPORTED_SEEDS = ["InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                   "InsulaOP3RAnat", "InsulaOP3Sphere",
                   "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                   "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                   "Precuneus"]
SUPPORTED_MASKS = ["dmn"]


# Path to folder with all HALFpipe working directories
analysis_path = '/data_wgs04/ag-sensomotorik/PPPD/analysis/'
raw_data_path = '/data_wgs04/ag-sensomotorik/PPPD/data/all_subjects1/'
output_path = '/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/'
mask_path = '/data_wgs04/ag-sensomotorik/PPPD/masks/'


# Gets the path for analysis folder based on the selected feature
def _get_data_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if feature == "seed_based":
        return os.path.join(analysis_path, "both_parts_seed1")
    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff")
    return None


# Gets the part for the derivatives folder based on the selected feature
def _get_derivatives_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if feature == "seed_based":
        return os.path.join(analysis_path, "both_parts_seed1", "derivatives", "halfpipe")
    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff", "derivatives", "halfpipe")
    return None


# Gets participants.tsv from data folder
def _get_participants_tsv():
    tsv_path = os.path.join(raw_data_path, "participants.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"No participants.tsv found in {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


# Gets full filename for statistical maps via run, feature and seed
def _get_full_filename(subject_id, task, run, feature, seed=None):
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
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    return filename


# Gets the filename of subject-wise brain masks to create a group wide brain mask
def _get_mask_filename(subject_id, task, run, feature, seed=None):
    if feature == 'seed_based':
        if seed not in SUPPORTED_SEEDS:
            raise ValueError(f"Unsupported seed: {seed}")
        mask_filename = f"{subject_id}_task-{task}_{run}_feature-seedbased_seed-{seed}_mask.nii.gz"
    elif feature == 'falff':
        mask_filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_mask.nii.gz"
    else:
        raise ValueError(f"Unsupported feature: {feature}")
    return mask_filename


# Gets output path
def _get_output_path(feature):
    if feature == "seed_based":
        return os.path.join(output_path, "both_parts_seed1")
    elif feature == "falff":
        return os.path.join(output_path, "both_parts_falff")
    else:
        raise ValueError(f"Unsupported feature: {feature}")


def _get_mask_file(predefined_mask):
    if predefined_mask not in SUPPORTED_MASKS:
        raise ValueError(f"Unsupported mask: {predefined_mask}")
    mask_dir = os.path.join(mask_path, f"{predefined_mask}_mask_resampled.nii.gz")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"No resampled mask file found in {mask_dir}")
    mask_file = nib.load(mask_dir)
    print(f"Used mask file: {mask_dir}")
    return mask_file