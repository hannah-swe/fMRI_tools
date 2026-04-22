import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import load_img
from nilearn.reporting import get_clusters_table


SUPPORTED_TASKS = ["rest"]
SUPPORTED_FEATURES = ["seed_based", "falff", "alff"]
SUPPORTED_RUNS = ["run-01", "run-02"]
SUPPORTED_SEEDS = ["InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                   "InsulaOP3RAnat", "InsulaOP3Sphere",
                   "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                   "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                   "Precuneus"]
SUPPORTED_MASKS = ["dmn", "vvn"]


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
    if feature == "alff":
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
    if feature == "alff":
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
    elif feature == 'alff':
        filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_alff.nii.gz"
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
    elif feature == 'alff':
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
    elif feature == "alff":
        return os.path.join(output_path, "both_parts_alff")
    else:
        raise ValueError(f"Unsupported feature: {feature}")


# Gets mask path and load predefined mask file
def _get_mask_file(predefined_mask):
    if predefined_mask not in SUPPORTED_MASKS:
        raise ValueError(f"Unsupported mask: {predefined_mask}")
    mask_dir = os.path.join(mask_path, f"{predefined_mask}_mask_resampled.nii.gz")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"No resampled mask file found in {mask_dir}")
    mask_file = nib.load(mask_dir)
    print(f"Used mask file: {mask_dir}")
    return mask_file


# Extract cluster table from a stat image and annotate peak coordinates with AAL atlas labels
def _get_cluster_table_with_aal_labels(stat_img, stat_threshold, cluster_threshold=0, two_sided=False, min_distance=8.0,
                                      aal_version="3v2", atlas_img=None, atlas_labels=None, atlas_indices=None):
    """
    Parameters
    ----------
    stat_img : Niimg-like object
        Thresholded or unthresholded statistical image.
    stat_threshold : float
        Cluster-forming threshold passed to nilearn.reporting.get_clusters_table.
    cluster_threshold : int, default=0
        Minimum cluster size in voxels.
    two_sided : bool, default=False
        Whether to extract positive and negative clusters.
    min_distance : float, default=8.0
        Minimum distance between subpeaks in mm.
    aal_version : str, default="3v2"
        AAL atlas version for nilearn.datasets.fetch_atlas_aal.
    atlas_img : Niimg-like or None
        Optional preloaded atlas image. If None, AAL is fetched automatically.
    atlas_labels : list[str] or None
        Optional atlas labels.
    atlas_indices : list[str] or None
        Optional atlas indices as provided by fetch_atlas_aal().

    Returns
    -------
    pandas.DataFrame
        Cluster table with an added 'aal_label' column.
    """
    # Cluster table from Nilearn
    clusters_table = get_clusters_table(stat_img, stat_threshold=stat_threshold, cluster_threshold=cluster_threshold,
                                        two_sided=two_sided, min_distance=min_distance, return_label_maps=False)
    if clusters_table.empty:
        clusters_table["aal_label"] = pd.Series(dtype="object")
        return clusters_table

    # Load AAL atlas if not provided
    if atlas_img is None or atlas_labels is None or atlas_indices is None:
        aal = datasets.fetch_atlas_aal(version=aal_version)
        atlas_img = load_img(aal.maps)
        atlas_labels = list(aal.labels)
        atlas_indices = list(aal.indices)

    # Build robust AAL value -> label mapping
    # Important because AAL map values are not guaranteed to match label list indices
    atlas_data = atlas_img.get_fdata()
    value_to_label = {int(idx): label for idx, label in zip(atlas_indices, atlas_labels)}

    # Convert MNI coordinate to atlas label using nearest voxel lookup
    def coord_to_aal_label(x, y, z):
        xyz_h = np.array([x, y, z, 1.0])
        ijk = np.linalg.inv(atlas_img.affine).dot(xyz_h)[:3]
        ijk = np.round(ijk).astype(int)
        # Bounds check
        if np.any(ijk < 0) or np.any(ijk >= atlas_data.shape):
            return "out_of_bounds"
        atlas_value = atlas_data[tuple(ijk)]
        # Background
        if atlas_value == 0:
            return "no_label"
        return value_to_label.get(int(atlas_value), f"unknown_label_{int(atlas_value)}")

    # Add anatomical label for each peak row; get_clusters_table usually provides X, Y, Z columns
    coord_cols = None
    for candidate in [("X", "Y", "Z"), ("x", "y", "z")]:
        if all(col in clusters_table.columns for col in candidate):
            coord_cols = candidate
            break
    if coord_cols is None:
        raise ValueError(
            f"Could not find coordinate columns in clusters table. Available columns: {list(clusters_table.columns)}"
        )
    x_col, y_col, z_col = coord_cols
    clusters_table["aal_label"] = clusters_table.apply(lambda row: coord_to_aal_label(row[x_col], row[y_col], row[z_col]),
                                                       axis=1)
    return clusters_table