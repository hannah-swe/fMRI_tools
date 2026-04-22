import os
import pandas as pd
import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET
from nilearn.image import load_img
from nilearn.reporting import get_clusters_table
from scipy.spatial.distance import cdist


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
aal_path='/home/hannahschewe/nilearn_data/aal_3v2/'


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


# Load manually downloaded AAL atlas (NIfTI + XML) from local directory
def _load_local_aal_atlas(aal_dir=aal_path):
    """
    Returns
    -------
    atlas_img : nibabel image
    atlas_data : np.ndarray
    value_to_label : dict[int, str]
    """
    nii_path = os.path.join(aal_dir, "AAL3v1.nii")
    xml_path = os.path.join(aal_dir, "AAL3v1.xml")

    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"AAL NIfTI not found: {nii_path}")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"AAL XML not found: {xml_path}")

    atlas_img = load_img(nii_path)
    atlas_data = atlas_img.get_fdata()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    value_to_label = {}
    for label in root.findall(".//label"):
        index_elem = label.find("index")
        name_elem = label.find("name")

        if index_elem is None or name_elem is None:
            continue

        value_to_label[int(index_elem.text)] = name_elem.text.strip()

    if not value_to_label:
        raise ValueError(f"No labels could be parsed from XML: {xml_path}")

    return atlas_img, atlas_data, value_to_label


# Compute minimal Euclidean distance (mm) from a coordinate to all voxels belonging to a given atlas region
def _distance_to_region(x, y, z, atlas_img, atlas_data, target_value):
    # alle Voxels der Region finden
    region_voxels = np.argwhere(atlas_data == target_value)

    if region_voxels.size == 0:
        return np.nan

    # voxel → MNI transformieren
    region_coords = np.dot(
        atlas_img.affine,
        np.c_[region_voxels, np.ones(len(region_voxels))].T
    ).T[:, :3]

    peak = np.array([[x, y, z]])

    # minimale Distanz
    distances = cdist(peak, region_coords)
    return float(distances.min())


# Convert MNI coordinates to nearest AAL atlas label.
def _coord_to_label_and_distance(x, y, z, atlas_img, atlas_data, value_to_label):
    xyz_h = np.array([x, y, z, 1.0])
    ijk = np.linalg.inv(atlas_img.affine).dot(xyz_h)[:3]
    ijk = np.round(ijk).astype(int)

    if np.any(ijk < 0) or np.any(ijk >= atlas_data.shape):
        return "out_of_bounds"

    atlas_value = atlas_data[tuple(ijk)]

    if atlas_value == 0:
        # keine Region → nächstgelegene suchen
        possible_values = list(value_to_label.keys())
        min_dist = np.inf
        best_label = "no_label"

        for val in possible_values:
            d = _distance_to_region(x, y, z, atlas_img, atlas_data, val)
            if d < min_dist:
                min_dist = d
                best_label = value_to_label[val]

        return best_label, min_dist

    else:
        label = value_to_label.get(int(atlas_value), "unknown")
        return label, 0.0


# Extract cluster table from a stat image and annotate peak coordinates with AAL atlas labels
def _get_cluster_table_with_aal_labels(
    stat_img,
    stat_threshold,
    cluster_threshold=0,
    two_sided=False,
    min_distance=8.0,
    aal_dir=aal_path,
):
    clusters_table = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        min_distance=min_distance,
        return_label_maps=False,
    )

    if clusters_table.empty:
        clusters_table["aal_label"] = pd.Series(dtype="object")
        return clusters_table

    atlas_img, atlas_data, value_to_label = _load_local_aal_atlas(aal_dir=aal_dir)

    coord_cols = None
    for candidate in [("X", "Y", "Z"), ("x", "y", "z")]:
        if all(col in clusters_table.columns for col in candidate):
            coord_cols = candidate
            break

    if coord_cols is None:
        raise ValueError(
            f"Could not find coordinate columns in cluster table. Available columns: {list(clusters_table.columns)}"
        )

    x_col, y_col, z_col = coord_cols

    clusters_table[["aal_label", "distance_mm"]] = clusters_table.apply(
        lambda row: pd.Series(
            _coord_to_label_and_distance(
                row[x_col], row[y_col], row[z_col],
                atlas_img, atlas_data, value_to_label
            )
        ),
        axis=1
    )

    return clusters_table