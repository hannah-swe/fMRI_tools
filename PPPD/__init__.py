import os
import yaml
import pandas as pd
import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET
from nilearn.image import load_img
from nilearn.reporting import get_clusters_table
from scipy.spatial.distance import cdist
from nilearn.datasets import fetch_atlas_juelich


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
SUPPORTED_SEEDS = SUPPORTED_SEEDS_1 + SUPPORTED_SEEDS_2
SUPPORTED_MASKS = ["dmn", "vvn"]


# Load config.yml
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(PROJECT_DIR, "config.yml")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

analysis_path = config["analysis_path"]
raw_data_path = config["raw_data_path"]
output_path = config["output_path"]
mask_path = config["mask_path"]
aal_path = config["aal_path"]
suit_path = config["suit_path"]


# Gets the path for analysis folder based on the selected feature
def _get_data_path(feature, seed=None):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")

    if feature == "seed_based":
        if seed is None:
            raise ValueError("For feature = 'seed_based', a seed must be provided")
        if seed in SUPPORTED_SEEDS_1:
            return os.path.join(analysis_path, "both_parts_seed1")
        if seed in SUPPORTED_SEEDS_2:
            return os.path.join(analysis_path, "both_parts_seed2")
        raise ValueError(f"Unsupported seed: {seed}")

    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff")
    if feature == "alff":
        return os.path.join(analysis_path, "both_parts_falff")

    return None


# Gets the part for the derivatives folder based on the selected feature
def _get_derivatives_path(feature, seed=None):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")

    if feature == "seed_based":
        if seed in SUPPORTED_SEEDS_1:
            return os.path.join(analysis_path, "both_parts_seed1", "derivatives", "halfpipe")
        if seed in SUPPORTED_SEEDS_2:
            return os.path.join(analysis_path, "both_parts_seed2", "derivatives", "halfpipe")
        raise ValueError(f"Unsupported seed: {seed}")

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
def _get_output_path(part, feature, seed=None):
    if feature == "seed_based":
        if seed is None:
            raise ValueError("For feature='seed_based', you must provide a seed.")
        if seed in SUPPORTED_SEEDS_1:
            seed_folder = "seed1"
        elif seed in SUPPORTED_SEEDS_2:
            seed_folder = "seed2"
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
def _get_mask_file(predefined_mask):
    if predefined_mask not in SUPPORTED_MASKS:
        raise ValueError(f"Unsupported mask: {predefined_mask}")
    mask_dir = os.path.join(mask_path, f"{predefined_mask}_mask_resampled.nii.gz")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"No resampled mask file found in {mask_dir}")
    mask_file = nib.load(mask_dir)
    print(f"Used mask file: {mask_dir}")

    return mask_file


# Define croup comparison
def _define_group_comparison(group_comparison):
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


#
def _get_selected_subject_list(part, subs, subjects_to_exclude):
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


# Load manually downloaded AAL atlas (NIfTI + XML) from local directory
def _load_local_aal_atlas(aal_dir=aal_path):
    """
    Returns:
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
    region_voxels = np.argwhere(atlas_data == target_value)
    if region_voxels.size == 0:
        return np.nan
    # transform voxel to MNI
    region_coords = np.dot(
        atlas_img.affine,
        np.c_[region_voxels, np.ones(len(region_voxels))].T
    ).T[:, :3]

    peak = np.array([[x, y, z]])
    # minimal distance
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
        # no region; get nearest region
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
def _get_cluster_table_with_aal_labels(stat_img, stat_threshold, cluster_threshold=0, two_sided=False,
                                       min_distance=8.0, aal_dir=aal_path, return_label_maps=False):
    result  = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        min_distance=min_distance,
        return_label_maps=True,
    )
    clusters_table = result[0]
    label_maps = result[1:]

    if clusters_table.empty:
        clusters_table["aal_label"] = pd.Series(dtype="object")
        if return_label_maps:
            return clusters_table, label_maps
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
                row[x_col],
                row[y_col],
                row[z_col],
                atlas_img,
                atlas_data,
                value_to_label,
            )
        ),
        axis=1,
    )

    if return_label_maps:
        return clusters_table, label_maps
    return clusters_table


# Load Juelich brain atlas
def _load_juelich_prob_atlas(atlas_name="prob-2mm"):
    atlas = fetch_atlas_juelich(atlas_name)
    atlas_img = load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()  # 4D: X, Y, Z, region
    labels = list(atlas.labels)

    return atlas_img, atlas_data, labels


# Get coordinates to juelich probability labels
def _coord_to_juelich_prob_labels_sphere(x, y, z, atlas_img, atlas_data, labels, radius_mm=4, top_n=5, min_prob=0.0):
    shape = atlas_data.shape[:3]

    # alle voxel koordinaten
    ijk_grid = np.array(np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij"
    )).reshape(3, -1).T

    # voxel -> mni
    xyz = np.dot(
        atlas_img.affine,
        np.c_[ijk_grid, np.ones(len(ijk_grid))].T
    ).T[:, :3]

    peak = np.array([[x, y, z]])
    dists = cdist(peak, xyz).flatten()
    sphere_mask = dists <= radius_mm
    sphere_voxels = ijk_grid[sphere_mask]

    if len(sphere_voxels) == 0:
        return "no_label", 0.0, ""
    probs_list = []
    for voxel in sphere_voxels:
        probs_list.append(atlas_data[tuple(voxel)])
    probs_array = np.array(probs_list)
    mean_probs = probs_array.mean(axis=0)
    rows = [
        (labels[i], float(p))
        for i, p in enumerate(mean_probs)
        if p > min_prob
    ]
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
    if not rows:
        return "no_label", 0.0, ""
    best_label, best_prob = rows[0]
    all_probs = "; ".join([
        f"{lab}: {p:.2f}"
        for lab, p in rows
    ])

    return best_label, best_prob, all_probs


# Get the significant cluster table with labels of probabilistic juelich brain atlas
def _get_cluster_table_with_juelich_prob_labels(stat_img, stat_threshold, cluster_threshold=0, two_sided=False,
                                                min_distance=8.0, atlas_name="prob-2mm", top_n=5, min_prob=0 ):
    clusters_table = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        min_distance=min_distance,
        return_label_maps=False,
    )

    if clusters_table.empty:
        clusters_table["juelich_label"] = pd.Series(dtype="object")
        clusters_table["juelich_prob"] = pd.Series(dtype="float")
        clusters_table["juelich_top_probs"] = pd.Series(dtype="object")
        return clusters_table
    atlas_img, atlas_data, labels = _load_juelich_prob_atlas(atlas_name)
    coord_cols = None
    for candidate in [("X", "Y", "Z"), ("x", "y", "z")]:
        if all(col in clusters_table.columns for col in candidate):
            coord_cols = candidate
            break
    if coord_cols is None:
        raise ValueError(f"Could not find coordinate columns. Available: {list(clusters_table.columns)}")
    x_col, y_col, z_col = coord_cols
    clusters_table[
        ["juelich_label", "juelich_prob", "juelich_top_probs"]
    ] = clusters_table.apply(
        lambda row: pd.Series(_coord_to_juelich_prob_labels_sphere(
                row[x_col], row[y_col], row[z_col],
                atlas_img, atlas_data, labels,
                top_n=top_n,
                min_prob=min_prob
            )
        ),
        axis=1
    )

    return clusters_table


# Get path to significant post-hoc cluster mask
def _get_posthoc_cluster_mask(feature, group_comparison, pre_post_diff=True, direction=None, part=None, seed=None,):
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
            filename = f"{feature}_{seed}_{group_comparison}_{part_label}_{direction}_cluster_id_map.nii.gz"
        else:
            filename = f"{feature}_{group_comparison}_{part_label}_{direction}_cluster_id_map.nii.gz"
    else:
        if feature == "seed_based":
            filename = f"{feature}_{seed}_{group_comparison}_{part_label}_cluster_id_map.nii.gz"
        else:
            filename = f"{feature}_{group_comparison}_{part_label}_cluster_id_map.nii.gz"

    # build directory
    if pre_post_diff:
        mask_dir = os.path.join(_get_output_path(part, feature, seed), "pre_post_diff", "sig_cluster_masks", direction)
    else:
        mask_dir = os.path.join(_get_output_path(part, feature, seed), "sig_cluster_masks")
    os.makedirs(mask_dir, exist_ok=True)

    return os.path.join(mask_dir, filename)


# Get signed cluster-mass corrected permutation map
def _get_signed_posthoc_map(feature, group_comparison, part=None, seed=None,):
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

    map_dir = os.path.join(_get_output_path(part, feature, seed), "pre_post_diff", "sig_cluster_masks", "signed")
    os.makedirs(map_dir, exist_ok=True)

    return os.path.join(map_dir, filename)


# Get SUIT atlas image and lut image
def _get_suit_atlas():
    lut_file = os.path.join(suit_path, "atl-Anatom.lut")
    atlas_img = nib.load(os.path.join(suit_path, "atl-Anatom_space-MNI_dseg.nii"))

    return lut_file, atlas_img


def get_main_values_tables_path():
    main_values_tables_path = config["main_values_tables_path"]
    return main_values_tables_path


def get_posturography_path():
    posturography_path = config["posturography_path"]
    return posturography_path


def get_connectivity_path():
    connectivity_path = config["connectivity_path"]
    return connectivity_path