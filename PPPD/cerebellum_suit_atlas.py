import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
import nibabel as nib
from nilearn.image import resample_to_img, math_img
from PPPD import _get_output_path, _get_suit_atlas
from scipy import ndimage


# --- Script configuration:
task = "rest"
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "IPLPFL" # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII"
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
pre_post_diff = True
# mask settings
# mask_strategy = "subject_based" # supported strategies: "subject_based", "predefined"
# predefined_mask = "vvn" # supported masks: "dmn", "vvn"
# threshold_mask = 0.8 # only used if mask_strategy == "subject_based"


# --- all directories
# get data dir
data_dir = _get_output_path(part, feature, seed)

# get output path to save results
output_dir = os.path.join(data_dir, "cerebellum_labeling")
os.makedirs(output_dir, exist_ok=True)


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


# --- Load significant cluster mask from previous analysis
if pre_post_diff is True:
    sig_cluster_dir = os.path.join(data_dir, "pre_post_diff", "sig_cluster_masks", "signed")
    try:
        stat_img = nib.load(os.path.join(sig_cluster_dir, f"{feature}_{seed}_{group_comparison}_all_signed_logp_clustermass_fwer05.nii.gz"))
    except FileNotFoundError:
        stat_img = None
        raise FileNotFoundError()
else:
    sig_cluster_dir = os.path.join(data_dir, "sig_cluster_masks")
    try:
        stat_img = nib.load(os.path.join(sig_cluster_dir, f"{feature}_{seed}_{group_comparison}_submask-0.8_logp_clustermass_fwer05.nii.gz"))
    except FileNotFoundError:
        stat_img = None
        raise FileNotFoundError()


# --- Get path to lut file and load atlas image
lut_file, atlas_img = _get_suit_atlas()


# --- Resample stat/cluster map to atlas grid
# stat img space: MNI152NLin2009cAsym (2mm)
# suit atlas space: MNI152NLin6AsymC (1mm)
stat_resampled = resample_to_img(stat_img, atlas_img, interpolation="nearest")
stat_data = stat_resampled.get_fdata()
atlas_data = atlas_img.get_fdata().astype(int)


# --- Threshold significant clusters
neglog_alpha_05 = -np.log10(0.05)
if pre_post_diff:
    # signed map: positive and negative clusters
    sig_mask = np.abs(stat_data) > neglog_alpha_05
else:
    # one-sided positive map
    sig_mask = stat_data > neglog_alpha_05


# --- Label connected clusters
# cluster_data: image where each connected cluster gets a unique integer ID
# n_clusters: total number of significant clusters
cluster_data, n_clusters = ndimage.label(sig_mask)
print(f"Found {n_clusters} significant clusters")


# --- Load atlas lookup table (LUT)
# Maps atlas integer labels to human-readable region names
lut = {}

with open(lut_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        region_id = int(parts[0])
        region_name = " ".join(parts[4:])
        lut[region_id] = region_name

# initialize list to store overlap results
rows = []

# --- Loop over all significant clusters
for cluster_id in range(1, n_clusters + 1):
    # create boolean mask for current cluster
    cluster_mask = cluster_data == cluster_id
    # compute voxel volume (mm³) from image header (here: 1x1x1 mm after resampling to SUIT atlas grid)
    voxel_volume_mm3 = np.prod(stat_resampled.header.get_zooms()[:3])
    # number of voxels belonging to current cluster
    cluster_size_vox = int(np.sum(cluster_mask))
    # convert cluster size into mm³
    cluster_size_mm3 = cluster_size_vox * voxel_volume_mm3

    # determine atlas regions overlapping with current cluster
    atlas_labels, counts = np.unique(atlas_data[cluster_mask], return_counts=True)

    peak_logp = np.max(np.abs(stat_data[cluster_mask]))
    cluster_p = 10 ** (-peak_logp)

    # loop over all overlapping atlas regions
    for label, count in zip(atlas_labels, counts):
        if label == 0:
            continue

        rows.append({
            "Cluster": cluster_id,
            "Atlas label": int(label),
            "Region": lut.get(int(label), f"region {int(label)}"),
            "Size (voxels)": cluster_size_vox,
            "Overlap voxels": int(count),
            "Overlap %": 100 * count / cluster_size_vox,
            "Cluster Stat": np.max(stat_data[cluster_mask]),
            "Cluster p-value": cluster_p,
        })

# get table with overlap information
overlap_df = pd.DataFrame(rows)
overlap_df = overlap_df.sort_values(["Cluster", "Overlap voxels"], ascending=[True, False])

# extract main atlas label per cluster (largest voxel overlap)
main_labels = overlap_df.loc[overlap_df.groupby("Cluster")["Overlap voxels"].idxmax()].reset_index(drop=True)

# save overlap tables
overlap_table_dir = os.path.join(output_dir, "overlap_tables")
os.makedirs(overlap_table_dir, exist_ok=True)
overlap_table_path = os.path.join(overlap_table_dir, f"{file_suffix}_suit_overlap_full.csv")
overlap_df.to_csv(overlap_table_path, index=False)


# --- Plots
# create directory for surface plots
surf_plot_dir = os.path.join(output_dir, "surf_plots")
os.makedirs(surf_plot_dir, exist_ok=True)

# Version 1: Plot binary significant cluster mask
# create binary mask
binary_img = math_img(f"(img > {neglog_alpha_05}).astype(int)", img=stat_resampled)
binary_surf_data = flatmap.vol_to_surf(binary_img, space="MNI")
# re-binarize after surface projection
binary_surf_data = (binary_surf_data > 0).astype(int)

flatmap.plot(
    binary_surf_data,
    overlay_type="label",
    cmap="Wistia_r",
    colorbar=False,
    render="matplotlib"
)
plt.suptitle(f"{base_title}\nsignificant clusters", fontsize=12, y=0.88)
binary_plot_path = os.path.join(surf_plot_dir, f"{file_suffix}_binary_suit_flatmap.png")
plt.savefig(binary_plot_path, dpi=300, bbox_inches="tight")
plt.show()


# Version 2: Plot thresholded -log10(p) statistical map
surf_data = flatmap.vol_to_surf(stat_resampled, space="MNI")
flatmap.plot(
    surf_data,
    overlay_type="func",
    threshold=0.01,
    colorbar=True,
    cmap="inferno" )
plt.suptitle(f"{base_title}\ncluster-mass corrected -log10(p)", fontsize=12, y=0.83)
stat_plot_path = os.path.join(surf_plot_dir, f"{file_suffix}_logp_suit_flatmap.png")
plt.savefig(stat_plot_path, dpi=300, bbox_inches="tight")
plt.show()

