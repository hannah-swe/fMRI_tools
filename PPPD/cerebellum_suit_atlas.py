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
direction = "negative" # possible directions for pre-post differences:
                        # "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)


# --- all directories
# get data dir
data_dir = _get_output_path(part, feature, seed)

# get output path to save results
if pre_post_diff is True:
    output_dir = os.path.join(data_dir, "pre_post_diff", "cerebellum_labeling")
else:
    output_dir = os.path.join(data_dir, "cerebellum_labeling")
os.makedirs(output_dir, exist_ok=True)


# --- Get path to lut file and load atlas image
lut_file, atlas_img = _get_suit_atlas()


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


# --- Load significant cluster map and matching permutation table
part_label = "all" if part is None else f"{part}"
if feature == "seed_based":
    table_file_suffix = f"{feature}_{seed}_{group_comparison}"
else:
    table_file_suffix = f"{feature}_{group_comparison}"

table_file_suffix = f"{table_file_suffix}"

if pre_post_diff is True:
    if direction not in ["positive", "negative"]:
        raise ValueError("For pre_post_diff=True, direction must be 'positive' or 'negative'.")
    sig_cluster_dir = os.path.join(data_dir, "pre_post_diff", "sig_cluster_masks", f"{direction}")
    cluster_map_path = os.path.join(sig_cluster_dir, f"{table_file_suffix}_all_{direction}_cluster_id_map.nii.gz")
    cluster_table_path = os.path.join(data_dir, "pre_post_diff", "cluster_tables",
                                      f"{table_file_suffix}_all_cluster_table_perm_mass.csv")

else:
    sig_cluster_dir = os.path.join(data_dir, "sig_cluster_masks")
    cluster_map_path = os.path.join(sig_cluster_dir, f"{table_file_suffix}_submask-0.8_logp_clustermass_fwer05.nii.gz")
    cluster_table_path = os.path.join(data_dir, "cluster_tables", f"{table_file_suffix}_submask-0.8_cluster_table_perm_mass.csv")

# load cluster map
if not os.path.exists(cluster_map_path):
    raise FileNotFoundError(f"Cluster map not found: {cluster_map_path}")
stat_img = nib.load(cluster_map_path)
print("Loaded cluster map:")
print(cluster_map_path)

# load permutation table
if not os.path.exists(cluster_table_path):
    raise FileNotFoundError(f"Cluster table not found: {cluster_table_path}")
cluster_table_perm_mass = pd.read_csv(cluster_table_path)
print("Loaded permutation cluster table:")
print(cluster_table_path)


# --- Select relevant clusters from table
if pre_post_diff is True:
    if direction == "negative":
        relevant_clusters = cluster_table_perm_mass[
            cluster_table_perm_mass["Stat"] < 0
        ].copy()
    elif direction == "positive":
        relevant_clusters = cluster_table_perm_mass[
            cluster_table_perm_mass["Stat"] > 0
        ].copy()
else:
    relevant_clusters = cluster_table_perm_mass.copy()

print("Relevant clusters from table:", len(relevant_clusters))
print("Cluster IDs from table:", sorted(relevant_clusters["Cluster"].unique()))


# --- Resample stat/cluster map to atlas grid
# stat img space: MNI152NLin2009cAsym (2mm)
# suit atlas space: MNI152NLin6AsymC (1mm)
stat_resampled = resample_to_img(stat_img, atlas_img, interpolation="nearest")
stat_data = stat_resampled.get_fdata()
atlas_data = atlas_img.get_fdata().astype(int)


# --- Get cluster labels
neglog_alpha_05 = -np.log10(0.05)

if pre_post_diff:
    # stat_img is already a cluster-ID map
    cluster_data = np.rint(stat_data).astype(int)

    cluster_ids = sorted(np.unique(cluster_data))
    cluster_ids = [c for c in cluster_ids if c != 0]
else:
    # stat_img is still a continuous logp map -> temporary old behavior
    sig_mask = stat_data > neglog_alpha_05
    cluster_data, n_clusters = ndimage.label(sig_mask)

    cluster_ids = list(range(1, n_clusters + 1))

print(f"Found {len(cluster_ids)} significant clusters")
print("Cluster IDs from map:", cluster_ids)


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
for cluster_id in cluster_ids:
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

    # get stat + p-value from original permutation table
    if pre_post_diff:
        table_row = relevant_clusters.loc[
            relevant_clusters["Cluster"] == cluster_id
        ]
    else:
        # temporary fallback because cluster map is still relabeled via ndimage
        table_row = relevant_clusters.iloc[[cluster_id - 1]]
    if len(table_row) != 1:
        raise ValueError(
            f"Expected exactly one table row for Cluster={cluster_id}, "
            f"found {len(table_row)}."
        )
    table_row = table_row.iloc[0]

    cluster_stat = table_row["Stat"]
    cluster_p = table_row["p-value"]

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
            "Cluster Stat": cluster_stat,
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

# Version 1: Plot labeled significant clusters

# use tab20 color IDs starting at 2:
# first cerebellar cluster -> dark orange
# second cerebellar cluster -> light orange
# etc.
plot_id_map = {
    cluster_id: plot_id
    for plot_id, cluster_id in enumerate(cluster_ids, start=2)
}

cluster_surf_data = np.zeros_like(
    flatmap.vol_to_surf(
        nib.Nifti1Image(
            (cluster_data != 0).astype(np.uint8),
            affine=stat_resampled.affine,
            header=stat_resampled.header
        ),
        space="MNI"
    ),
    dtype=int
)

for cluster_id in cluster_ids:
    single_cluster_vol = (cluster_data == cluster_id).astype(np.uint8)

    single_cluster_img = nib.Nifti1Image(
        single_cluster_vol,
        affine=stat_resampled.affine,
        header=stat_resampled.header
    )
    single_cluster_img.set_data_dtype(np.uint8)

    single_cluster_surf = flatmap.vol_to_surf(single_cluster_img, space="MNI")
    single_cluster_surf_mask = single_cluster_surf > 0

    # use plot ID, not original cluster ID
    cluster_surf_data[single_cluster_surf_mask] = plot_id_map[cluster_id]

flatmap.plot(
    cluster_surf_data,
    overlay_type="label",
    cmap="tab20",
    colorbar=False,
    render="matplotlib"
)

plt.suptitle(f"{base_title}\nsignificant clusters", fontsize=12, y=0.88)

binary_plot_path = os.path.join(
    surf_plot_dir,
    f"{file_suffix}_binary_suit_flatmap.png"
)

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

