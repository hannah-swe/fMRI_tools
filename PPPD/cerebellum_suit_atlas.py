from SUITPy import atlas
import SUITPy.flatmap as flatmap
import nibabel as nb
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
import nibabel as nib
from PPPD import _get_output_path
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from nilearn.image import math_img


# --- Script configuration:
task = "rest"
runs = ["run-01", "run-02"] # pre, post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seed = "V5L" # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII"
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
# mask settings
mask_strategy = "subject_based" # supported strategies: "subject_based", "predefined"
predefined_mask = "vvn" # supported masks: "dmn", "vvn"
threshold_mask = 0.8 # only used if mask_strategy == "subject_based"
pre_post_diff = False


sig_cluster_dir = _get_output_path(part, feature, seed)
if pre_post_diff is True:
    sig_cluster_dir = os.path.join(sig_cluster_dir, "pre_post_diff", "sig_cluster_masks")
else:
    sig_cluster_dir = os.path.join(sig_cluster_dir, "sig_cluster_masks")

suit_path = "/home/hannahschewe/Downloads/cerebellar_atlases-master/Diedrichsen_2009/"
lut_file = os.path.join(suit_path, "atl-Anatom.lut")

atlas_img = nib.load(os.path.join(suit_path, "atl-Anatom_space-MNI_dseg.nii"))
stat_img = nib.load(os.path.join(sig_cluster_dir, f"{feature}_{seed}_{group_comparison}_submask-0.8_logp_clustermass_fwer05.nii.gz"))


# --- Resample stat/cluster map to atlas grid
stat_resampled = resample_to_img(stat_img, atlas_img, interpolation="nearest")

stat_data = stat_resampled.get_fdata()
atlas_data = atlas_img.get_fdata().astype(int)

# --- threshold significant clusters
neglog_alpha_05 = -np.log10(0.05)

sig_mask = stat_data > neglog_alpha_05

# --- label connected clusters
cluster_data, n_clusters = ndimage.label(sig_mask)

print(f"Found {n_clusters} significant clusters")

lut = {}

with open(lut_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        region_id = int(parts[0])
        region_name = " ".join(parts[4:])
        lut[region_id] = region_name

rows = []

for cluster_id in range(1, n_clusters + 1):
    cluster_mask = cluster_data == cluster_id
    voxel_volume_mm3 = np.prod(stat_resampled.header.get_zooms()[:3])
    cluster_size_vox = int(np.sum(cluster_mask))
    cluster_size_mm3 = cluster_size_vox * voxel_volume_mm3

    atlas_labels, counts = np.unique(
        atlas_data[cluster_mask],
        return_counts=True
    )

    for label, count in zip(atlas_labels, counts):
        if label == 0:
            continue

        rows.append({
            "Cluster": cluster_id,
            "Atlas label": int(label),
            "Region": lut.get(int(label), f"region {int(label)}"),
            "Overlap voxels": int(count),
            "Cluster size voxels": cluster_size_vox,
            "Cluster size mm3": cluster_size_mm3,
            "Overlap %": 100 * count / cluster_size_vox,
            "Peak -log10(p)": np.max(stat_data[cluster_mask]),
            "Cluster p-value": 10 ** (-np.max(stat_data[cluster_mask])),
        })

overlap_df = pd.DataFrame(rows)
overlap_df = overlap_df.sort_values(["Cluster", "Overlap voxels"], ascending=[True, False])

main_labels = overlap_df.loc[overlap_df.groupby("Cluster")["Overlap voxels"].idxmax()].reset_index(drop=True)


# binary mask
binary_img = math_img(
    f"(img > {neglog_alpha_05}).astype(int)",
    img=stat_resampled
)

surf_data = flatmap.vol_to_surf(
    binary_img,
    space="MNI"
)

# re-binarize after surface projection
surf_data = (surf_data > 0).astype(int)

flatmap.plot(
    surf_data,
    overlay_type="label",
    cmap="autumn",
    colorbar=False,
    render="matplotlib"
)

plt.show()
