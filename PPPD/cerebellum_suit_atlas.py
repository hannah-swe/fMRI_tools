import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
import nibabel as nib
from nilearn.image import resample_to_img, math_img
from PPPD import get_output_path, get_suit_atlas
from PPPD.config import load_config
from scipy import ndimage

# ---- Load script configuration from config.yml --> don't change anything here, make all configurations in config.yml
config = load_config()

task = config["analysis"]["task"]
# runs = config["analysis"]["runs"] # "run-01" == pre, "run-02" == post
# run = "run-01"
part = config["analysis"]["part"] # supported: None, 1, 2 (None: all subjects; 1: subjects < 100; 2: subjects >= 100)
feature = config["analysis"]["feature"] # supported features: "falff", "seed_based", "alff"
seeds = config["analysis"]["seeds"] # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R",
                                    # "OperculumOP4L", "OperculumOP4R", "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII"
group_comparison = config["analysis"]["group_comparison"] # supported comparisons: "pat>HC", "HC>pat"
t_test_strategy = config["statistics"]["t_test_strategy"] # either two-sided or one-sided
pre_post_diff = False
direction = "negative" # possible directions for pre-post differences:
                        # "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)

if t_test_strategy == "two_sided":
    test_label = "twosided"
elif t_test_strategy == "one_sided":
    test_label = "onesided"
else:
    raise ValueError(f"Unknown t_test_strategy: {t_test_strategy}. Use 'two_sided' or 'one_sided'.")

print(f"Used configuration parameters:\n"
      f"task = {task}\npart = {part}\nfeature = {feature}\nseeds = {seeds}\n"
      f"group_comparison = {group_comparison}\n"
      f"t_test_strategy = {t_test_strategy}\n")


# --- Get path to lut file and load atlas image
lut_file, atlas_img = get_suit_atlas()


# --- Store all resampled cluster maps for combined overlap plot
combined_cluster_maps = []

# --- Loop over seeds
for seed in seeds:
    print(f"\n=== Running seed: {seed} ===")


    # --- All directories
    # get data dir
    data_dir = get_output_path(part, feature, seed)

    # get output path to save results
    if pre_post_diff is True:
        output_dir = os.path.join(data_dir, "pre_post_diff", "cerebellum_labeling")
    else:
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
    diff_file_suffix = f"{file_suffix}_{part_label}"
    file_suffix = f"{file_suffix}_{part_label}_{test_label}"


    # --- Load significant cluster map and matching permutation table
    if pre_post_diff is True:
        if direction not in ["positive", "negative"]:
            raise ValueError("For pre_post_diff=True, direction must be 'positive' or 'negative'.")
        sig_cluster_dir = os.path.join(data_dir, "pre_post_diff", "sig_cluster_masks", f"{direction}")
        cluster_map_path = os.path.join(sig_cluster_dir, f"{diff_file_suffix}_{direction}_cluster_id_map.nii.gz")
        cluster_table_path = os.path.join(data_dir, "pre_post_diff", "cluster_tables",
                                          f"{diff_file_suffix}_cluster_table_perm_mass.csv")

    else:
        sig_cluster_dir = os.path.join(data_dir, "sig_cluster_masks")
        cluster_map_path = os.path.join(sig_cluster_dir, f"{file_suffix}_cluster_id_map.nii.gz")
        cluster_table_path = os.path.join(data_dir, "cluster_tables", f"{file_suffix}_cluster_table_perm_mass.csv")

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

    # stat_img is already a cluster-ID map
    cluster_data = np.rint(stat_data).astype(int)

    combined_cluster_maps.append({
        "seed": seed,
        "cluster_map": cluster_data.copy(),
        "ref_img": stat_resampled
    })

    cluster_ids = sorted(np.unique(cluster_data))
    cluster_ids = [c for c in cluster_ids if c != 0]

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


# --- Plot all clusters from all seeds together + overlap count
if len(combined_cluster_maps) > 0:

    combined_output_dir = os.path.join(
        get_output_path(part, feature, seed) if feature == "seed_based"
        else get_output_path(part, feature, None),
        "cerebellum_labeling",
        "combined_seed_overlap_plot"
    )
    os.makedirs(combined_output_dir, exist_ok=True)

    cluster_surfaces = []
    cluster_labels = []

    # --- Project every original cluster separately to surface
    for entry in combined_cluster_maps:

        seed_name = entry["seed"]
        cluster_map = entry["cluster_map"]
        ref_img = entry["ref_img"]

        cluster_ids_this_seed = sorted(np.unique(cluster_map))
        cluster_ids_this_seed = [c for c in cluster_ids_this_seed if c != 0]

        for cluster_id in cluster_ids_this_seed:

            binary_cluster = (cluster_map == cluster_id).astype(np.uint8)

            cluster_img = nib.Nifti1Image(
                binary_cluster,
                affine=ref_img.affine,
                header=ref_img.header
            )
            cluster_img.set_data_dtype(np.uint8)

            cluster_surf = flatmap.vol_to_surf(
                cluster_img,
                space="MNI",
                stats="nanmean"
            )

            cluster_surfaces.append(cluster_surf > 0)
            cluster_labels.append(f"{seed_name}_cluster-{cluster_id}")

    print(f"Collected {len(cluster_surfaces)} clusters across all seeds")

    # --- Build combined surface maps
    template = cluster_surfaces[0]

    combined_label_surf = np.zeros_like(template, dtype=int)
    overlap_count_surf = np.zeros_like(template, dtype=int)

    for i, cluster_mask_surf in enumerate(cluster_surfaces, start=1):

        # counts how many original clusters overlap at each surface point
        overlap_count_surf[cluster_mask_surf] += 1

        # stores one visible cluster ID per surface point
        # if clusters overlap, the later one is shown in the label plot
        combined_label_surf[cluster_mask_surf] = i

    # --- Save cluster label lookup table
    label_lookup = pd.DataFrame({
        "Plot ID": np.arange(1, len(cluster_labels) + 1),
        "Seed cluster": cluster_labels
    })

    label_lookup_path = os.path.join(
        combined_output_dir,
        f"{feature}_{group_comparison}_{part_label}_{test_label}_combined_cluster_labels.csv"
    )
    label_lookup.to_csv(label_lookup_path, index=False)

    print("Saved cluster label lookup:")
    print(label_lookup_path)

    # ============================================================
    # Plot 1: all clusters, different colors
    # ============================================================
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap("tab10")

    custom_cmap = ListedColormap([
        cmap(0),
        cmap(0), #
        cmap(6), #
        cmap(3),
        cmap(8), #
        cmap(9), #
        cmap(4), #
        cmap(1), #
        cmap(8),
        cmap(9),
    ])
    flatmap.plot(
        combined_label_surf,
        overlay_type="label",
        cmap=custom_cmap,
        colorbar=False,
        render="matplotlib",
        alpha=1
    )

    combined_label_plot_path = os.path.join(
        combined_output_dir,
        f"{feature}_{group_comparison}_{part_label}_{test_label}_all_seeds_colored_clusters_flatmap.png"
    )

    plt.savefig(combined_label_plot_path, dpi=600)
    plt.show()
    print("Saved combined colored cluster flatmap:")
    print(combined_label_plot_path)