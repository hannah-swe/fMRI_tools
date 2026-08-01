import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_mask_filename,
                  get_output_path, define_group_comparison, get_selected_subject_list, get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import get_cluster_table_with_aal_labels
from PPPD.config import load_config
from PPPD.plotting import get_colorbar_limits
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img, new_img_like
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings
import gc
from scipy.stats import norm


# ---- Load script configuration from config.yml
config = load_config()

task = config["analysis"]["task"]
runs = config["analysis"]["runs"] # "run-01" == pre, "run-02" == post
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
                                    # "VermisUvulaL", "VermisVII",
                                    #  "HippocampusL", "HippocampusR"
group_comparison = config["analysis"]["group_comparison"] # supported comparisons: "pat>HC", "HC>pat"
mask_strategy = config["mask"]["strategy"] # supported strategies: "subject_based", "predefined"
predefined_mask = config["mask"]["predefined_mask"]
threshold_mask = config["mask"]["threshold"] # only used if mask_strategy == "subject_based"
n_perm = config["statistics"]["n_perm"] # number of permutations for non-parametric cluster-based permutation test


print(f"Used configuration parameters:\n"
      f"task = {task}\nruns = {runs}\npart = {part}\nfeature = {feature}\nseeds = {seeds}\n"
      f"group_comparison = {group_comparison}\nmask_strategy = {mask_strategy}\nn_perm = {n_perm}\n")


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")
participants_df["part"] = participants_df["participant_id"].apply(lambda x: 1 if x < 100 else 2)

# path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)


# --- Group mapping for contrast via predefined comparison strategy:
group_mapping = define_group_comparison(group_comparison)


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"\n=== Running seed: {seed} ===")

    # get output path
    output_dir = get_output_path(part, feature, seed)
    output_dir = os.path.join(output_dir, "pre_post_diff_group_effects")
    os.makedirs(output_dir, exist_ok=True)


    # --- Define the file suffix
    if feature == "seed_based":
        base_title = f"{feature} {seed}"
        file_suffix = f"{feature}_{seed}"
    else:
        base_title = f"{feature}"
        file_suffix = f"{feature}"
    if part is None:
        part_label = "all"
        base_title = f"{base_title}; subjects: {part_label}"
    else:
        part_label = f"{part}"
        base_title = f"{base_title}; subjects part: {part_label}"
    file_suffix = f"{file_suffix}_{part_label}"


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
            filename = get_full_filename(subject_id, task, run, feature, seed)
            img_path = os.path.join(deriv_dir, subject_id, filename)
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
                mask_filename = get_mask_filename(subject_id, task, run, feature, seed)
                mask_path = os.path.join(deriv_dir, subject_id, mask_filename)
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
        diff_dir = os.path.join(output_dir, "diff_images")
        os.makedirs(diff_dir, exist_ok=True)
        diff_path = os.path.join(diff_dir, diff_filename)
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
        analysis_mask = get_mask_file(predefined_mask)
    else:
        raise ValueError(f"Unknown mask strategy {mask_strategy}")


    # --- Get design matrix for pre-post model and plot it
    design_df = included_df.merge(participants_df[["subject_id", "group", "part"]], on="subject_id", how="left")
    design_df["group_code"] = design_df["group"].map(group_mapping)
    if design_df["group_code"].isna().any():
        print(design_df[design_df["group_code"].isna()][["subject_id", "group"]])
        raise ValueError("Some subjects have missing or unmapped group labels.")
    design_df["part_code"] = design_df["part"].map({
        1: -1,
        2: 1
    })
    # Keep exact image order
    diff_imgs = design_df["diff_img"].tolist()

    seed_base_title = base_title
    seed_file_suffix = file_suffix

    # Output directories
    uncorrected_dir = os.path.join(output_dir, "01_uncorrected")
    nonparametric_dir = os.path.join(output_dir, "02_nonparametric")
    cluster_table_dir = os.path.join(output_dir, "cluster_tables")
    posthoc_mask_dir = os.path.join(output_dir, "sig_cluster_masks")

    for directory in [
        uncorrected_dir,
        nonparametric_dir,
        cluster_table_dir,
        posthoc_mask_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    for direction in ["positive", "negative"]:
        os.makedirs(
            os.path.join(posthoc_mask_dir, direction),
            exist_ok=True
        )

    # Two-sided uncorrected voxel threshold p < .001
    voxel_p = 0.001
    z_threshold_p001 = norm.isf(voxel_p / 2)

    # Cluster-level corrected alpha
    cluster_alpha = 0.05
    neglog_alpha = -np.log10(cluster_alpha)

    for group_name in design_df["group"].dropna().unique():

        group_base_title = f"{seed_base_title}; {group_name}"
        group_file_suffix = f"{seed_file_suffix}_{group_name}"

        group_selection = design_df["group"].eq(group_name)

        # Gemeinsames gruppenspezifisches DataFrame:
        # Dadurch bleiben Bilder, Part-Codes und Subject-IDs exakt ausgerichtet.
        group_df = (
            design_df.loc[
                group_selection,
                ["subject_id", "diff_img", "part_code"]
            ]
            .reset_index(drop=True)
        )

        group_diff_imgs = group_df["diff_img"].tolist()
        n_group = len(group_df)

        print(f"\nGroup {group_name}: n = {n_group}")
        print(group_df["part_code"].value_counts(dropna=False))

        if n_group < 2:
            print(f"Skipping {group_name}: fewer than two subjects.")
            continue

        if group_df["part_code"].isna().any():
            print(
                group_df.loc[
                    group_df["part_code"].isna(),
                    ["subject_id", "part_code"]
                ]
            )
            raise ValueError(
                f"Missing part codes in group {group_name}."
            )

        group_design = pd.DataFrame({
            "intercept": np.ones(n_group, dtype=float),
            "part": group_df["part_code"].astype(float).to_numpy(),
        })

        print(group_design.head())
        print("Design shape:", group_design.shape)
        print("Number of images:", len(group_diff_imgs))

        # ----------------------------------------
        # Parametric one-sample test against zero
        # ----------------------------------------
        group_model = SecondLevelModel(
            mask_img=analysis_mask
        ).fit(
            group_diff_imgs,
            design_matrix=group_design
        )

        z_map = group_model.compute_contrast(
            "intercept",
            output_type="z_score"
        )

        stat_map = group_model.compute_contrast(
            "intercept",
            output_type="stat"
        )

        thresholded_map = threshold_img(
            z_map,
            threshold=z_threshold_p001,
            cluster_threshold=10,
            two_sided=True
        )

        thresholded_data = thresholded_map.get_fdata()
        has_uncorrected_clusters = np.any(thresholded_data != 0)

        if has_uncorrected_clusters:
            vmin, vmax = get_colorbar_limits(
                data=thresholded_data,
                threshold=z_threshold_p001,
                two_sided=True
            )

            fig = plt.figure(figsize=(9, 5))
            display = plot_glass_brain(
                thresholded_map,
                cmap="RdBu_r",
                threshold=z_threshold_p001,
                vmin=vmin,
                vmax=vmax,
                figure=fig,
                title=None,
                plot_abs=False,
                symmetric_cbar=True
            )

            display.frame_axes.figure.suptitle(
                "Difference z map: post − pre\n"
                f"{group_base_title}; two-sided p < .001; "
                "clusters >= 10 voxels"
            )

            display.savefig(
                os.path.join(
                    uncorrected_dir,
                    f"{group_file_suffix}_uncorrected_p001_cluster10.png"
                )
            )
            plt.close(fig)
        else:
            print("No uncorrected suprathreshold clusters.")

        # ----------------------------------------
        # Non-parametric one-sample test
        # ----------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            perm_out = non_parametric_inference(
                second_level_input=group_diff_imgs,
                design_matrix=group_design,
                second_level_contrast="intercept",
                mask=analysis_mask,
                model_intercept=False,
                n_perm=n_perm,
                two_sided_test=True,
                threshold=0.001,
                random_state=42,
                n_jobs=8,
                verbose=1
            )

        signed_logp_mass_thr = math_img(
            (
                f"np.where(logp > {neglog_alpha}, "
                "np.sign(stat) * logp, 0)"
            ),
            logp=perm_out["logp_max_mass"],
            stat=stat_map
        )

        signed_data = signed_logp_mass_thr.get_fdata()
        has_sig_clusters = np.any(
            np.abs(signed_data) > neglog_alpha
        )

        if not has_sig_clusters:
            print(
                "No clusters survive permutation "
                "cluster-mass FWER correction."
            )
            continue

        vmin, vmax = get_colorbar_limits(
            data=signed_data,
            threshold=neglog_alpha,
            two_sided=True
        )

        fig = plt.figure(figsize=(9, 5))
        display = plot_glass_brain(
            signed_logp_mass_thr,
            cmap="RdBu_r",
            threshold=neglog_alpha,
            vmin=vmin,
            vmax=vmax,
            plot_abs=False,
            symmetric_cbar=True,
            figure=fig,
            title=None,
            colorbar=True
        )

        display.frame_axes.figure.suptitle(
            "Post − pre: permutation cluster-mass FWER\n"
            f"{group_base_title}; corrected p < .05"
        )

        display.savefig(
            os.path.join(
                nonparametric_dir,
                f"{group_file_suffix}_perm_clustermass_fwer05.png"
            )
        )
        plt.close(fig)

        # ----------------------------------------
        # Cluster table
        # ----------------------------------------
        logp_mass_thr_float = math_img(
            "img.astype(float)",
            img=signed_logp_mass_thr
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cluster_table, label_maps = (
                get_cluster_table_with_aal_labels(
                    stat_img=logp_mass_thr_float,
                    stat_threshold=neglog_alpha,
                    cluster_threshold=0,
                    two_sided=True,
                    return_label_maps=True
                )
            )

        if cluster_table.empty:
            print(
                "Cluster table is empty despite significant voxels."
            )
            continue

        cluster_p_values = (
                10 ** (-cluster_table["Peak Stat"].abs())
        )

        peak_idx = cluster_table.columns.get_loc("Peak Stat") + 1
        cluster_table.insert(
            peak_idx,
            "p-value",
            cluster_p_values
        )

        cluster_table = cluster_table.rename(columns={
            "Cluster ID": "Cluster",
            "Peak Stat": "Stat",
            "Cluster Size (mm3)": "Size (mm3)"
        })

        cluster_table.to_csv(
            os.path.join(
                cluster_table_dir,
                f"{group_file_suffix}_cluster_table_perm_mass.csv"
            ),
            index=False
        )

        # Inspect custom return structure
        if not isinstance(label_maps, (list, tuple)):
            label_maps = [label_maps]

        print(f"Found {len(label_maps)} label map(s).")

        # Save one binary image per cluster
        signed_full_data = signed_logp_mass_thr.get_fdata()

        for map_idx, label_map in enumerate(label_maps, start=1):
            label_data = label_map.get_fdata()

            cluster_ids = np.unique(label_data)
            cluster_ids = cluster_ids[cluster_ids > 0]

            for cluster_id in cluster_ids:
                cluster_bool = label_data == cluster_id

                mean_signed_value = np.nanmean(
                    signed_full_data[cluster_bool]
                )

                direction = (
                    "positive"
                    if mean_signed_value > 0
                    else "negative"
                )

                binary_cluster_img = new_img_like(
                    label_map,
                    cluster_bool.astype(np.uint8)
                )

                cluster_path = os.path.join(
                    posthoc_mask_dir,
                    direction,
                    (
                        f"{group_file_suffix}_{direction}_"
                        f"map-{map_idx:02d}_"
                        f"cluster-{int(cluster_id):03d}.nii.gz"
                    )
                )

                binary_cluster_img.to_filename(cluster_path)
                print(f"Saved cluster mask: {cluster_path}")


    # --- Clean up memory after each seed
    # plt.close("all")
    vars_to_delete = ["diff_imgs", "included_rows", "sub_mask_imgs", "included_df", "design_df", "analysis_mask",
                      "second_level_design", "group_selection", "group_diff_imgs", "group_design", "group_model",
                      "z_map", "stat_map", "thresholded_map", "thresholded_data", "perm_out", "signed_logp_mass_thr",
                      "signed_data", "logp_mass_thr_float", "cluster_table", "label_maps", "cluster_p_values",
                      "signed_full_data", "label_map", "label_data", "cluster_ids", "cluster_id", "cluster_bool",
                      "binary_cluster_img"]
    for var in vars_to_delete:
        if var in locals():
            del locals()[var]
    gc.collect()


