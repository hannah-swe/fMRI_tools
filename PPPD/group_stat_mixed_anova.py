import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (get_derivatives_path, get_participants_tsv, get_full_filename, get_mask_filename,
                  get_output_pre_post_path, define_group_comparison, get_selected_subject_list, get_mask_file)
from PPPD.subjects import subs, subjects_to_exclude
from PPPD.utils import get_cluster_table_with_aal_labels
from PPPD.config import load_config
from PPPD.plotting import get_colorbar_limits
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import threshold_img, math_img
from nilearn.plotting import plot_stat_map, plot_design_matrix, plot_glass_brain
from nilearn.masking import intersect_masks
import numpy as np
import pandas as pd
import warnings
import gc


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
                                    # "HippocampusL", "HippocampusR"
                                    # "PrecuneusL", "PrecuneusR"
group_comparison = config["analysis"]["group_comparison"] # supported comparisons: "pat>HC", "HC>pat"
mask_strategy = config["mask"]["strategy"] # supported strategies: "subject_based", "predefined"
predefined_mask = config["mask"]["predefined_mask"]
threshold_mask = config["mask"]["threshold"] # only used if mask_strategy == "subject_based"
n_perm = config["statistics"]["n_perm"] # number of permutations for non-parametric cluster-based permutation test
effect = config["analysis"]["effect"]


# --- Validate effect
valid_effects = {"intercept", "group", "part", "groupxpart"}

if effect not in valid_effects:
    raise ValueError(
        f"Unknown effect '{effect}'. "
        f"Supported effects: {sorted(valid_effects)}"
    )

# Part and group x part effects require subjects from both parts
if part is not None and effect in {"part", "groupxpart"}:
    raise ValueError(
        f"effect='{effect}' requires subjects from both parts, "
        f"but config analysis.part={part}. "
        f"Set analysis.part to null."
    )

print(f"Used configuration parameters:\n"
      f"task = {task}\nruns = {runs}\npart = {part}\nfeature = {feature}\nseeds = {seeds}\n"
      f"group_comparison = {group_comparison}\neffect = {effect}\nmask_strategy = {mask_strategy}\nn_perm = {n_perm}\n")


# --- Load participants.tsv
participants_df = get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")
participants_df["part"] = participants_df["participant_id"].apply(lambda x: 1 if x < 100 else 2)


# --- Path to halfpipe derivatives directory
deriv_dir = get_derivatives_path(feature)


# --- Get output path
output_dir = get_output_pre_post_path(part, feature)
os.makedirs(output_dir, exist_ok=True)


# --- Group mapping for contrast via predefined comparison strategy:
group_mapping = define_group_comparison(group_comparison)


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Loop over seeds
for seed in seeds:
    print(f"\n=== Running seed: {seed} ===")


    # --- Define the file suffix
    if feature == "seed_based":
        base_title = f"{feature} {seed}"
        file_suffix = f"{feature}_{seed}_{group_comparison}"
    else:
        base_title = f"{feature}"
        file_suffix = f"{feature}_{group_comparison}"
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
    design_df["groupxpart"] = (design_df["group_code"] * design_df["part_code"])
    # keep exact image order
    diff_imgs = design_df["diff_img"].tolist()
    # intercept: main effect stimulation/time
    # group: 2-way interaction stimulation x group
    # part: 2-way interaction stimulation x part
    # groupxpart: 3-way interaction stimulation x group x part
    second_level_design = pd.DataFrame({
        "intercept": np.ones(len(design_df)),
        "group": design_df["group_code"].astype(float),
        "part": design_df["part_code"].astype(float),
        "groupxpart": design_df["groupxpart"].astype(float),
    })
    print("Number of difference images:", len(diff_imgs))
    print("Design matrix shape:", second_level_design.shape)
    plot_design_matrix(second_level_design)
    plt.show()


    # --- Get statistical output folders for different effects
    stats_output_dir = os.path.join(output_dir, effect)
    os.makedirs(stats_output_dir, exist_ok=True)


    # --- PARAMETRIC TEST (voxel-wise two sample t-test unpaired)
    # fit model (here: z-scores are used, also possible: 'z_score', 'stat', 'p_value', 'effect_size', 'effect_variance', 'all')
    second_level_model = SecondLevelModel(mask_img=analysis_mask)
    second_level_model = second_level_model.fit(diff_imgs, design_matrix=second_level_design)
    z_map = second_level_model.compute_contrast(effect, output_type='z_score')

    # Significance test:
    # Version 1: abs(z) > 3.09 (equivalent to p < 0.001 one-sided test), cluster size > 10 voxels
    # z(threshold)=3.09 for p=0.001 when testing one-sided; 3.29 for two-sided
    z_threshold_p001 = 3.09
    thresholded_map1 = threshold_img(z_map, cluster_threshold=10, threshold=z_threshold_p001, two_sided=True)
    thr1_data = thresholded_map1.get_fdata()
    # plot thresholded maps if there are any voxels/clusters left
    if np.any(thr1_data != 0):
        # get colorbar thresholds
        vmin, vmax = get_colorbar_limits(data=thr1_data, threshold=z_threshold_p001, two_sided=True)
        # plot_stat_map(thresholded_map1, display_mode='mosaic', cmap="RdBu_r",
        # title=f"difference z map (post - pre) \n {base_title}; z > 3.09; clusters > 10 voxels")
        # plt.show()
        fig = plt.figure(figsize=(9,5))
        display = plot_glass_brain(thresholded_map1, cmap="RdBu_r", threshold=z_threshold_p001, vmin=vmin, vmax=vmax,
                                   figure=fig, title=None, plot_abs=False, symmetric_cbar=True)
        display.frame_axes.figure.suptitle(f"difference z map (post - pre) \n {base_title}; "
                                           f"|z| > {z_threshold_p001}; clusters > 10 voxels")
        stats_output_path = os.path.join(stats_output_dir, "01_uncorrected")
        os.makedirs(stats_output_path, exist_ok=True)
        display.savefig(os.path.join(stats_output_path, f"{file_suffix}_uncorrected_p001_cluster10.png"))
        plt.show()
    else:
        print("No suprathreshold clusters; skipping plots.")

    # get cluster table with anatomical labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster_table1 = get_cluster_table_with_aal_labels(
            stat_img=thresholded_map1,
            stat_threshold=3.09,
            cluster_threshold=10,
            two_sided=True
        )


    # --- NON-PARAMETRIC TESTS: permutation inference with cluster-level correction
    # threshold is in p-scale, not z-scale;
    # threshold=0.001 corresponds to a cluster-forming threshold of p < .001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        perm_out = non_parametric_inference(
            second_level_input=diff_imgs,
            design_matrix=second_level_design,
            second_level_contrast=effect,
            mask=analysis_mask,
            model_intercept=False,  # intercept is already in the design matrix
            n_perm=n_perm,
            two_sided_test=True,
            threshold=0.001,  # cluster-forming threshold in p-scale
            random_state=42,  # reproducible permutations
            n_jobs=8,
            verbose=1,
        )

    # -------------------------------------------------------------------------
    # Prepare non-parametric results
    # -------------------------------------------------------------------------

    # corrected p < .05  <=>  -log10(p) > 1.30103
    neglog_alpha_05 = -np.log10(0.05)

    # cluster-mass FWER corrected significance map
    logp_img = perm_out["logp_max_mass"]

    # t-statistic from the same non-parametric model
    t_img = perm_out["t"]

    logp_data = logp_img.get_fdata()
    t_data = t_img.get_fdata()

    # voxels belonging to cluster-mass FWER significant clusters
    sig_mask = logp_data > neglog_alpha_05

    has_sig_clusters = np.any(sig_mask)

    # -------------------------------------------------------------------------
    # 03_tvalues
    # Diagnostic plot:
    # t-statistics only inside cluster-mass FWER significant clusters
    # -------------------------------------------------------------------------

    tvalue_output_dir = os.path.join(stats_output_dir, "02_tvalues")
    os.makedirs(tvalue_output_dir, exist_ok=True)

    if has_sig_clusters:

        signed_t_sig = math_img(
            f"np.where(logp > {neglog_alpha_05}, t, 0)",
            logp=logp_img,
            t=t_img,
        )

        # optional diagnostic information
        sig_t_values = t_data[sig_mask]

        print(
            f"T-values inside significant clusters:\n"
            f"  significant voxels = {len(sig_t_values)}\n"
            f"  positive t voxels  = {np.sum(sig_t_values > 0)}\n"
            f"  negative t voxels  = {np.sum(sig_t_values < 0)}\n"
            f"  min t              = {np.nanmin(sig_t_values):.3f}\n"
            f"  max t              = {np.nanmax(sig_t_values):.3f}"
        )

        fig = plt.figure(figsize=(9, 5))

        display = plot_glass_brain(
            signed_t_sig,
            threshold=0,
            plot_abs=False,
            symmetric_cbar=True,
            cmap="RdBu_r",
            figure=fig,
            title=None,
            colorbar=True,
        )

        display.frame_axes.figure.suptitle(
            f"t-statistics within cluster-mass FWER significant clusters\n"
            f"{base_title}; effect: {effect} | corrected p < .05"
        )

        display.savefig(
            os.path.join(
                tvalue_output_dir,
                f"{file_suffix}_{effect}_tvalues_in_significant_clusters.png",
            )
        )

        display.close()
        plt.close(fig)

    else:
        signed_t_sig = None
        print("No significant clusters; skipping t-value plot.")

    # -------------------------------------------------------------------------
    # Create unsigned thresholded cluster-mass significance image
    # -------------------------------------------------------------------------

    logp_mass_thr = threshold_img(
        logp_img,
        threshold=neglog_alpha_05,
        two_sided=False,  # logp_max_mass itself is unsigned
    )

    # -------------------------------------------------------------------------
    # Identify significant clusters
    # -------------------------------------------------------------------------

    if not has_sig_clusters:

        print("No clusters survive permutation cluster-mass FWER correction.")

        cluster_table_perm_mass = None
        label_maps = None
        signed_logp_mass_thr = None

    else:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cluster_table_perm_mass, label_maps = (
                get_cluster_table_with_aal_labels(
                    stat_img=logp_mass_thr,
                    stat_threshold=neglog_alpha_05,
                    cluster_threshold=0,
                    two_sided=False,
                    return_label_maps=True,
                )
            )

        # ---------------------------------------------------------------------
        # Get one cluster-ID image
        # ---------------------------------------------------------------------

        if (
                cluster_table_perm_mass.empty
                or label_maps is None
                or len(label_maps) == 0
        ):
            print(
                "No cluster label map returned despite significant voxels. "
                "Skipping cluster-wise direction assignment."
            )

            signed_logp_mass_thr = None

        else:

            # The returned label image contains integer cluster IDs:
            # background = 0, clusters = 1, 2, 3, ...
            label_img = label_maps[0]
            label_data = label_img.get_fdata()

            cluster_ids = np.unique(label_data)
            cluster_ids = cluster_ids[cluster_ids != 0]

            print(f"Found {len(cluster_ids)} significant cluster(s).")

            # -----------------------------------------------------------------
            # Assign ONE direction to each complete cluster
            # based on mean non-parametric t-value
            # -----------------------------------------------------------------

            signed_logp_data = np.zeros_like(logp_data, dtype=float)

            cluster_directions = []
            cluster_mean_t_values = []

            # separate true cluster-ID maps for positive and negative clusters
            positive_cluster_ids = np.zeros_like(label_data, dtype=np.int32)
            negative_cluster_ids = np.zeros_like(label_data, dtype=np.int32)

            for cluster_id in cluster_ids:

                cluster_mask = label_data == cluster_id

                # t-values inside this cluster
                cluster_t_values = t_data[cluster_mask]

                # mean t determines direction of the whole cluster
                mean_t = np.nanmean(cluster_t_values)

                if mean_t > 0:
                    direction = "positive"
                    sign = 1

                    positive_cluster_ids[cluster_mask] = int(cluster_id)

                elif mean_t < 0:
                    direction = "negative"
                    sign = -1

                    negative_cluster_ids[cluster_mask] = int(cluster_id)

                else:
                    # should be extremely unlikely, but handle explicitly
                    direction = "zero"
                    sign = 0

                # Give all voxels of this cluster the same direction
                signed_logp_data[cluster_mask] = (
                        sign * logp_data[cluster_mask]
                )

                cluster_mean_t_values.append(mean_t)
                cluster_directions.append(direction)

                print(
                    f"Cluster {int(cluster_id)}: "
                    f"mean t = {mean_t:.3f} -> {direction}"
                )

            # -----------------------------------------------------------------
            # Convert cluster-wise signed logp data back to NIfTI image
            # -----------------------------------------------------------------

            signed_logp_mass_thr = nib.Nifti1Image(
                signed_logp_data.astype(np.float32),
                affine=logp_img.affine,
                header=logp_img.header.copy(),
            )

            # -----------------------------------------------------------------
            # Save thresholded cluster-wise signed logp map
            # -----------------------------------------------------------------

            perm_mask_dir = os.path.join(
                stats_output_dir,
                "tresh_cluster_masks"
            )
            os.makedirs(perm_mask_dir, exist_ok=True)

            signed_logp_mass_path = os.path.join(
                perm_mask_dir,
                f"{file_suffix}_signed_logp_clustermass_fwer05.nii.gz"
            )

            signed_logp_mass_thr.to_filename(signed_logp_mass_path)

            print(
                f"Saved thresholded signed cluster-mass FWER map: "
                f"{signed_logp_mass_path}"
            )

            # -----------------------------------------------------------------
            # 03_nonparametric
            # Plot cluster-mass FWER corrected map with cluster-wise direction
            # -----------------------------------------------------------------

            nonparam_output_dir = os.path.join(
                stats_output_dir,
                "03_nonparametric",
            )
            os.makedirs(nonparam_output_dir, exist_ok=True)

            signed_data = signed_logp_mass_thr.get_fdata()

            vmin, vmax = get_colorbar_limits(
                data=signed_data,
                threshold=neglog_alpha_05,
                two_sided=True,
            )

            fig = plt.figure(figsize=(9, 5))

            display = plot_glass_brain(
                signed_logp_mass_thr,
                cmap="RdBu_r",
                threshold=neglog_alpha_05,
                vmin=vmin,
                vmax=vmax,
                plot_abs=False,
                symmetric_cbar=True,
                figure=fig,
                title=None,
                colorbar=True,
            )

            display.frame_axes.figure.suptitle(
                f"difference map permutation test cluster-mass FWER\n"
                f"{base_title}; effect: {effect} | corrected p < .05"
            )

            display.savefig(
                os.path.join(
                    nonparam_output_dir,
                    f"{file_suffix}_{effect}_perm_clustermass_fwer05.png",
                )
            )

            display.close()
            plt.close(fig)

            # -----------------------------------------------------------------
            # Cluster table
            # -----------------------------------------------------------------

            # Convert -log10(p) back to corrected p-value.
            # The cluster table was generated from the UNSIGNED logp image.
            cluster_table_perm_mass["p-value"] = (
                    10 ** (-np.abs(cluster_table_perm_mass["Peak Stat"]))
            )

            # Add direction information
            #
            # This assumes rows of the cluster table correspond to the returned
            # cluster IDs. If your helper returns additional subpeaks, see note
            # below.
            if len(cluster_table_perm_mass) == len(cluster_ids):

                cluster_table_perm_mass["Mean t"] = cluster_mean_t_values
                cluster_table_perm_mass["Direction"] = cluster_directions

            else:
                print(
                    "WARNING: Number of rows in cluster table does not equal "
                    "number of cluster IDs. Direction columns were not added "
                    "automatically."
                )

            cluster_table_perm_mass = cluster_table_perm_mass.rename(
                columns={
                    "Cluster ID": "Cluster",
                    "Peak Stat": "Stat",
                    "Cluster Size (mm3)": "Size (mm3)",
                }
            )

            cluster_table_dir = os.path.join(
                stats_output_dir,
                "cluster_tables",
            )
            os.makedirs(cluster_table_dir, exist_ok=True)

            cluster_table_path = os.path.join(
                cluster_table_dir,
                f"{file_suffix}_{effect}_cluster_table_perm_mass.csv",
            )

            cluster_table_perm_mass.to_csv(
                cluster_table_path,
                index=False,
            )

            # -----------------------------------------------------------------
            # Save significant cluster masks
            # -----------------------------------------------------------------

            posthoc_mask_dir = os.path.join(
                stats_output_dir,
                "sig_cluster_masks",
            )

            positive_dir = os.path.join(
                posthoc_mask_dir,
                "positive",
            )

            negative_dir = os.path.join(
                posthoc_mask_dir,
                "negative",
            )

            signed_dir = os.path.join(
                posthoc_mask_dir,
                "signed",
            )

            os.makedirs(positive_dir, exist_ok=True)
            os.makedirs(negative_dir, exist_ok=True)
            os.makedirs(signed_dir, exist_ok=True)

            # -----------------------------------------------------------------
            # Save signed cluster-mass significance map
            # -----------------------------------------------------------------

            signed_logp_mass_path = os.path.join(
                signed_dir,
                f"{file_suffix}_{effect}_signed_logp_clustermass_fwer05.nii.gz",
            )

            signed_logp_mass_thr.to_filename(
                signed_logp_mass_path
            )

            # -----------------------------------------------------------------
            # Save positive cluster-ID map
            # -----------------------------------------------------------------

            if np.any(positive_cluster_ids > 0):

                positive_cluster_img = nib.Nifti1Image(
                    positive_cluster_ids,
                    affine=label_img.affine,
                    header=label_img.header.copy(),
                )

                positive_cluster_path = os.path.join(
                    positive_dir,
                    f"{file_suffix}_{effect}_positive_cluster_id_map.nii.gz",
                )

                positive_cluster_img.to_filename(
                    positive_cluster_path
                )

                print("Saved positive cluster ID map.")

            else:
                positive_cluster_img = None
                print("No positive significant clusters.")

            # -----------------------------------------------------------------
            # Save negative cluster-ID map
            # -----------------------------------------------------------------

            if np.any(negative_cluster_ids > 0):

                negative_cluster_img = nib.Nifti1Image(
                    negative_cluster_ids,
                    affine=label_img.affine,
                    header=label_img.header.copy(),
                )

                negative_cluster_path = os.path.join(
                    negative_dir,
                    f"{file_suffix}_{effect}_negative_cluster_id_map.nii.gz",
                )

                negative_cluster_img.to_filename(
                    negative_cluster_path
                )

                print("Saved negative cluster ID map.")

            else:
                negative_cluster_img = None
                print("No negative significant clusters.")


    # --- Clean up memory after each seed
    try:
        display.close()
    except (NameError, AttributeError):
        pass

    plt.close("all")

    # Always existing large objects
    del diff_imgs
    del included_rows
    del included_df
    del sub_mask_imgs
    del analysis_mask
    del design_df
    del second_level_design

    del second_level_model
    del z_map
    del thresholded_map1
    del thr1_data

    del perm_out
    del logp_img
    del t_img
    del logp_data
    del t_data
    del sig_mask
    del logp_mass_thr

    # Objects that only exist in certain branches
    for name in [
        "diff_img",
        "signed_t_sig",
        "sig_t_values",
        "cluster_table_perm_mass",
        "label_maps",
        "label_img",
        "label_data",
        "cluster_ids",
        "signed_logp_data",
        "signed_logp_mass_thr",
        "signed_data",
        "cluster_t_values",
        "cluster_mask",
        "cluster_mean_t_values",
        "cluster_directions",
        "positive_cluster_ids",
        "negative_cluster_ids",
        "positive_cluster_img",
        "negative_cluster_img",
    ]:
        if name in globals():
            del globals()[name]

    gc.collect()