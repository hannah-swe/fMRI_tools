from __future__ import annotations
from pathlib import Path
import pandas as pd
from pppd.config.paths import HALFPIPE_DIRS, get_fmriprep_dir
from pppd.config.masks import dmn_mask, roi_masks
from pppd.metrics.nifti import mean_in_mask_nii, weighted_mean_in_probseg
from pppd.metrics.roi import extract_roi_means
from pppd.io.features import seedfc_map_path, falff_map_path
from pppd.io.fmriprep import (
    gm_probseg_mni,
    wm_probseg_mni,
    csf_probseg_mni,
    brain_mask_mni,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

TISSUE_THRESHOLD = 0.2   # Threshold for probabilistic tissue masks
USE_WEIGHTED = False     # False -> thresholded mask; True -> probabilistic weighted mean

threshold = None if USE_WEIGHTED else TISSUE_THRESHOLD


def build_feature_rows(
    base_dirs: tuple[Path, ...],
    feature: str,
    pipelines: list[str],
    run: str = "run-01",
) -> pd.DataFrame:
    """
    Create a per-subject, per-pipeline feature table.

    Searches all provided HALFpipe derivative directories.

    Output columns include:
        sub
        pipeline
        <feature>_dmn_mean
        <feature>_gm_mean
        <feature>_wm_mean
        <feature>_csf_mean
        <feature>_wb_mean
        ROI means
    """

    if feature not in {"seedfc", "falff"}:
        raise ValueError("feature must be one of: {'seedfc', 'falff'}")

    map_path_fn = seedfc_map_path if feature == "seedfc" else falff_map_path
    dmn_col = f"{feature}_dmn_mean"

    rows: list[dict] = []

    # -----------------------------------------------------------------
    # Search subjects across both HALFpipe derivative directories
    # -----------------------------------------------------------------

    for base_dir in base_dirs:
        sub_dirs = sorted(base_dir.glob("sub-*"))

        if not sub_dirs:
            print(f"[WARNING] No sub-* directories found in: {base_dir}")
            continue

        print(f"\nSearching HALFpipe directory: {base_dir}")
        print(f"Found {len(sub_dirs)} subject directories.")

        for sub_dir in sub_dirs:
            sub = sub_dir.name

            # Determine corresponding fMRIPrep directory for this subject
            fmri_sub_dir = get_fmriprep_dir(sub) / sub

            if not fmri_sub_dir.exists():
                print(
                    f"[WARNING] Missing fMRIPrep directory for {sub}: "
                    f"{fmri_sub_dir}"
                )

            for pipeline in pipelines:
                img_path = map_path_fn(
                    sub_dir=sub_dir,
                    sub=sub,
                    pipeline=pipeline,
                    run=run,
                )

                if not img_path.exists():
                    print(
                        f"[SKIP] Missing {feature} map for "
                        f"{sub} ({pipeline}): {img_path}"
                    )
                    continue

                row = {
                    "sub": sub,
                    "pipeline": pipeline,
                    dmn_col: mean_in_mask_nii(
                        img_path,
                        dmn_mask,
                    ),
                }

                # -----------------------------------------------------
                # Tissue-specific global means from fMRIPrep derivatives
                # -----------------------------------------------------

                gm_path = gm_probseg_mni(fmri_sub_dir, sub)

                if gm_path.exists():
                    row[f"{feature}_gm_mean"] = weighted_mean_in_probseg(
                        img_path,
                        gm_path,
                        threshold=threshold,
                    )
                else:
                    row[f"{feature}_gm_mean"] = float("nan")
                    print(f"[WARNING] Missing GM probseg for {sub}: {gm_path}")

                wm_path = wm_probseg_mni(fmri_sub_dir, sub)

                if wm_path.exists():
                    row[f"{feature}_wm_mean"] = weighted_mean_in_probseg(
                        img_path,
                        wm_path,
                        threshold=threshold,
                    )
                else:
                    row[f"{feature}_wm_mean"] = float("nan")
                    print(f"[WARNING] Missing WM probseg for {sub}: {wm_path}")

                csf_path = csf_probseg_mni(fmri_sub_dir, sub)

                if csf_path.exists():
                    row[f"{feature}_csf_mean"] = weighted_mean_in_probseg(
                        img_path,
                        csf_path,
                        threshold=threshold,
                    )
                else:
                    row[f"{feature}_csf_mean"] = float("nan")
                    print(f"[WARNING] Missing CSF probseg for {sub}: {csf_path}")

                brain_mask_path = brain_mask_mni(fmri_sub_dir, sub)

                if brain_mask_path.exists():
                    row[f"{feature}_wb_mean"] = mean_in_mask_nii(
                        img_path,
                        brain_mask_path,
                    )
                else:
                    row[f"{feature}_wb_mean"] = float("nan")
                    print(
                        f"[WARNING] Missing whole-brain mask for "
                        f"{sub}: {brain_mask_path}"
                    )

                # -----------------------------------------------------
                # ROI means
                # -----------------------------------------------------

                roi_means = extract_roi_means(
                    img_path,
                    roi_masks,
                )

                # Prefix ROI columns so they never collide across features
                roi_means = {
                    f"{feature}_{key}": value
                    for key, value in roi_means.items()
                }

                row.update(roi_means)

                rows.append(row)

    # -----------------------------------------------------------------
    # Build dataframe
    # -----------------------------------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"Feature table is empty for feature='{feature}'. "
            "Most likely the map filename patterns in io/features.py "
            "are not matching the current HALFpipe outputs."
        )

    # Protect against accidental duplicate subject/pipeline rows
    duplicates = df.duplicated(
        subset=["sub", "pipeline"],
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = df.loc[
            duplicates,
            ["sub", "pipeline"],
        ].sort_values(["sub", "pipeline"])

        raise RuntimeError(
            "Duplicate subject/pipeline combinations found while "
            "building the feature table.\n"
            "This may mean that the same subject exists in more than "
            "one HALFpipe root.\n\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return (
        df
        .sort_values(["sub", "pipeline"])
        .reset_index(drop=True)
    )


def main() -> None:
    # -----------------------------------------------------------------
    # 1) Load QC table
    # -----------------------------------------------------------------

    in_path = Path.cwd() / "outputs" / "qc_table.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            f"QC table not found: {in_path}\n"
            "Run scripts/build_qc_table.py first."
        )

    qc_df = pd.read_csv(in_path)

    print("Loaded QC table:", in_path)
    print("Rows:", len(qc_df))
    print("Unique subjects:", qc_df["sub"].nunique())

    # -----------------------------------------------------------------
    # 2) Determine pipelines
    # -----------------------------------------------------------------

    pipelines = sorted(
        qc_df["pipeline"]
        .dropna()
        .unique()
        .tolist()
    )

    print("Pipelines in QC table:", pipelines)

    if not pipelines:
        raise RuntimeError("No pipelines found in QC table.")

    # -----------------------------------------------------------------
    # 3) Select feature
    # -----------------------------------------------------------------

    # Change this to "seedfc" when running the SeedFC analysis.
    feature = "falff"

    print("\nBuilding feature table for:", feature)

    # SeedFC and fALFF now live in the same HALFpipe derivative roots,
    # therefore no feature-specific base directory is needed anymore.
    feat_df = build_feature_rows(
        base_dirs=HALFPIPE_DIRS,
        feature=feature,
        pipelines=pipelines,
        run="run-01",
    )

    print("\nFeature table created.")
    print("Rows:", len(feat_df))
    print("Unique subjects:", feat_df["sub"].nunique())

    # -----------------------------------------------------------------
    # 4) Merge feature values into QC table
    # -----------------------------------------------------------------

    out_df = qc_df.merge(
        feat_df,
        on=["sub", "pipeline"],
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # 5) Sanity checks
    # -----------------------------------------------------------------

    dmn_col = f"{feature}_dmn_mean"

    print("\n=== Feature sanity checks ===")

    print(
        "Rows without feature DMN mean:",
        int(out_df[dmn_col].isna().sum()),
    )

    print(
        "Subjects with feature values:",
        out_df.loc[
            out_df[dmn_col].notna(),
            "sub",
        ].nunique(),
    )

    print(
        "Example feature columns:",
        [
            column
            for column in out_df.columns
            if column.startswith(feature)
        ][:10],
    )

    # Show subjects for which no feature map was found
    missing_feature_subs = sorted(
        out_df.loc[
            out_df[dmn_col].isna(),
            "sub",
        ].unique()
    )

    if missing_feature_subs:
        print(
            "\nSubjects with at least one missing feature row:",
            missing_feature_subs,
        )

    # -----------------------------------------------------------------
    # 6) Save
    # -----------------------------------------------------------------

    out_dir = Path.cwd() / "outputs"
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / f"qc_{feature}_table.csv"

    out_df.to_csv(
        out_path,
        index=False,
    )

    print("\nSaved:")
    print(" -", out_path)


if __name__ == "__main__":
    main()
