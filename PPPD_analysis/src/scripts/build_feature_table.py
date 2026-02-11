from __future__ import annotations
from pathlib import Path
import pandas as pd
from pppd.config.paths import base_dir_halfpipe_seedfc, base_dir_halfpipe_falff
from pppd.config.masks import dmn_mask, roi_masks
from pppd.metrics.nifti import mean_in_mask_nii
from pppd.metrics.roi import extract_roi_means
from pppd.io.features import seedfc_map_path, falff_map_path


def build_feature_rows(
    base_dir: Path,
    feature: str,
    pipelines: list[str],
    run: str = "run-01",
) -> pd.DataFrame:
    """
    Create a per-subject, per-pipeline feature table:
    columns: sub, pipeline, <feature>_dmn_mean, + ROI means
    """
    if feature not in {"seedfc", "falff"}:
        raise ValueError("feature must be one of: {'seedfc', 'falff'}")

    map_path_fn = seedfc_map_path if feature == "seedfc" else falff_map_path
    dmn_col = f"{feature}_dmn_mean"

    rows: list[dict] = []

    for sub_dir in sorted(base_dir.glob("sub-*")):
        sub = sub_dir.name

        for pipeline in pipelines:
            img_path = map_path_fn(sub_dir=sub_dir, sub=sub, pipeline=pipeline, run=run)
            if not img_path.exists():
                print(f"[SKIP] Missing {feature} map for {sub} ({pipeline}): {img_path}")
                continue

            row = {
                "sub": sub,
                "pipeline": pipeline,
                dmn_col: mean_in_mask_nii(img_path, dmn_mask),
            }

            # ROI means (re-using your Harvard-Oxford masks)
            roi_means = extract_roi_means(img_path, roi_masks)

            # prefix ROI columns so they never collide across features
            roi_means = {f"{feature}_{k}": v for k, v in roi_means.items()}
            row.update(roi_means)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"Feature table is empty for feature='{feature}'. "
            f"Most likely the map filename patterns in io/features.py are not matching."
        )

    return df.sort_values(["sub", "pipeline"]).reset_index(drop=True)


def main() -> None:
    # 1) Load QC table
    in_path = Path.cwd() / "outputs" / "qc_table.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"QC table not found: {in_path}\n"
            f"Run scripts/build_qc_table.py first."
        )

    qc_df = pd.read_csv(in_path)

    # Ensure consistent pipeline list
    pipelines = sorted(qc_df["pipeline"].unique().tolist())
    print("Pipelines in QC table:", pipelines)

    # 2) Build feature table
    feature = "falff"

    if feature == "seedfc":
        feature_base_dir = base_dir_halfpipe_seedfc
    else:
        feature_base_dir = base_dir_halfpipe_falff

    feat_df = build_feature_rows(
        base_dir=feature_base_dir,
        feature=feature,
        pipelines=pipelines,
        run="run-01",
    )

    # 3) Merge into QC table
    out_df = qc_df.merge(feat_df, on=["sub", "pipeline"], how="left")

    # 4) Sanity checks
    dmn_col = f"{feature}_dmn_mean"
    print("\n=== Feature sanity checks ===")
    print("Rows without feature DMN mean:", int(out_df[dmn_col].isna().sum()))
    print("Example columns:", [c for c in out_df.columns if c.startswith(feature)][:10])

    # 5) Save
    out_dir = Path.cwd() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"qc_{feature}_table.csv"
    out_df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
