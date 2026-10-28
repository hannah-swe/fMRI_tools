from __future__ import annotations

from pathlib import Path

import pandas as pd

from pppd.config.paths import (
    HALFPIPE_DIRS,
    PARTICIPANTS_PATHS,
    REPORTVALS_PATHS,
    subs_to_drop,
    get_fmriprep_dir,
)
from pppd.config.pipelines import pipeline_patterns


# =========================================================
# CONFIG
# =========================================================

FD_THRESHOLD = 0.5
DVARS_THRESHOLD = 1.5

RUN = "run-01"


# =========================================================
# HELPERS
# =========================================================

def normalize_subject_id(value) -> str:
    """
    Normalize subject IDs to the format used by HALFpipe:

    004 -> sub-004
    17 -> sub-017
    sub-101 -> sub-101
    """
    value = str(value).strip()

    if value.startswith("sub-"):
        value = value[4:]

    # pandas may occasionally read IDs as floats, e.g. "4.0"
    if value.endswith(".0"):
        value = value[:-2]

    return f"sub-{int(value):03d}"


def find_confounds_timeseries(
    sub_dir: Path,
    sub: str,
    run: str = RUN,
) -> Path | None:
    """
    Find the fMRIPrep confounds timeseries for one resting-state run.
    """
    func_dir = sub_dir / "func"

    hits = sorted(
        func_dir.glob(
            f"{sub}_task-rest_{run}_desc-confounds_timeseries.tsv"
        )
    )

    return hits[0] if hits else None


def compute_motion_outlier_metrics(
    confounds_path: Path,
    fd_threshold: float = FD_THRESHOLD,
    dvars_threshold: float = DVARS_THRESHOLD,
) -> dict[str, float | int]:
    """
    Compute motion-outlier metrics from a fMRIPrep confounds TSV.

    HALFpipe motion scrubbing uses motion_outlierXX spike regressors.
    These regressors correspond to volumes flagged by FD and/or
    standardized DVARS.

    Returns subject/run-level QC metrics.
    """
    confounds = pd.read_csv(
        confounds_path,
        sep="\t",
    )

    # -----------------------------------------------------
    # Actual motion_outlier spike regressors
    # -----------------------------------------------------
    outlier_cols = [
        col
        for col in confounds.columns
        if col.startswith("motion_outlier")
    ]

    if outlier_cols:
        outlier_mask = (
            confounds[outlier_cols]
            .fillna(0)
            .astype(bool)
            .any(axis=1)
        )
    else:
        outlier_mask = pd.Series(
            False,
            index=confounds.index,
        )

    # -----------------------------------------------------
    # Reconstruct FD / DVARS threshold masks
    # -----------------------------------------------------
    if "framewise_displacement" in confounds.columns:
        fd = pd.to_numeric(
            confounds["framewise_displacement"],
            errors="coerce",
        )

        fd_mask = (
            fd > fd_threshold
        ).fillna(False)

    else:
        fd_mask = pd.Series(
            False,
            index=confounds.index,
        )

    if "std_dvars" in confounds.columns:
        std_dvars = pd.to_numeric(
            confounds["std_dvars"],
            errors="coerce",
        )

        dvars_mask = (
            std_dvars > dvars_threshold
        ).fillna(False)

    else:
        dvars_mask = pd.Series(
            False,
            index=confounds.index,
        )

    # -----------------------------------------------------
    # Counts
    # -----------------------------------------------------
    n_volumes = len(confounds)

    n_motion_outliers = int(
        outlier_mask.sum()
    )

    n_fd_outliers = int(
        fd_mask.sum()
    )

    n_dvars_outliers = int(
        dvars_mask.sum()
    )

    n_fd_only_outliers = int(
        (fd_mask & ~dvars_mask).sum()
    )

    n_dvars_only_outliers = int(
        (~fd_mask & dvars_mask).sum()
    )

    n_fd_and_dvars_outliers = int(
        (fd_mask & dvars_mask).sum()
    )

    if n_volumes > 0:
        motion_outlier_perc = (
            100
            * n_motion_outliers
            / n_volumes
        )
    else:
        motion_outlier_perc = float("nan")

    return {
        "n_volumes": n_volumes,
        "n_motion_outliers": n_motion_outliers,
        "motion_outlier_perc": motion_outlier_perc,
        "n_fd_outliers": n_fd_outliers,
        "n_dvars_outliers": n_dvars_outliers,
        "n_fd_only_outliers": n_fd_only_outliers,
        "n_dvars_only_outliers": n_dvars_only_outliers,
        "n_fd_and_dvars_outliers": n_fd_and_dvars_outliers,
    }


def build_motion_outlier_table(
    subjects: list[str],
    run: str = RUN,
) -> pd.DataFrame:
    """
    Build one row per subject containing motion-outlier metrics
    from the appropriate fMRIPrep derivatives directory.
    """
    rows = []

    for sub in sorted(subjects):

        fmriprep_root = get_fmriprep_dir(sub)

        if not fmriprep_root.exists():
            raise FileNotFoundError(
                f"fMRIPrep directory does not exist: {fmriprep_root}"
            )

        sub_dir = fmriprep_root / sub

        if not sub_dir.exists():
            print(
                f"[WARNING] Subject directory not found for {sub}: "
                f"{sub_dir}"
            )

            rows.append(
                {
                    "sub": sub,
                    "n_volumes": pd.NA,
                    "n_motion_outliers": pd.NA,
                    "motion_outlier_perc": pd.NA,
                    "n_fd_outliers": pd.NA,
                    "n_dvars_outliers": pd.NA,
                    "n_fd_only_outliers": pd.NA,
                    "n_dvars_only_outliers": pd.NA,
                    "n_fd_and_dvars_outliers": pd.NA,
                }
            )

            continue

        confounds_path = find_confounds_timeseries(
            sub_dir=sub_dir,
            sub=sub,
            run=run,
        )

        if confounds_path is None:
            print(
                f"[WARNING] No confounds file found for {sub} "
                f"in {sub_dir / 'func'}"
            )

            rows.append(
                {
                    "sub": sub,
                    "n_volumes": pd.NA,
                    "n_motion_outliers": pd.NA,
                    "motion_outlier_perc": pd.NA,
                    "n_fd_outliers": pd.NA,
                    "n_dvars_outliers": pd.NA,
                    "n_fd_only_outliers": pd.NA,
                    "n_dvars_only_outliers": pd.NA,
                    "n_fd_and_dvars_outliers": pd.NA,
                }
            )

            continue

        metrics = compute_motion_outlier_metrics(
            confounds_path=confounds_path,
        )

        rows.append(
            {
                "sub": sub,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def build_subject_pipeline_table(
    base_dirs: tuple[Path, ...],
    pipelines: list[str],
) -> pd.DataFrame:
    """
    Find all HALFpipe subjects and create one row per
    subject x pipeline.

    No confounds or preprocessed images are required.
    """
    subjects: list[str] = []

    for base_dir in base_dirs:
        print(
            f"Searching HALFpipe directory: {base_dir}"
        )

        sub_dirs = sorted(
            base_dir.glob("sub-*")
        )

        print(
            f"Found {len(sub_dirs)} subject directories."
        )

        subjects.extend(
            sub_dir.name
            for sub_dir in sub_dirs
        )

    # Safety check:
    # subjects should occur in only one HALFpipe root
    subjects_series = pd.Series(subjects)

    duplicates = subjects_series[
        subjects_series.duplicated()
    ].unique()

    if len(duplicates) > 0:
        raise RuntimeError(
            "Subjects found in more than one "
            "HALFpipe directory: "
            f"{duplicates.tolist()}"
        )

    rows = [
        {
            "sub": sub,
            "pipeline": pipeline,
        }
        for sub in sorted(subjects)
        for pipeline in pipelines
    ]

    return pd.DataFrame(rows)


def load_participants(
    paths: tuple[Path, ...],
) -> pd.DataFrame:
    """
    Combine participants.tsv files and normalize
    participant IDs.
    """
    dfs = []

    for path in paths:
        print(
            f"Loading participants: {path}"
        )

        df = pd.read_csv(
            path,
            sep="\t",
            dtype={
                "participant_id": str,
            },
        )

        if "participant_id" not in df.columns:
            raise KeyError(
                f"'participant_id' column missing in {path}"
            )

        if "group" not in df.columns:
            raise KeyError(
                f"'group' column missing in {path}"
            )

        df["participant_id"] = (
            df["participant_id"]
            .apply(normalize_subject_id)
        )

        dfs.append(
            df[
                [
                    "participant_id",
                    "group",
                ]
            ].copy()
        )

    participants = pd.concat(
        dfs,
        ignore_index=True,
    )

    # Protect against duplicate entries
    # across the two TSVs
    duplicates = participants[
        participants["participant_id"]
        .duplicated(keep=False)
    ]

    if not duplicates.empty:
        print(
            "\n[WARNING] Duplicate participants found:"
        )

        print(
            duplicates
            .sort_values("participant_id")
            .to_string(index=False)
        )

        participants = (
            participants
            .drop_duplicates(
                subset="participant_id",
                keep="last",
            )
        )

    return participants


def load_reportvals(
    paths: tuple[Path, ...],
) -> pd.DataFrame:
    """
    Combine HALFpipe reportvals.txt files.

    These FD columns are retained for QC analyses.
    """
    dfs = []

    for path in paths:
        print(
            f"Loading reportvals: {path}"
        )

        df = pd.read_fwf(path)

        df.columns = [
            c.strip()
            for c in df.columns
        ]

        required = [
            "sub",
            "fd_mean",
            "fd_max",
            "fd_perc",
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:
            raise KeyError(
                f"Missing columns in {path}: {missing}"
            )

        df = df[
            required
        ].copy()

        # Remove separator/header rows
        # from fixed-width report
        df = df[
            df["sub"]
            .astype(str)
            .str.fullmatch(r"\d+")
        ].copy()

        df["sub"] = (
            df["sub"]
            .apply(normalize_subject_id)
        )

        df = df.rename(
            columns={
                "fd_mean":
                    "fd_mean_halfpipe",

                "fd_max":
                    "fd_max_halfpipe",

                "fd_perc":
                    "fd_perc_halfpipe",
            }
        )

        numeric_cols = [
            "fd_mean_halfpipe",
            "fd_max_halfpipe",
            "fd_perc_halfpipe",
        ]

        df[numeric_cols] = (
            df[numeric_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        dfs.append(df)

    reportvals = pd.concat(
        dfs,
        ignore_index=True,
    )

    duplicates = reportvals[
        reportvals["sub"]
        .duplicated(keep=False)
    ]

    if not duplicates.empty:
        print(
            "\n[WARNING] Duplicate subjects "
            "in reportvals:"
        )

        print(
            duplicates
            .sort_values("sub")
            .to_string(index=False)
        )

        reportvals = (
            reportvals
            .drop_duplicates(
                subset="sub",
                keep="last",
            )
        )

    return reportvals


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # 1. Build subject x pipeline table
    # -----------------------------------------------------
    pipelines = list(
        pipeline_patterns.keys()
    )

    qc_df = build_subject_pipeline_table(
        base_dirs=HALFPIPE_DIRS,
        pipelines=pipelines,
    )

    print(
        "\n=== Initial subject table ==="
    )

    print(
        "Rows:",
        len(qc_df),
    )

    print(
        "Unique subjects:",
        qc_df["sub"].nunique(),
    )

    print(
        "Pipelines:",
        qc_df[
            "pipeline"
        ].value_counts().to_dict(),
    )


    # -----------------------------------------------------
    # 2. Add group information
    # -----------------------------------------------------
    participants_df = load_participants(
        PARTICIPANTS_PATHS
    )

    qc_df = qc_df.merge(
        participants_df,
        left_on="sub",
        right_on="participant_id",
        how="left",
        validate="many_to_one",
    )

    qc_df = qc_df.drop(
        columns=[
            "participant_id",
        ]
    )


    # -----------------------------------------------------
    # 3. Add HALFpipe FD report values
    # -----------------------------------------------------
    reportvals_df = load_reportvals(
        REPORTVALS_PATHS
    )

    qc_df = qc_df.merge(
        reportvals_df,
        on="sub",
        how="left",
        validate="many_to_one",
    )


    # -----------------------------------------------------
    # 4. Add actual motion-outlier / scrubbing metrics
    # -----------------------------------------------------
    motion_outlier_df = build_motion_outlier_table(
        subjects=qc_df["sub"].unique().tolist(),
        run=RUN,
    )

    qc_df = qc_df.merge(
        motion_outlier_df,
        on="sub",
        how="left",
        validate="many_to_one",
    )


    # -----------------------------------------------------
    # 5. Drop subjects explicitly excluded
    #    from analysis
    # -----------------------------------------------------
    qc_df = qc_df[
        ~qc_df["sub"].isin(
            subs_to_drop
        )
    ].copy()

    qc_df = (
        qc_df
        .sort_values(
            [
                "sub",
                "pipeline",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # 6. Sanity checks
    # -----------------------------------------------------
    print(
        "\n=== Final QC table ==="
    )

    print(
        "Rows:",
        len(qc_df),
    )

    print(
        "Unique subjects:",
        qc_df["sub"].nunique(),
    )

    print(
        "\nRows per pipeline:"
    )

    print(
        qc_df[
            "pipeline"
        ].value_counts()
    )

    print(
        "\nFirst subjects:"
    )

    print(
        sorted(
            qc_df[
                "sub"
            ].unique()
        )[:20]
    )

    print(
        "\nMissing groups:"
    )

    print(
        qc_df.loc[
            qc_df[
                "group"
            ].isna(),
            "sub",
        ].unique()
    )

    print(
        "\nMissing reportvals:"
    )

    print(
        qc_df.loc[
            qc_df[
                "fd_mean_halfpipe"
            ].isna(),
            "sub",
        ].unique()
    )

    print(
        "\nMissing motion-outlier metrics:"
    )

    print(
        qc_df.loc[
            qc_df[
                "n_motion_outliers"
            ].isna(),
            "sub",
        ].unique()
    )


    # -----------------------------------------------------
    # 7. Scrubbing summary
    # -----------------------------------------------------
    subject_motion_df = (
        qc_df[
            [
                "sub",
                "n_volumes",
                "n_motion_outliers",
                "motion_outlier_perc",
                "n_fd_outliers",
                "n_dvars_outliers",
                "n_fd_only_outliers",
                "n_dvars_only_outliers",
                "n_fd_and_dvars_outliers",
            ]
        ]
        .drop_duplicates(
            subset="sub"
        )
    )

    print(
        "\n=== Motion-outlier summary ==="
    )

    print(
        subject_motion_df[
            [
                "n_motion_outliers",
                "motion_outlier_perc",
                "n_fd_outliers",
                "n_dvars_outliers",
                "n_fd_only_outliers",
                "n_dvars_only_outliers",
                "n_fd_and_dvars_outliers",
            ]
        ].describe()
    )


    # -----------------------------------------------------
    # 8. Compare HALFpipe fd_perc with actual
    #    motion-outlier percentage
    # -----------------------------------------------------
    comparison_df = (
        qc_df[
            [
                "sub",
                "fd_perc_halfpipe",
                "motion_outlier_perc",
            ]
        ]
        .drop_duplicates(
            subset="sub"
        )
        .dropna()
    )

    if not comparison_df.empty:
        print(
            "\n=== fd_perc vs actual motion-outlier % ==="
        )

        print(
            comparison_df[
                [
                    "fd_perc_halfpipe",
                    "motion_outlier_perc",
                ]
            ].corr()
        )


    # -----------------------------------------------------
    # 9. Save
    # -----------------------------------------------------
    out_dir = (
        Path.cwd()
        / "outputs"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / "qc_table.csv"
    )

    qc_df.to_csv(
        out_path,
        index=False,
    )

    print(
        "\nSaved:",
        out_path,
    )


if __name__ == "__main__":
    main()