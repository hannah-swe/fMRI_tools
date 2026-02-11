from __future__ import annotations
from pathlib import Path
import pandas as pd
from pppd.config.paths import ( base_dir_halfpipe_qc, participants_path, reportvals_path, subs_to_drop)
from pppd.config.pipelines import pipeline_patterns
from pppd.io.halfpipe import ( get_tsnr_map, get_confounds_tsv, get_brain_mask)
from pppd.metrics.qc import extract_qc_metrics


def load_participants(participants_tsv: Path) -> pd.DataFrame:
    """Load participants.tsv and normalize participant_id to 'sub-XXX'."""
    df = pd.read_csv(participants_tsv, sep="\t")
    df["participant_id"] = "sub-" + df["participant_id"].astype(str)
    return df


def load_reportvals_qc(reportvals_txt: Path) -> pd.DataFrame:
    """Load HALFpipe reportvals.txt and return FD-related columns with 'sub-XX' style IDs."""
    reportvals_df = pd.read_fwf(reportvals_txt)
    reportvals_df.columns = [c.strip() for c in reportvals_df.columns]

    qc = reportvals_df[["sub", "fd_mean", "fd_max", "fd_perc"]].copy()
    qc["sub"] = "sub-" + qc["sub"].astype(str).str.zfill(2)

    qc = qc.rename(
        columns={
            "fd_mean": "fd_mean_halfpipe",
            "fd_max": "fd_max_halfpipe",
            "fd_perc": "fd_perc_halfpipe",
        }
    )

    num_cols = ["fd_mean_halfpipe", "fd_max_halfpipe", "fd_perc_halfpipe"]
    qc[num_cols] = qc[num_cols].apply(pd.to_numeric, errors="coerce")
    return qc


def build_qc_table(
    base_dir: Path,
    pipeline_patterns: dict[str, str],
) -> pd.DataFrame:
    """Create subject x pipeline QC table from HALFpipe outputs."""
    qc_rows: list[dict] = []

    sub_dirs = sorted(base_dir.glob("sub-*"))
    if not sub_dirs:
        raise FileNotFoundError(f"No sub-* directories found in: {base_dir}")

    for sub_dir in sub_dirs:
        sub = sub_dir.name

        # One tSNR map per run (independent of scrubbing)
        tsnr_map = get_tsnr_map(sub_dir, sub=sub, run="run-01")
        if not tsnr_map.exists():
            print(f"[SKIP] No tSNR map for {sub}: {tsnr_map}")
            continue

        for pipeline in pipeline_patterns.keys():
            confounds_tsv = get_confounds_tsv(
                sub_dir=sub_dir,
                sub=sub,
                run="run-01",
                pipeline=pipeline,
                pipeline_patterns=pipeline_patterns,
            )
            if not confounds_tsv.exists():
                print(f"[SKIP] No confounds for {sub} ({pipeline}): {confounds_tsv}")
                continue

            brain_mask = get_brain_mask(
                sub_dir=sub_dir,
                sub=sub,
                run="run-01",
                pipeline=pipeline,
                pipeline_patterns=pipeline_patterns,
            )
            brain_mask = brain_mask if brain_mask.exists() else None

            qc_rows.append(
                extract_qc_metrics(
                    sub=sub,
                    run="run-01",
                    pipeline=pipeline,
                    confounds_tsv=confounds_tsv,
                    tsnr_map=tsnr_map,
                    brain_mask=brain_mask,
                )
            )

    qc_df = pd.DataFrame(qc_rows)
    if qc_df.empty:
        raise RuntimeError("QC table is empty. Check paths/patterns or missing files.")

    qc_df = qc_df.sort_values(["sub", "pipeline"]).reset_index(drop=True)
    return qc_df


def main() -> None:
    # 1) Build QC from HALFpipe confounds + tSNR
    qc_df = build_qc_table(base_dir_halfpipe_qc, pipeline_patterns)

    # 2) Merge group labels
    participants_df = load_participants(participants_path)
    if "group" not in participants_df.columns:
        raise KeyError("participants.tsv must contain a 'group' column.")

    qc_df = qc_df.merge(
        participants_df[["participant_id", "group"]],
        left_on="sub",
        right_on="participant_id",
        how="left",
    ).drop(columns=["participant_id"])

    # 3) Merge HALFpipe dashboard QC
    reportvals_qc = load_reportvals_qc(reportvals_path)
    qc_df = qc_df.merge(reportvals_qc, on="sub", how="left")

    # 4) Drop subjects
    qc_df = qc_df[~qc_df["sub"].isin(subs_to_drop)].reset_index(drop=True)

    # 5) Sanity checks (prints)
    print("\n=== Sanity checks ===")
    print("Rows total:", len(qc_df))
    print("Unique subjects:", qc_df["sub"].nunique())
    print("Rows without group label:", int(qc_df["group"].isna().sum()))
    print("Rows without fd_mean_halfpipe:", int(qc_df["fd_mean_halfpipe"].isna().sum()))
    print("Pipelines:", qc_df["pipeline"].unique().tolist())

    # 6) Save
    out_dir = Path.cwd() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qc_table.csv"
    qc_df.to_csv(csv_path, index=False)
    print("\nSaved:")
    print(" -", csv_path)

if __name__ == "__main__":
    main()
