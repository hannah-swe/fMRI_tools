from __future__ import annotations
from pathlib import Path
from typing import Mapping


def get_func_dir(sub_dir: Path) -> Path:
    return sub_dir / "func"


def get_tsnr_map(sub_dir: Path, sub: str, run: str = "run-01") -> Path:
    func_dir = get_func_dir(sub_dir)
    return func_dir / f"{sub}_task-rest_{run}_stat-tsnr_boldmap.nii.gz"


def get_confounds_tsv(
    sub_dir: Path,
    sub: str,
    run: str,
    pipeline: str,
    pipeline_patterns: Mapping[str, str],
) -> Path:
    func_dir = get_func_dir(sub_dir)
    pat = pipeline_patterns[pipeline]
    return func_dir / f"{sub}_task-rest_{run}_{pat}_desc-confounds_regressors.tsv"


def get_brain_mask(
    sub_dir: Path,
    sub: str,
    run: str,
    pipeline: str,
    pipeline_patterns: Mapping[str, str],
) -> Path:
    func_dir = get_func_dir(sub_dir)
    pat = pipeline_patterns[pipeline]
    return func_dir / f"{sub}_task-rest_{run}_{pat}_desc-brain_mask.nii.gz"


def seedfc_statmap_path(sub_dir: Path, sub: str, variant: str, run: str = "run-01") -> Path:
    """Paths to seed-based outputs (your current naming)."""
    func_task = sub_dir / "func" / "task-rest"
    if variant == "no_scrub":
        return func_task / f"{sub}_task-rest_{run}_feature-seedPrecNoscrub_seed-Precuneus_stat-effect_statmap.nii.gz"
    else:
        return func_task / f"{sub}_task-rest_{run}_feature-seedPrecScrub_seed-Precuneus_stat-effect_statmap.nii.gz"


def find_confounds_timeseries(sub_dir: Path, sub: str, run: str = "run-01") -> Path | None:
    func_dir = sub_dir / "func"
    hits = sorted(func_dir.glob(f"{sub}_task-rest_{run}_desc-confounds_timeseries.tsv"))
    return hits[0] if hits else None