from __future__ import annotations
from pathlib import Path


def seedfc_map_path(sub_dir: Path, sub: str, pipeline: str, run: str = "run-01") -> Path:
    """
    SeedFC statmap paths (current Naming).
    pipeline: "no_scrub" | "scrub"
    """
    func_task = sub_dir / "func" / "task-rest"
    if pipeline == "no_scrub":
        return func_task / f"{sub}_task-rest_{run}_feature-seedbasedNoscrub_seed-Precuneus_stat-effect_statmap.nii.gz"
    else:
        return func_task / f"{sub}_task-rest_{run}_feature-seedbasedScrub_seed-Precuneus_stat-effect_statmap.nii.gz"


def falff_map_path(sub_dir: Path, sub: str, pipeline: str, run: str = "run-01") -> Path:
    """
    fALFF map paths (current Naming).
    pipeline: "no_scrub" | "scrub"
    """
    func_task = sub_dir / "func" / "task-rest"

    if pipeline == "no_scrub":
        return func_task / f"{sub}_task-rest_{run}_feature-falffNoscrub_falff.nii.gz"
    else:
        return func_task / f"{sub}_task-rest_{run}_feature-falffScrub_falff.nii.gz"
