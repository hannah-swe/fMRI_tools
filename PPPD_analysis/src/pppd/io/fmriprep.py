from __future__ import annotations
from pathlib import Path
import pandas as pd


def find_confounds_timeseries(sub_dir: Path, sub: str, run: str = "run-01") -> Path | None:
    func_dir = sub_dir / "func"
    hits = sorted(func_dir.glob(f"{sub}_task-rest_{run}_desc-confounds_timeseries.tsv"))
    return hits[0] if hits else None


def load_confounds(
    confounds_path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        confounds_path,
        sep="\t",
    )


def gm_probseg_mni(sub_dir: Path, sub: str) -> Path:
    return sub_dir / "anat" / f"{sub}_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz"


def wm_probseg_mni(sub_dir: Path, sub: str) -> Path:
    return sub_dir / "anat" / f"{sub}_space-MNI152NLin2009cAsym_res-2_label-WM_probseg.nii.gz"


def csf_probseg_mni(sub_dir: Path, sub: str) -> Path:
    return sub_dir / "anat" / f"{sub}_space-MNI152NLin2009cAsym_res-2_label-CSF_probseg.nii.gz"

def brain_mask_mni(sub_dir: Path, sub: str) -> Path:
    return sub_dir / "anat" / f"{sub}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
