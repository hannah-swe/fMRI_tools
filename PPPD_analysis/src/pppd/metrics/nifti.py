from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib


def mean_in_mask_nii(img_path: Path, mask_path: Path | None = None) -> float:
    """Mean of NIfTI image; optionally restricted to binary mask (>0)."""
    img = nib.load(str(img_path)).get_fdata()
    if mask_path is None:
        return float(np.nanmean(img))
    mask = nib.load(str(mask_path)).get_fdata() > 0
    return float(np.nanmean(img[mask]))