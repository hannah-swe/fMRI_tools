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


def weighted_mean_in_probseg(img_path: Path, probseg_path: Path, threshold: float | None = None) -> float:
    """
    Weighted mean of img within a probabilistic tissue map (0..1).
    If threshold is set, weights below threshold are set to 0 (i.e. hard mask).
    """
    img = nib.load(str(img_path)).get_fdata()
    w = nib.load(str(probseg_path)).get_fdata()

    if threshold is not None:
        w = (w > threshold).astype(float)

    denom = np.sum(w)
    if denom <= 0:
        return float("nan")

    return float(np.nansum(img * w) / denom)