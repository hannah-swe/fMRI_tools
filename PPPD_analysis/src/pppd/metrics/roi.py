from __future__ import annotations
from pathlib import Path
from typing import Mapping
from pppd.metrics.nifti import mean_in_mask_nii


def extract_roi_means(img_path: Path, roi_masks: Mapping[str, Path]) -> dict:
    """Return dict of mean values in each ROI mask."""
    out = {}
    for col, mask_path in roi_masks.items():
        out[col] = mean_in_mask_nii(img_path, mask_path)
    return out
