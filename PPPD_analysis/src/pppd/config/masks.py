from pathlib import Path

dmn_mask = Path("/data_wgs04/ag-sensomotorik/PPPD/masks/dmn_mask_resampled.nii.gz")

roi_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/masks/harvard_oxford_rois_tf")
roi_masks = {
    "roi_precuneous": roi_dir / "roi_precuneous.nii.gz",
    "roi_angular_L":  roi_dir / "roi_angular_L.nii.gz",
    "roi_angular_R":  roi_dir / "roi_angular_R.nii.gz",
    "roi_mpfc":       roi_dir / "roi_mpfc.nii.gz",
}
