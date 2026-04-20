import numpy as np
from pathlib import Path
from nilearn.image import load_img, get_data, new_img_like

# Ordner mit deinen ROI-NIfTI-Dateien
roi_dir = Path("W:/PPPD/Auswertung_Part2/MRI/ROIs_Koordinaten/final_rois/roi_masks/")

roi_files = sorted(roi_dir.glob("*.nii*"))

roi_imgs = [load_img(f) for f in roi_files]
ref_img = roi_imgs[0]

mask_data = np.zeros(ref_img.shape, dtype=np.uint8)

for img in roi_imgs:
    data = get_data(img)
    mask_data[data > 0] = 1   # alles >0 wird Maskenvoxel

mask_img = new_img_like(ref_img, mask_data)
mask_img.to_filename("W:/PPPD/Auswertung_Part2/MRI/ROIs_Koordinaten/final_rois/vvn_mask_resampled.nii.gz")