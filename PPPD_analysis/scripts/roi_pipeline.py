#!/usr/bin/env python
"""
ALL-IN-ONE ROI PIPELINE

Dieses Skript:
1. Benennt ROIs anhand eines manuell geprüften anatomischen Mappings um
2. Prüft den Space der ROIs
3. Resampled alle ROIs in den Target-MNI152NLin2009cAsym 2mm Space
4. Erstellt ROI-zentrierte QC-PNGs
5. Schreibt ein vollständiges Logfile

Output-Struktur:
final_rois/
├── roi_masks/   (.nii)
├── roi_qc/      (.png)
└── roi_pipeline_log.txt
"""

import os
import re
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import plotting

# Configuration
input_dir = "W:/PPPD/Auswertung_Part2/MRI/ROIs/ROI_masks"
target_img_path = "W:/PPPD/Auswertung_Part2/MRI/ROIs/sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
output_dir = "W:/PPPD/Auswertung_Part2/MRI/ROIs/final_rois"

ROI_MASKS_DIR = os.path.join(output_dir, "roi_masks")
ROI_QC_DIR = os.path.join(output_dir, "roi_qc")
LOGFILE = os.path.join(output_dir, "roi_pipeline_log.txt")

ROI_ALPHA = 0.5
THRESHOLD = 0.1
DISPLAY_MODE = "ortho"

# Mapping of old and new file names
CUSTOM_NAME_MAP = {
    "ROI_Cerebellum_4_5_R_sphere_6_21_-37_-22.nii": "Cerebellum_IV_V_R_ROI_2009.nii",
    "ROI_Cerebellum_VIIa_crusI_Hem_L_MNI.nii": "Cerebellum_CrusI_L_ROI_2009.nii",
    "ROI_Cerebellum_VIIa_crusI_Hem_R_MNI.nii": "Cerebellum_CrusI_R_ROI_2009.nii",
    "ROI_Cerebellum_VIIa_crusI_Verm_L_MNI.nii": "Cerebellum_CrusI_Vermis_L_ROI_2009.nii",
    "ROI_Cerebellum_VIIa_crusI_Verm_R_MNI.nii": "Cerebellum_CrusI_Vermis_R_ROI_2009.nii",
    "ROI_Cerebellum_VIIb_sphere_6--37_-60_-51.nii": "Cerebellum_VIIb_ROI_2009.nii",
    "ROI_Vermis_Uvula_L_sphere_6-0_-55_-40.nii": "Vermis_Uvula_L_ROI_2009.nii",
    "ROI_Vermis_VII_sphere_6--3_-67_-28.nii": "Vermis_VII_ROI_2009.nii",

    "ROI_Insula_Id1_L_MNI.nii": "Insula_Id1_L_ROI_2009.nii",
    "ROI_Insula_Id1_R_MNI.nii": "Insula_Id1_R_ROI_2009.nii",
    "ROI_Insula_Ig1_L_MNI.nii": "Insula_Ig1_L_ROI_2009.nii",
    "ROI_Insula_Ig1_R_MNI.nii": "Insula_Ig1_R_ROI_2009.nii",
    "ROI_Insula_Ig2_L_MNI.nii": "Insula_Ig2_L_ROI_2009.nii",
    "ROI_Insula_Ig2_R_MNI.nii": "Insula_Ig2_R_ROI_2009.nii",
    "ROI_Insula_OP3_sphere_6-36_0_10.nii": "Insula_OP3_sphere_ROI_2009.nii",
    "Area-Op3_rh_MNI152.nii": "Insula_OP3_R_anat_ROI_2009.nii",

    "ROI_IPL_PFcm_L_MNI.nii": "IPL_PFcm_L_ROI_2009.nii",
    "ROI_IPL_PFcm_R_MNI.nii": "IPL_PFcm_R_ROI_2009.nii",
    "ROI_IPL_PF_L_MNI.nii": "IPL_PF_L_ROI_2009.nii",
    "ROI_IPL_PF_R_MNI.nii": "IPL_PF_R_ROI_2009.nii",
    "ROI_Operculum_OP1_L_MNI.nii": "Operculum_OP1_L_ROI_2009.nii",
    "ROI_Operculum_OP1_R_MNI.nii": "Operculum_OP1_R_ROI_2009.nii",
    "ROI_Operculum_OP2_L_MNI.nii": "Operculum_OP2_L_ROI_2009.nii",
    "ROI_Operculum_OP2_R_MNI.nii": "Operculum_OP2_R_ROI_2009.nii",
    "ROI_Operculum_OP4_L_MNI.nii": "Operculum_OP4_L_ROI_2009.nii",
    "ROI_Operculum_OP4_R_MNI.nii": "Operculum_OP4_R_ROI_2009.nii",

    "ROI_GyrusPraecentralis_Area4a_left_sphere_6_-12_-30_67.nii": "Precentral_Area4a_L_ROI_2009.nii",
    "ROI_GyrusPraecentralis_Area4p_left_sphere_6_-38_-18_46.nii": "Precentral_Area4p_L_ROI_2009.nii",

    "ROI_Supramarginalis_Left_sphere_6--51_-43_32.nii": "Supramarginal_L_ROI_2009.nii",
    "ROI_Supramarginalis_Left_sphere_6--58_-38_42.nii": "Supramarginal_L_v2_ROI_2009.nii",
    "ROI_Supramarginalis_Right_sphere_6-58_-35_24.nii": "Supramarginal_R_ROI_2009.nii",
    "ROI_Supramarginalis_sphere_6--45_-31_21.nii": "Supramarginal_Midline_ROI_2009.nii",

    "ROI_CSvR_sphere_6_14_-29_47.nii": "CSv_R_ROI_2009.nii",
    "ROI_CSv_sphere_6-10_-16_45.nii": "CSv_ROI_2009.nii",

    "ROI_V1L_MNI_roi.nii": "V1_L_ROI_2009.nii",
    "ROI_V1R_MNI_roi.nii": "V1_R_ROI_2009.nii",
    "ROI_V2L_MNI_roi.nii": "V2_L_ROI_2009.nii",
    "ROI_V2R_MNI_roi.nii": "V2_R_ROI_2009.nii",
    "ROI_V5L_MNI_roi.nii": "V5_L_ROI_2009.nii",
    "ROI_V5R_MNI_roi.nii": "V5_R_ROI_2009.nii",
    "Area-hOc6_rh_MNI152.nii": "V6_R_ROI_2009.nii",
    "Area-hOc6_lh_MNI152.nii": "V6_L_ROI_2009.nii",

    "Precuneus_ROI_2009.nii": "Precuneus_ROI_2009.nii",
}


# Check if resampling is needed
def roi_matches_target(roi_img, target_img, atol_affine=1e-3, atol_zoom=0.05):
    """
    Prüft, ob ROI exakt im selben Space wie das Target ist.
    """
    same_shape = roi_img.shape == target_img.shape

    roi_zooms = roi_img.header.get_zooms()[:3]
    target_zooms = target_img.header.get_zooms()[:3]
    same_zooms = all(abs(a - b) < atol_zoom for a, b in zip(roi_zooms, target_zooms))

    same_affine = np.allclose(
        roi_img.affine,
        target_img.affine,
        atol=atol_affine
    )

    return same_shape and same_zooms and same_affine


# Space classification for documentation
def classify_roi_space(img, target_img):
    """
    Heuristische Klassifikation des Spaces der ROI.
    Dient nur zur Dokumentation und Warnung – nicht als exakte Registrierung.
    """
    shape = img.shape
    zooms = img.header.get_zooms()[:3]
    affine = img.affine

    target_shape = target_img.shape
    target_zooms = target_img.header.get_zooms()[:3]
    target_affine = target_img.affine

    same_shape = shape == target_shape
    same_zooms = all(abs(a - b) < 0.1 for a, b in zip(zooms, target_zooms))
    same_affine = np.allclose(affine, target_affine, atol=1e-3)

    if same_shape and same_zooms and same_affine:
        return "match_target"

    ratio_shape = [s / t for s, t in zip(shape, target_shape)]
    if all(0.45 < r < 0.55 or 1.9 < r < 2.1 for r in ratio_shape):
        return "likely_same_template_diff_res"

    if (80 <= shape[0] <= 220 and 80 <= shape[1] <= 260 and 80 <= shape[2] <= 220):
        return "MNI_like_unsure"

    return "unknown_or_native"


# Helper
def get_roi_center_mm(roi_img):
    data = roi_img.get_fdata()
    coords = np.array(np.where(data > THRESHOLD))
    if coords.size == 0:
        return (0, 0, 0)
    center_voxel = coords.mean(axis=1)
    return tuple(nib.affines.apply_affine(roi_img.affine, center_voxel))


# Main
def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ROI_MASKS_DIR, exist_ok=True)
    os.makedirs(ROI_QC_DIR, exist_ok=True)

    target_img = nib.load(target_img_path)
    roi_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    with open(LOGFILE, "w", encoding="utf-8") as log:
        log.write("ROI PIPELINE LOG\n")
        log.write(f"Input: {input_dir}\n")
        log.write(f"Target: {target_img_path}\n")
        log.write(f"Output: {output_dir}\n")
        log.write("="*60 + "\n")

        for fname in roi_files:
            old_path = os.path.join(input_dir, fname)

            base_name = CUSTOM_NAME_MAP.get(fname, fname)

            new_name = base_name
            version = 2
            while os.path.exists(os.path.join(ROI_MASKS_DIR, new_name)):
                core = base_name.replace("_ROI_2009.nii", "")
                new_name = f"{core}_v{version}_ROI_2009.nii"
                version += 1

            final_path = os.path.join(ROI_MASKS_DIR, new_name)
            qc_path = os.path.join(ROI_QC_DIR, new_name.replace('.nii', '_QC.png'))

            try:
                roi_img = nib.load(old_path)
                space_class = classify_roi_space(roi_img, target_img)
            except Exception as e:
                log.write(f"[LOAD_ERROR] {fname}: {e}\n")
                continue

            if roi_matches_target(roi_img, target_img):
                # ROI ist bereits korrekt → unverändert übernehmen
                final_img = roi_img
                nib.save(final_img, final_path)
                resample_flag = "not_resampled"
            else:
                # ROI ist im falschen Space → resamplen
                final_img = resample_to_img(
                    roi_img,
                    target_img,
                    interpolation="nearest"
                )
                nib.save(final_img, final_path)
                resample_flag = "resampled"

            cut_coords = get_roi_center_mm(final_img)
            display = plotting.plot_roi(
                final_img,
                bg_img=target_img,
                alpha=ROI_ALPHA,
                threshold=THRESHOLD,
                display_mode=DISPLAY_MODE,
                title=new_name,
                cut_coords=cut_coords
            )
            display.savefig(qc_path, dpi=200)
            display.close()

            log.write(
                f"{fname} -> {new_name} | Space: {space_class} | {resample_flag}\n"
            )

            if space_class in ["MNI_like_unsure", "unknown_or_native"]:
                log.write(f"WARNING: Space unsure! Visual check recommended\n")


# ENTRY POINT

if __name__ == "__main__":
    main()