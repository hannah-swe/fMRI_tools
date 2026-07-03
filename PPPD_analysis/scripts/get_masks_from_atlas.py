#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import xml.etree.ElementTree as ET
from templateflow import api as tf


# Config
tpl = "MNI152NLin2009cAsym"
desc = "th25"
res = 2
OFFSET = 1  # TemplateFlow dseg_id = FSL xml_id + 1

out_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/masks/harvard_oxford_rois_tf")
out_dir.mkdir(parents=True, exist_ok=True)


def load_atlas(atlas_name, xml_path):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    dseg_path = tf.get(
        tpl,
        resolution=res,
        atlas=atlas_name,
        desc=desc,
        suffix="dseg"
    )

    atlas_img = nib.load(str(dseg_path))
    atlas_data = atlas_img.get_fdata().astype(int)

    root = ET.parse(str(xml_path)).getroot()
    xml_id2name = {}

    for lab in root.findall(".//label"):
        if "index" not in lab.attrib:
            continue
        idx = int(lab.attrib["index"])
        name = (lab.text or "").strip()
        xml_id2name[idx] = name

    name2xml = {v: k for k, v in xml_id2name.items()}

    print("\nLoaded:", atlas_name)
    print("  dseg:", dseg_path)
    print("  shape:", atlas_data.shape)
    print("  n labels:", len(name2xml))

    return atlas_img, atlas_data, name2xml


def save_mask(atlas_img, atlas_data, xml_ids, filename):
    dseg_ids = [i + OFFSET for i in xml_ids]
    mask = np.isin(atlas_data, dseg_ids).astype(np.uint8)

    out_path = out_dir / filename
    nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header).to_filename(str(out_path))

    print("Saved:", out_path)
    print("  xml_ids:", xml_ids)
    print("  dseg_ids:", dseg_ids)
    print("  n_vox:", int(mask.sum()))

    if int(mask.sum()) == 0:
        print("  WARNING: mask is empty!")


# -------------------------
# Load cortical atlas
# -------------------------
cort_img, cort_data, cort_name2xml = load_atlas(
    atlas_name="HOCPAL",
    xml_path="/home/hannahschewe/nilearn_data/fsl/data/atlases/HarvardOxford-Cortical-Lateralized.xml"
)

# -------------------------
# Load subcortical atlas
# -------------------------
sub_img, sub_data, sub_name2xml = load_atlas(
    atlas_name="HOSPA",
    xml_path="/home/hannahschewe/nilearn_data/fsl/data/atlases/HarvardOxford-Subcortical.xml"
)


# =========================
# Cortical ROIs
# =========================

# Angular gyrus
ang_L = cort_name2xml["Left Angular Gyrus"]
ang_R = cort_name2xml["Right Angular Gyrus"]

# Precuneus
prec_L = cort_name2xml["Left Precuneous Cortex"]
prec_R = cort_name2xml["Right Precuneous Cortex"]

# mPFC proxy
mpfc_L = cort_name2xml["Left Frontal Medial Cortex"]
mpfc_R = cort_name2xml["Right Frontal Medial Cortex"]

save_mask(cort_img, cort_data, [prec_L, prec_R], "roi_precuneous.nii.gz")
save_mask(cort_img, cort_data, [ang_L], "roi_angular_L.nii.gz")
save_mask(cort_img, cort_data, [ang_R], "roi_angular_R.nii.gz")
save_mask(cort_img, cort_data, [mpfc_L, mpfc_R], "roi_mpfc.nii.gz")


# =========================
# Subcortical ROIs
# =========================

hipp_L = sub_name2xml["Left Hippocampus"]
hipp_R = sub_name2xml["Right Hippocampus"]

save_mask(sub_img, sub_data, [hipp_L], "roi_hippocampus_L.nii.gz")
save_mask(sub_img, sub_data, [hipp_R], "roi_hippocampus_R.nii.gz")
