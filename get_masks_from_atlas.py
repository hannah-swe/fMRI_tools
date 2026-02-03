#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import xml.etree.ElementTree as ET
from templateflow import api as tf


# Config
tpl = "MNI152NLin2009cAsym"
atlas = "HOCPAL"
desc = "th25"
res = 2

out_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/masks/harvard_oxford_rois_tf")
out_dir.mkdir(parents=True, exist_ok=True)

xml_path = Path("/home/hannahschewe/nilearn_data/fsl/data/atlases/HarvardOxford-Cortical-Lateralized.xml")
if not xml_path.exists():
    raise FileNotFoundError(f"XML not found: {xml_path}")

# dseg ids are 1..96, xml ids are 0..95
OFFSET = 1  # dseg_id = xml_id + 1


# Load dseg from TemplateFlow
dseg_path = tf.get(tpl, resolution=res, atlas=atlas, desc=desc, suffix="dseg")
atlas_img = nib.load(str(dseg_path))
atlas_data = atlas_img.get_fdata().astype(int)

print("Using dseg:", dseg_path)
print("Atlas shape:", atlas_data.shape)


# Parse XML (xml_id -> name)
root = ET.parse(str(xml_path)).getroot()
label_nodes = root.findall(".//label")

xml_id2name = {}
for lab in label_nodes:
    if "index" not in lab.attrib:
        continue
    idx = int(lab.attrib["index"])
    name = (lab.text or "").strip()
    xml_id2name[idx] = name

# Invert mapping: name -> xml_id (names are unique here)
name2xml = {v: k for k, v in xml_id2name.items()}


# Helper: save mask for one or multiple xml_ids
def save_mask_from_xml_ids(xml_ids, filename):
    dseg_ids = [i + OFFSET for i in xml_ids]
    mask = np.isin(atlas_data, dseg_ids).astype(np.uint8)

    out_path = out_dir / filename
    nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header).to_filename(str(out_path))

    print("Saved:", out_path)
    print("  xml_ids:", xml_ids)
    print("  dseg_ids:", dseg_ids)
    print("  n_vox:", int(mask.sum()))


# Define ROIs by exact atlas names (no substring pitfalls)
# Angular gyrus
ang_L = name2xml["Left Angular Gyrus"]
ang_R = name2xml["Right Angular Gyrus"]

# Precuneus (note spelling: Precuneous)
prec_L = name2xml["Left Precuneous Cortex"]
prec_R = name2xml["Right Precuneous Cortex"]

# mPFC proxy: Frontal Medial Cortex (combine L+R into one mask)
mpfc_L = name2xml["Left Frontal Medial Cortex"]
mpfc_R = name2xml["Right Frontal Medial Cortex"]


# Save masks
save_mask_from_xml_ids([prec_L, prec_R], "roi_precuneous.nii.gz")

save_mask_from_xml_ids([ang_L],  "roi_angular_L.nii.gz")
save_mask_from_xml_ids([ang_R],  "roi_angular_R.nii.gz")

save_mask_from_xml_ids([mpfc_L, mpfc_R], "roi_mpfc.nii.gz")
