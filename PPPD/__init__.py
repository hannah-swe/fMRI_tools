import os
import pandas as pd
from PPPD.subjects import subs

SUPPORTED_FEATURES = ["seed_based", "falff"]

# creates list of full subject ids
subject_ids = []
for s in subs:
    subject_id = f"sub-{s:03d}"
    subject_ids.append(subject_id)

# Path to folder with all HALFpipe working directories
analysis_path = '/data_wgs04/ag-sensomotorik/PPPD/analysis/'
raw_data_path = '/data_wgs04/ag-sensomotorik/PPPD/data/all_subjects1/'


# Gets the path for analysis folder based on the selected feature
def _get_data_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if feature == "seed_based":
        return os.path.join(analysis_path, "both_parts_seed1")
    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff") # TODO: what going on here?
    return None


# Gets the part for the derivatives folder based on the selected feature
def _get_derivatives_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if feature == "seed_based":
        return os.path.join(analysis_path, "both_parts_seed1", "derivatives")
    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff", "derivatives")
    return None


# Gets participants.tsv from data folder
def _get_participants_tsv():
    tsv_path = os.path.join(raw_data_path, "participants.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"No participants.tsv found in {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df

participants = _get_participants_tsv()