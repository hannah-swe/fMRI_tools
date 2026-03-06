import os
from PPPD.subjects import subs

SUPPORTED_FEATURES = ["seed_based", "falff"]

# creates list of full subject ids
subject_ids = []
for s in subs:
    subject_id = f"sub-{s:03d}"
    subject_ids.append(subject_id)

# Path to folder with all HALFpipe working directories
analysis_path = '/data_wgs04/ag-sensomotorik/PPPD/analysis/'

# Gets the path for first seed-based analysis folder
def _get_data_path(feature):
    if feature not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if feature == "seed_based":
        return os.path.join(analysis_path, "both_parts_seed1")
    if feature == "falff":
        return os.path.join(analysis_path, "both_parts_falff") # TODO: what going on here?


# TODO: def _get_derivates_path(feature):
# TODO: add function to participants.tsv

