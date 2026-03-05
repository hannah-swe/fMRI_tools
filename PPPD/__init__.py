import os
from PPPD.subjects import subs

# creates list of full subject ids
subject_ids = []
for s in subs:
    subject_id = f"sub-{s:03d}"
    subject_ids.append(subject_id)

# Path to folder with all HALFpipe working directories
data_path = '/data_wgs04/ag-sensomotorik/PPPD/analysis/'

# Gets the path for first seed-based analysis folder
def _get_seed_based_path():
    seed_based_path = os.path.join(data_path, "both_parts_seed1")
    return seed_based_path

# Gets the path for fALFF analysis folder
def _get_falff_path():
    falff_path = os.path.join(data_path, "both_parts_falff")
    return falff_path

