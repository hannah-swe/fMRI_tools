from pathlib import Path

# QC reference run (pick one; could be seedfc run)
base_dir_halfpipe_qc = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/halfpipe")

# Path to HALFpipe derivatives (contains feature-specific outputs)
base_dir_halfpipe_seedfc = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/halfpipe")
base_dir_halfpipe_falff  = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/derivatives/halfpipe")

# Path to fMRIPrep derivatives
fmriprep_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/fmriprep")

# Path to participants.tsv in BIDS data folder
participants_path = Path("/data_wgs04/ag-sensomotorik/PPPD/data/part2_pre/participants.tsv")

# Path to reportvals textfile
reportvals_path = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/reports/reportvals.txt")

# CONFIG: which subjects to drop? (those subjects don't have complete data sets atm)
subs_to_drop = ["sub-118", "sub-124", "sub-126", "sub-134", "sub-140", "sub-144", "sub-164"]

# Path to save plots
out_dir = Path("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots")
