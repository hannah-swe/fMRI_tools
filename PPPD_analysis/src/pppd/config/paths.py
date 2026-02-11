from pathlib import Path

# QC reference run (pick one; could be seedfc run)
base_dir_halfpipe_qc = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/halfpipe")

# Feature-specific runs
base_dir_halfpipe_seedfc = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/halfpipe")
base_dir_halfpipe_falff  = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/derivatives/halfpipe")

fmriprep_dir = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre_old/derivatives/fmriprep")

participants_path = Path("/data_wgs04/ag-sensomotorik/PPPD/data/part2_pre/participants.tsv")
reportvals_path = Path("/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/reports/reportvals.txt")

subs_to_drop = ["sub-118", "sub-124", "sub-126", "sub-134", "sub-140", "sub-144", "sub-164"]

# optional outputs
out_dir = Path("/home/hannahschewe/Documents/PPPD_analysis/quality_check_plots")
