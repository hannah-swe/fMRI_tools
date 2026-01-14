import glob
import numpy as np
from nilearn import image

# Pfad-Muster anpassen
files = sorted(glob.glob("/data_wgs04/ag-sensomotorik/PPPD/workdir_precuneus/derivatives/halfpipe/sub-*/func/task-rest/sub-*_task-rest_run-01_feature-seedbasedPrec_seed-Precuneus_stat-z_statmap.nii.gz"))
assert len(files) > 0, "Keine Dateien gefunden."

imgs = [image.load_img(f) for f in files]

# Voxelweise Mittelwert über Subjects
mean_img = image.mean_img(imgs)
mean_img.to_filename("/data_wgs04/ag-sensomotorik/PPPD/workdir_precuneus/mean_zmap.nii.gz")