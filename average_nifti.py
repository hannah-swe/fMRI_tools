import glob
import numpy as np
from nilearn import image
import nibabel as nib
import matplotlib.pyplot as plt

# Pfad-Muster anpassen
files = sorted(glob.glob(
    "/data_wgs04/ag-sensomotorik/PPPD/analysis/part2_pre/derivatives/halfpipe/sub-*/func/task-rest/sub-*_task-rest_run-01_feature-fALFFScrub_alff.nii.gz"
))
assert len(files) > 0, "Keine Dateien gefunden."

imgs = [image.load_img(f) for f in files]

# Voxelweise Mittelwert über Subjects
mean_img = image.mean_img(imgs)

data = mean_img.get_fdata()
print(np.nanmin(data), np.nanmax(data), np.nanmean(data))

mean_img.to_filename("/data_wgs04/ag-sensomotorik/PPPD/mean_alff_scrubbed.nii.gz")


x = data[np.isfinite(data)]

q01, q99 = np.percentile(x, [1, 99])
q02, q98 = np.percentile(x, [2, 98])
q05, q95 = np.percentile(x, [5, 95])

print("1–99%:", q01, q99)
print("2–98%:", q02, q98)
print("5–95%:", q05, q95)

plt.figure(figsize=(6,4))
plt.hist(x, bins=50, density=True)
plt.xlabel("Wert")
plt.ylabel("Dichte")
plt.title("Voxelwert-Verteilung")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

kde = gaussian_kde(x)

xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 500)
ys = kde(xs)

plt.figure(figsize=(6,4))
plt.plot(xs, ys)
plt.xlabel("z-value")
plt.ylabel("Dichte")
plt.title("Density Curve (KDE)")
plt.tight_layout()
plt.show()