import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

outdir = Path("/home/hannahschewe/")

# --- Get split RdBu cmap
vmin = 0.7
vmax = 2.1

# obere Skala: Blau aus RdBu_r > 0
blue_cmap = mpl.cm.get_cmap("RdBu_r")
blue_half = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_half",
    blue_cmap(np.linspace(0.5, 1.0, 256))
)

# untere Skala: Rot aus RdBu > 0
red_cmap = mpl.cm.get_cmap("RdBu")
red_half = mpl.colors.LinearSegmentedColormap.from_list(
    "red_half",
    red_cmap(np.linspace(0.5, 1.0, 256))
)

fig, axes = plt.subplots(
    nrows=2,
    figsize=(3, 1.2),
    constrained_layout=True
)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
ticks = [0.8, 1.2, 1.6, 2.0]

for ax, cmap in zip(axes, [blue_half, red_half]):
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
        ticks=ticks
    )
    cb.ax.tick_params(labelsize=10, length=4, width=0.8)
    cb.outline.set_linewidth(0.7)

# als Vektorgrafik speichern
fig.savefig(outdir / "RdBu_split_colorbars.svg", bbox_inches="tight")
fig.savefig(outdir / "RdBu_split_colorbars.pdf", bbox_inches="tight")

plt.show()


# --- Get full cmap
vmin_full = -1
vmax_full = 1

fig, ax = plt.subplots(figsize=(6, 0.6), constrained_layout=True)

norm = mpl.colors.Normalize(vmin=vmin_full, vmax=vmax_full)
ticks = np.linspace(-1, 1, 9)

cb = mpl.colorbar.ColorbarBase(
    ax=ax,
    cmap=mpl.cm.RdBu_r,
    norm=norm,
    orientation="horizontal",
    ticks=ticks,
)

cb.ax.tick_params(labelsize=10, length=4, width=0.8)
cb.outline.set_linewidth(0.7)
fig.savefig(outdir / "RdBu_colorbar.svg", bbox_inches="tight")
plt.show()
