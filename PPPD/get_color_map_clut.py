from pathlib import Path
import matplotlib.cm as cm
import numpy as np

# Zielpfad anpassen
outdir = Path(r"/data_wgs04/ag-sensomotorik/MRIcroGL/Resources/lut")


# Originale RdBu-Colormap
cmap = cm.get_cmap("RdBu_r")

def save_clut(cmap, filename, n_nodes=21):
    with open(filename, "w") as f:
        f.write("[FLT]\nmin=0\nmax=0\n")

        f.write("[INT]\n")
        f.write(f"numnodes={n_nodes}\n")

        f.write("[BYT]\n")
        for i in range(n_nodes):
            intensity = round(i * 255 / (n_nodes - 1))
            f.write(f"nodeintensity{i}={intensity}\n")

        f.write("[RGBA255]\n")
        for i in range(n_nodes):
            r, g, b, a = cmap(i / (n_nodes - 1))
            f.write(
                f"nodergba{i}="
                f"{int(r*255)}|{int(g*255)}|{int(b*255)}|255\n"
            )

# Gesamte RdBu-Skala
save_clut(cmap, outdir / "RdBu_r.clut")

# Nur blaue Hälfte
blue_half = lambda x: cmap(x * 0.5)
save_clut(blue_half, outdir / "Bu.clut")

# Nur rote Hälfte
red_half = lambda x: cmap(0.5 + x * 0.5)
save_clut(red_half, outdir / "Rd.clut")

print("Bu.clut und Rd.clut gespeichert.")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

outdir = Path("/Pfad/zu/deinem/Ordner")
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
    figsize=(7, 1.0),
    constrained_layout=True
)

norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
ticks = [0, 0.5, 1.0, 1.5, 2.0]

for ax, cmap in zip(axes, [blue_half, red_half]):
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
        ticks=ticks
    )
    cb.ax.tick_params(labelsize=10, length=4, width=0.8)
    cb.outline.set_linewidth(0.5)

# als Vektorgrafik speichern
fig.savefig(outdir / "RdBu_split_colorbars.svg", bbox_inches="tight")
fig.savefig(outdir / "RdBu_split_colorbars.pdf", bbox_inches="tight")

plt.show()