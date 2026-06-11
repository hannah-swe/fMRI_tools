from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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


# einfarbige LUT
def save_solid_clut(rgb, filename, n_nodes=21):
    r, g, b = [int(255 * c) for c in rgb[:3]]

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
            f.write(f"nodergba{i}={r}|{g}|{b}|255\n")


# weiß -> tab10-Farbe
def save_monochrome_clut(rgb, filename, n_nodes=21):
    r0, g0, b0 = rgb[:3]

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
            alpha = i / (n_nodes - 1)

            r = int(255 * ((1 - alpha) + alpha * r0))
            g = int(255 * ((1 - alpha) + alpha * g0))
            b = int(255 * ((1 - alpha) + alpha * b0))

            f.write(f"nodergba{i}={r}|{g}|{b}|255\n")



# RdBu scale (change cmap to get another matplotlib colormap)
save_clut(cmap, outdir / "RdBu_r.clut")
# blue half
blue_half = lambda x: cmap(x * 0.5)
save_clut(blue_half, outdir / "Bu.clut")
# red half
red_half = lambda x: cmap(0.5 + x * 0.5)
save_clut(red_half, outdir / "Rd.clut")

print("Bu.clut und Rd.clut gespeichert.")


# Neue tab10-Farben
tab10 = plt.get_cmap("tab10")

# einfarbig, ohne Abstufung
# save_solid_clut(tab10(0), outdir / "tab10_blue_solid.clut")
# save_solid_clut(tab10(1), outdir / "tab10_orange_solid.clut")
# save_solid_clut(tab10(2), outdir / "tab10_green_solid.clut")
# save_solid_clut(tab10(3), outdir / "tab10_red_solid.clut")
# save_solid_clut(tab10(4), outdir / "tab10_purple_solid.clut")

# optional: weiß -> Farbe
save_monochrome_clut(tab10(6), outdir / "tab10_pink.clut")
save_monochrome_clut(tab10(8), outdir / "tab10_green.clut")
save_monochrome_clut(tab10(9), outdir / "tab10_blue.clut")

print("Alle CLUT-Dateien gespeichert.")