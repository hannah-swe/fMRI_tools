import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pycirclize import Circos
import os
import numpy as np

path = "/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/circle_figure/"

# -----------------------------
# 1. CSV-Dateien laden
# -----------------------------
regions = pd.read_csv(os.path.join(path, "regions.csv"))
connections = pd.read_csv(os.path.join(path, "connectivity.csv"))

required_region_cols = {"network", "region"}
required_connection_cols = {"source", "target", "value"}

if not required_region_cols.issubset(regions.columns):
    raise ValueError(f"regions.csv braucht diese Spalten: {required_region_cols}")

if not required_connection_cols.issubset(connections.columns):
    raise ValueError(f"connections.csv braucht diese Spalten: {required_connection_cols}")


# -----------------------------
# 2. Sektoren aus networks bauen
# -----------------------------
network_order = regions["network"].drop_duplicates().tolist()

sectors = {
    network: len(regions[regions["network"] == network])
    for network in network_order
}

circos = Circos(sectors, space=5)


# -----------------------------
# 3. Farben für große Sektoren
# -----------------------------
cmap = plt.get_cmap("tab20c")

network_colors = {
    "XX": cmap(0),
    "Visual": cmap(7),
    "Vestibular": cmap(11),
    "Cerebellum": cmap(15),
}
default_color = "#999999"


# -----------------------------
# 4. Positionen der Regionen speichern
# -----------------------------
region_pos = {}

for sector in circos.sectors:
    network = sector.name
    subregions = regions.loc[regions["network"] == network, "region"].tolist()

    # Äußerer Track: große Netzwerk-Sektoren
    outer_track = sector.add_track((83, 100))
    # outer_track.axis(fc=network_colors.get(network, default_color), ec="none")
    outer_track.text(network, color="black", size=16)

    # Innerer Track: einzelne Hirnregionen
    inner_track = sector.add_track((70, 82))
    inner_track.axis( ec="black", lw=1.5) # fc=network_colors.get(network, default_color),

    for i, region in enumerate(subregions):
        start = i
        end = i + 1
        center = i + 0.5

        region_pos[region] = (network, center)

        inner_track.text(
            region,
            x=center,
            size=16,
            orientation="horizontal"
        )

    # inner_track.xticks_by_interval(
        # interval=1,
        # show_label=False,
        # tick_length=1
    # )


# -----------------------------
# 5. Link-Farben getrennt nach Vorzeichen
# -----------------------------
pos_norm = mcolors.Normalize(vmin=0.7, vmax=2.1)
neg_norm = mcolors.Normalize(vmin=-2.1, vmax=-0.7)

rdBu = cm.RdBu_r

# positive Werte: roter Bereich von RdBu_r
pos_cmap = mcolors.LinearSegmentedColormap.from_list(
    "pos_reds",
    rdBu(np.linspace(0.5, 1.0, 256))
)

# negative Werte: blauer Bereich von RdBu_r
neg_cmap = mcolors.LinearSegmentedColormap.from_list(
    "neg_blues",
    rdBu(np.linspace(0.0, 0.5, 256))
)

# Für Colorbar, falls du weiterhin eine gemeinsame Skala willst
sm_pos = cm.ScalarMappable(norm=pos_norm, cmap=pos_cmap)
sm_neg = cm.ScalarMappable(norm=neg_norm, cmap=neg_cmap)

sm_pos.set_array([])
sm_neg.set_array([])


# -----------------------------
# 6. Verbindungen plotten
# -----------------------------
for _, row in connections.iterrows():
    source = row["source"]
    target = row["target"]
    value = float(row["value"])

    if source not in region_pos:
        raise ValueError(f"Source-Region fehlt in regions.csv: {source}")

    if target not in region_pos:
        raise ValueError(f"Target-Region fehlt in regions.csv: {target}")

    source_link = region_pos[source]
    target_link = region_pos[target]

    if value > 0:
        color = pos_cmap(pos_norm(value))
    elif value < 0:
        color = neg_cmap(neg_norm(value))
    else:
        color = "#999999"

    circos.link_line(
        source_link,
        target_link,
        color=color,
        lw=7
    )


# -----------------------------
# 7. Figure erzeugen + Colorbar
# -----------------------------
fig = circos.plotfig()
fig.set_size_inches(14, 10)

# Hauptachse deutlich nach links/kleiner
main_ax = fig.axes[0]
main_ax.set_position([0.03, 0.08, 0.65, 0.84])

# Neue Colorbar-Achse ganz rechts
# Negative Colorbar
cax_neg = fig.add_axes([0.88, 0.25, 0.02, 0.5])

# Positive Colorbar
cax_pos = fig.add_axes([0.94, 0.25, 0.02, 0.5])

cbar_neg = fig.colorbar(sm_neg, cax=cax_neg)
cbar_pos = fig.colorbar(sm_pos, cax=cax_pos)

cbar_neg.set_label("Negative connectivity")
cbar_pos.set_label("Positive connectivity")

plt.savefig("/data_wgs04/ag-sensomotorik/PPPD/analysis/group_level/plots/circos_connectivity.svg")
plt.show()
for ax in fig.axes:
    print(ax.get_position())