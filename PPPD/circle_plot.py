import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pycirclize import Circos
import os

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
    outer_track = sector.add_track((93, 100))
    outer_track.axis(fc=network_colors.get(network, default_color), ec="none")
    outer_track.text(network, color="black", size=16)

    # Innerer Track: einzelne Hirnregionen
    inner_track = sector.add_track((82, 92))
    inner_track.axis(fc=network_colors.get(network, default_color), ec="black", lw=1.5)

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
# 5. Link-Farben anhand value
# -----------------------------
vmax = abs(connections["value"]).max()
norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
cmap = cm.RdBu_r

sm = cm.ScalarMappable(norm=norm, cmap=cmap)


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

    color = cmap(norm(value))

    circos.link_line(
        source_link,
        target_link,
        color=color,
        lw=5
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
cax = fig.add_axes([0.90, 0.25, 0.025, 0.5])

cbar = fig.colorbar(sm, cax=cax)

plt.savefig("/home/hannahschewe/circos_connectivity.svg")
plt.show()
for ax in fig.axes:
    print(ax.get_position())