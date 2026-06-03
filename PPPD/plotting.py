import numpy as np

def get_colorbar_limits(data, threshold, two_sided=False, padding=1.05):
    if two_sided:
        visible = np.abs(data[np.abs(data) > threshold])
        if len(visible) > 0:
            vmax = np.ceil(np.nanmax(visible) * padding * 10) / 10
        else:
            vmax = threshold + 0.1
        vmin = -vmax
    else:
        visible = data[data > threshold]
        if len(visible) > 0:
            vmax = np.ceil(np.nanmax(visible) * padding * 10) / 10
        else:
            vmax = threshold + 0.1
        vmin = threshold

    return vmin, vmax