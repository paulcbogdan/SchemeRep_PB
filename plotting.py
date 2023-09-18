from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def plot_connectivity(conn, ticks, tick_labels, tick_lows, title='', fp=None,
                      ax=None, t=False, no_avg=False, cbar_label='',
                      vmin=None, vmax=None):
    font = {'size': 14}
    matplotlib.rc('font', **font)

    if not no_avg:
        M_connect = np.nanmean(conn, axis=0)
        if t:
            M_connect = M_connect / np.nanstd(conn, axis=0) * np.sqrt(len(conn))
    else:
        M_connect = conn
    # M_connect = np.nanmedian(conn, axis=0)
    if vmin is None:
        vmin = np.nanquantile(M_connect, .001)
        vmax = np.nanquantile(M_connect, .999)
    print(f'vmin: {vmin}, vmax: {vmax}')
    # vmin = .2
    # vmax = .8
    if ax is None:
        plt.figure(figsize=(10, 10))
    else:
        plt.sca(ax)
    plt.title(title)
    plt.ylabel('Activity')
    plt.xlabel('IRAF')
    plt.imshow(M_connect, vmin=vmin, vmax=vmax, cmap='turbo')
    plt.yticks(ticks, tick_labels, fontsize=10)
    plt.xticks(ticks, tick_labels, fontsize=10, rotation=90)
    for low in tick_lows:
        plt.plot([0, M_connect.shape[0]], [low, low], 'w', linewidth=0.5)
        plt.plot([low, low], [0, M_connect.shape[0]], 'w', linewidth=0.5)
    plt.xlim([-0.5, M_connect.shape[0]-0.5])
    plt.ylim([-0.5, M_connect.shape[0]-0.5])
    cbar = plt.colorbar(shrink=0.7, aspect=20*0.7, label=cbar_label,
                        )
    cbar.set_label(cbar_label, y=1.05, labelpad=-40, rotation=0)

    # cbar.ax.tick_params(rotation=45, fontsize=12)
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp)
    if ax is None:
        plt.show()
