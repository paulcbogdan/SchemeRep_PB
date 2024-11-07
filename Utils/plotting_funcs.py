from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def plot_connectivity(conn, ticks=None, tick_labels=None, tick_lows=None,
                      atlas=None, title='', fp=None, ax=None,
                      t=False, no_avg=True, cbar_label='', vmin=None, vmax=None,
                      xlabel=None, ylabel=None, tile=.001,
                      tick_low=None, tick_high=None, minimal=False,
                      colorbar=True, cmap='turbo', adjust_HC_AMY=True,
                      dontoverride=False):

    if atlas is not None:
        ticks = atlas['ticks']
        tick_labels = atlas['tick_labels']
        tick_lows = atlas['tick_lows']
    if adjust_HC_AMY and ticks[-1] > 100:
        ticks[20] = 210.5
        ticks[21] = 216.5

    font = {'size': 14}
    matplotlib.rc('font', **font)

    if not no_avg:
        M_connect = np.nanmean(conn, axis=0)
        if t:
            M_connect = M_connect / np.nanstd(conn, axis=0) * np.sqrt(len(conn))
    else:
        M_connect = conn

    if vmin is None:
        vmin = np.nanquantile(M_connect, tile)
        vmax = np.nanquantile(M_connect, 1 - tile)
        print(f'Plot connectivity ({tile=}), {vmin=:.2f}, {vmax=:.2f}')
        vmin = min(vmin, -vmax)
        vmax = max(vmax, -vmin)

    if ax is None:
        plt.figure(figsize=(10, 10))
    else:
        plt.sca(ax)
    if xlabel is None:
        plt.xlabel('Conceptual retrieval IRAF (object)')
        plt.xlabel('Activity object presentation')
        plt.xlabel(xlabel)

    if ylabel is None:
        plt.ylabel('Baseline IRAF (object)')
        plt.ylabel('Encoding object presentation (abs-dif IRAF)')
        plt.ylabel(ylabel)

    if dontoverride:
        pass
    elif M_connect.shape[0] == 52:
        ticks = 0.5 + np.arange(26) * 2
        tick_labels = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG', 'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
                       'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'EVC', 'LOC', 'sOcG', 'Amyg', 'Hipp', 'Str',
                       'Tha']
    elif M_connect.shape[0] == 54:
        ticks = 0.5 + np.arange(27) * 2
        tick_labels = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG',
                       'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
                       'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'PCC', 'ACC', 'EVC',
                       'LOC', 'sOcG', 'Amyg', 'Hipp', 'Str', 'Tha']

    plt.imshow(M_connect, vmin=vmin, vmax=vmax,
               cmap=cmap, interpolation='none')

    fontsize = 20 if len(tick_labels) < 20 else 14

    for low in tick_lows:
        low -= 0.5
        plt.plot([-0.5, M_connect.shape[0]], [low, low], 'k', linewidth=0.5)
        plt.plot([low, low], [-0.5, M_connect.shape[0]], 'k', linewidth=0.5)
    plt.xlim([-0.5, M_connect.shape[0]-0.5])
    plt.ylim([-0.5, M_connect.shape[0]-0.5])
    if minimal:
        plt.gca().invert_yaxis()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.tight_layout()
        if fp is not None:
            Path(fp).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fp, dpi=300)
            print(f'Saving: {fp=}')
        plt.show()
        return
    plt.title(title, fontsize=fontsize * 1.75, pad=12)

    plt.yticks(ticks, tick_labels, fontsize=fontsize)
    plt.xticks(ticks, tick_labels, fontsize=fontsize, rotation=90)
    if colorbar:
        cbar = plt.colorbar(shrink=0.77, aspect=20*0.7, label=cbar_label,
                            pad=0.04,)

        cbar.ax.tick_params(labelsize=fontsize * 1.75)

        tick_tests = cbar.ax.get_yticklabels()[1:-1]  # ends aren't shown for some reason
        lowest_val = tick_tests[0]._y
        not_neg = lowest_val >= 0
        cbar.set_label(cbar_label, rotation=-90, labelpad=20 + not_neg * 15,
                       fontsize=fontsize * 1.75)

    if tick_low is not None or tick_high is not None:
        ticks = list(cbar.get_ticks())[1:-1]
        tick_labels = [f'{t:.2f}' for t in ticks]
        tick_labels[0] = f'{tick_labels[0]} {tick_low}'
        tick_labels[-1] = f'{tick_labels[-1]} {tick_high}'
        cbar.set_ticks(ticks, labels=tick_labels)

    plt.gca().invert_yaxis()
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp, dpi=300)
        print(f'Saving: {fp=}')
    if ax is None:
        plt.show()
