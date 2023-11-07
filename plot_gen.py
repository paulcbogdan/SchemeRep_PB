from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import image
from nilearn import plotting

from atlas_utils import get_combined_BNA, get_atlas


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
        # vmin = np.nanquantile(M_connect, .001)
        vmin = np.nanmin(M_connect)
        # vmax = np.nanquantile(M_connect, .999)
        # vmax = max(vmax, 4)
        vmax = np.nanmax(M_connect)

    print(f'vmin: {vmin}, vmax: {vmax}')
    # vmin = .2
    # vmax = .8
    if ax is None:
        plt.figure(figsize=(10, 10))
    else:
        plt.sca(ax)
    plt.title(title)
    plt.xlabel('Conceptual retrieval IRAF (object)')
    plt.ylabel('Baseline IRAF (object)')

    plt.xlabel('Activity object presentation')
    plt.ylabel('Encoding object presentation (abs-dif IRAF)')

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

# def plot_surf(combine_regions=True, bilateral=False):
#     atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)

def get_split_cmap(vabs, thresh, cmap):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    n = 256
    vals = cmap(np.linspace(0., 1., n))
    vals_st = vals[:int(n/2)]
    vals_end = vals[int(n/2):]
    # vabs = np.min([vabs, 5.])
    vabs_minus_thresh = (vabs - thresh)
    # print(f'{vabs_minus_thresh=}')
    # print(f'{thresh=}')
    # print(f'{n=}')
    size_mid = int(vabs_minus_thresh / thresh * n / 2)
    # print(f'{size_mid=}, {vabs_minus_thresh=:.3f} {thresh} {n}')
    # print()


    vals_mid = np.repeat(np.array([0, 0, 0, 1])[:, None], size_mid, axis=1).T
    vals = np.concatenate([vals_st, vals_mid, vals_end])
    # print(vals_mid.shape)

    # print(vals.shape)
    # quit()

    cmap = ListedColormap(vals)

    # print(vals)
    return cmap

    # quit()
    cmap = ListedColormap(cmap(np.linspace(-vabs, vabs, 128)))


    print(cmap)
    quit()


def my_plot_surf(Ms, atlas, title):
    img_data = np.zeros(atlas['maps'].shape)
    atlas_data = atlas['maps'].get_fdata()
    Ms = [np.min([m, 5]) for m in Ms]
    for i, val in enumerate(Ms):
        i += 1
        img_data[atlas_data == i] = val

    vabs = np.nanmax(np.abs(Ms))
    print(f'{vabs=}')
    thresh = 2
    cmap = get_split_cmap(vabs, thresh, 'cold_hot')

    img = image.new_img_like(atlas['maps'], img_data)
    # print(f'{thresh=}')
    fig = plotting.plot_img_on_surf(img, threshold=thresh,
                                    cmap=cmap, title=title)

    fig[1][4].set_xticklabels([-2.5, None, 2.5])
    plotting.show()

if __name__ == '__main__':
    pass