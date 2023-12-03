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
                      vmin=None, vmax=None, xlabel=None, ylabel=None):


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
    if xlabel is None:
        plt.xlabel('Conceptual retrieval IRAF (object)')
        plt.xlabel('Activity object presentation')
        plt.xlabel(xlabel)

    if ylabel is None:
        plt.ylabel('Baseline IRAF (object)')
        plt.ylabel('Encoding object presentation (abs-dif IRAF)')
        plt.ylabel(ylabel)
    print(f'{len(ticks)=} | {ticks=}')
    print(f'{len(tick_labels)=} | {tick_labels=}')

    plt.imshow(M_connect, vmin=vmin, vmax=vmax, cmap='turbo')
    plt.yticks(ticks, tick_labels, fontsize=10)
    plt.xticks(ticks, tick_labels, fontsize=10, rotation=90)
    # for low in tick_lows:
    #     plt.plot([0, M_connect.shape[0]], [low, low], 'w', linewidth=0.5)
    #     plt.plot([low, low], [0, M_connect.shape[0]], 'w', linewidth=0.5)
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
    # size_mid = int(vabs_minus_thresh / thresh * n)# * 2)
    # print(f'{size_mid=}, {vabs_minus_thresh=:.3f} {thresh} {n}')
    # print()


    vals_mid = np.repeat(np.array([0, 0, 0, 1])[:, None], size_mid,
                         axis=1).T
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
    from statsmodels.stats.multitest import multipletests
    import scipy.stats as stats
    from nilearn import plotting
    atlas['coords'] = np.array(atlas['coords'])
    # print(atlas['coords'])
    # print(coords_not_nans)
    # quit()
    Ms[np.isinf(Ms)] = np.nan
    n_non_nans = np.sum(~np.isnan(Ms))
    img_data = np.zeros(atlas['maps'].shape)
    atlas_data = atlas['maps'].get_fdata()
    Ms = [np.min([m, 5]) for m in Ms]
    thresh = 2.45 # p = .01
    thresh = 1
    # thresh = 2.05
    # thresh = 1.7 # p = .05
    num_above_thresh = np.sum(np.abs(Ms) > thresh)
    print(f'{thresh=} | {num_above_thresh=}')
    print(Ms)
    for i, val in enumerate(Ms):
        print(f'go={val:.3f}')
        img_data[atlas_data == (i + 1)] = val
        above_thresh = np.abs(val) > thresh
        continue
        if above_thresh:
            # print(f'{i=}, {val=:.3f}')
            # continue
            mat = np.zeros((n_non_nans, n_non_nans))
            mat[i, :] = 1
            mat[:, i] = 1
            colors = ['b'] * i + ['r'] + ['b'] * (n_non_nans - i - 1)
            coords_not_nans = atlas['coords'][~np.isnan(Ms)]
            plotting.plot_connectome(mat, coords_not_nans,
                                     edge_threshold=0.1,
                                     edge_kwargs={'linewidth': 1,
                                                  'color': 'k'},
                                     node_size=10,
                                     node_color=colors,
                                     # node_kwargs={'c': 'k',
                                     #              'size': 10},
                                     title=f'i = {i}',
                                     )
            plotting.show()
    # quit()

    vabs = np.nanmax(np.abs(Ms))
    print(f'{vabs=}')
    Ms = np.array(Ms)

    Ms_clean = Ms[~np.isnan(Ms)]
    ps = [stats.t.sf(np.abs(m), 29) for m in Ms_clean]
    #
    rej, ps_corr, alpha_sidak, alpha_conf = \
        multipletests(ps, alpha=0.05, method='fdr_bh')
    #
    print(f'{Ms_clean=}')
    print(f'{ps=}')
    print(f'{ps_corr=}')
    #
    #
    # quit()
    cmap = get_split_cmap(vabs, thresh, 'cold_hot')
    # cmap = get_split_cmap(vabs, thresh, 'Reds')


    img = image.new_img_like(atlas['maps'], img_data)
    # print(f'{thresh=}')
    fig = plotting.plot_img_on_surf(img, threshold=thresh,
                                    cmap=cmap, title=title)

    # fig[1][4].set_xticklabels([-2.5, None, 2.5])
    plotting.show()



if __name__ == '__main__':
    atlas = get_atlas(combine_regions=False)
    L_SFG_coords = []
    L_SFG_idxs = []
    R_IFG_coords = []
    R_IFG_idxs = []
    node_colors = []
    idx = 0
    for i, roi in enumerate(atlas['ROIs']):
        if 'SFG_L' in roi:
            L_SFG_coords.append(atlas['coords'][i])
            node_colors.append('r')
            L_SFG_idxs.append(idx)
            print(f'{i=}', atlas['coords'][i])
            idx += 1
        elif 'IFG_R' in roi:
            R_IFG_coords.append(atlas['coords'][i])
            node_colors.append('b')
            R_IFG_idxs.append(idx)
            idx += 1
    coords = list(L_SFG_coords) + list(R_IFG_coords)
    n_rois = len(coords)
    mat = np.zeros((n_rois, n_rois))
    mesh = np.meshgrid(L_SFG_idxs, R_IFG_idxs)
    mat[mesh] = 1
    mesh = np.meshgrid(R_IFG_idxs, L_SFG_idxs)
    mat[mesh] = 1
    # plotting.plot_markers([1], [[-27, 43, 31]])
    # plotting.show()
    # quit()
    plotting.plot_connectome(mat, coords,
                             edge_threshold=0.1,
                             edge_kwargs={'linewidth': 1,
                                          'color': 'k'},
                             node_size=10,
                             node_color=node_colors,
                             # node_kwargs={'c': 'k',
                             #              'size': 10},
                             title=f'L SFG x R IFG')
    plotting.show()