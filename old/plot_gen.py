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
                      vmin=None, vmax=None, xlabel=None, ylabel=None,
                      tile=.001, tick_low=None, tick_high=None):

    # conn[np.triu_indices_from(conn, k=0)] = np.nan

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
        vmin = np.nanquantile(M_connect, tile)
        # vmin = np.nanmin(M_connect)
        vmax = np.nanquantile(M_connect, 1 - tile)
        print(f'Plot connectivity ({tile=}), {vmin=:.2f}, {vmax=:.2f}')

        # vmax = max(vmax, 4)
        # vmax = np.nanmax(M_connect)

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

    if M_connect.shape[0] == 52:
        ticks = 0.5 + np.arange(26) * 2
        tick_labels = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG', 'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
                       'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'EVC', 'LOC', 'sOcG', 'Amyg', 'Hipp', 'Str',
                       'Tha']
    elif M_connect.shape[0] == 54:
        ticks = 0.5 + np.arange(27) * 2
        tick_labels = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG', 'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
                       'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'PCC', 'ACC', 'EVC', 'LOC', 'sOcG', 'Amyg', 'Hipp', 'Str',
                       'Tha']

    plt.imshow(M_connect, vmin=vmin, vmax=vmax, cmap='turbo',
               interpolation='none')
    plt.yticks(ticks, tick_labels, fontsize=10)
    plt.xticks(ticks, tick_labels, fontsize=10, rotation=90)
    for low in tick_lows:
        low -= 0.5
        plt.plot([-0.5, M_connect.shape[0]], [low, low], 'k', linewidth=0.5)
        plt.plot([low, low], [-0.5, M_connect.shape[0]], 'k', linewidth=0.5)
    plt.xlim([-0.5, M_connect.shape[0]-0.5])
    plt.ylim([-0.5, M_connect.shape[0]-0.5])
    cbar = plt.colorbar(shrink=0.7, aspect=20*0.7, label=cbar_label,)
    cbar.set_label(cbar_label, y=1.05, labelpad=-40, rotation=0)

    if tick_low is not None or tick_high is not None:
        ticks = list(cbar.get_ticks())[1:-1]
        tick_labels = [f'{t:.2f}' for t in ticks]
        tick_labels[0] = f'{tick_labels[0]} {tick_low}'
        tick_labels[-1] = f'{tick_labels[-1]} {tick_high}'
        cbar.set_ticks(ticks, labels=tick_labels)
    # quit()

    # cbar.ax.tick_params(rotation=45, fontsize=12)
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fp)
    if ax is None:
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# def plot_surf(combine_regions=True, bilateral=False):
#     atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)

def get_split_cmap(vabs, thresh, cmap):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    n = 1000
    prop_black = thresh / vabs
    if prop_black > 1:
        prop_black = 1
        # raise ValueError(f'{prop_black=}')
    n_black = int(n * prop_black)
    vals_black = np.repeat(np.array([0, 0, 0, 1])[:, None], n_black,
                         axis=1).T
    prop_colored = 1 - prop_black
    n_colored = int(n * prop_colored)
    vals_colored = cmap(np.linspace(0., 1., n))
    vals_low = vals_colored[int(n/10):int(n/2)]
    vals_high = vals_colored[int(n/2):n-int(n/10)]
    vals = np.concatenate([vals_low, vals_black, vals_high])


    cmap = ListedColormap(vals)

    # print(vals)
    return cmap



def my_plot_surf(Ms, atlas, title, fp_out=None,
                 neg='', pos='', thresh=1.65, vmax=4):
    from statsmodels.stats.multitest import multipletests
    import scipy.stats as stats
    from nilearn import plotting
    atlas['coords'] = np.array(atlas['coords'])
    Ms = np.array(Ms)
    Ms[np.isinf(Ms)] = np.nan
    n_non_nans = np.sum(~np.isnan(Ms))
    img_data = np.zeros(atlas['maps'].shape)
    atlas_data = atlas['maps'].get_fdata()
    # Ms = [np.min([m, 5]) for m in Ms]

    if fp_out is not None:
        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)

    num_above_thresh = np.sum(np.abs(Ms) > thresh)
    for i, val in enumerate(Ms):
        img_data[atlas_data == (i + 1)] = val
        above_thresh = np.abs(val) > thresh
        continue
        # if above_thresh:
        #     mat = np.zeros((n_non_nans, n_non_nans))
        #     mat[i, :] = 1
        #     mat[:, i] = 1
        #     colors = ['b'] * i + ['r'] + ['b'] * (n_non_nans - i - 1)
        #     coords_not_nans = atlas['coords'][~np.isnan(Ms)]
        #     plotting.plot_connectome(mat, coords_not_nans,
        #                              edge_threshold=0.1,
        #                              edge_kwargs={'linewidth': 1,
        #                                           'color': 'k'},
        #                              node_size=10,
        #                              node_color=colors,
        #                              # node_kwargs={'c': 'k',
        #                              #              'size': 10},
        #                              title=f'i = {i}',
        #                              )
        #     plotting.show()

    vabs = np.nanmax(np.abs(Ms))
    # Ms = np.array(Ms)
    # Ms_clean = Ms[~np.isnan(Ms)]
    # ps = [stats.t.sf(np.abs(m), 29) for m in Ms_clean]
    # rej, ps_corr, alpha_sidak, alpha_conf = \
    #     multipletests(ps, alpha=0.05, method='fdr_bh')

    cmap = get_split_cmap(vabs, thresh, 'cold_hot')
    # cmap = 'cold_hot'
    img = image.new_img_like(atlas['maps'], img_data)

    fig, axs = plotting.plot_img_on_surf(img, threshold=thresh,
                                    cmap=cmap, title=title,
                                    vmin=-vmax,
                                    vmax=vmax)

    if thresh > 5:
        axs[4].set_xticks([-vmax, -thresh, thresh, vmax],
                          [-vmax, -thresh, thresh, vmax],
                          fontsize=8)
    elif neg:
        axs[4].set_xticks([-vmax, -2, 2, vmax],
                          [f'({neg})',
                           '-2', '2',
                           f'({pos})'], fontsize=8)
    if fp_out is None:
        plotting.show()
    else:
        fig.savefig(fp_out)
        plt.clf()



if __name__ == '__main__':
    atlas = get_atlas(combine_regions=False)
    L_SFG_coords = []
    L_SFG_idxs = []
    R_IFG_coords = []
    R_IFG_idxs = []
    node_colors = []
    idx = 0
    for i, roi in enumerate(atlas['ROIs']):
        if '_L' in roi and ('SFG' in roi or 'IFG' in roi or 'MFG' in roi or
                            'OrG' in roi or 'ACC' in roi):
            L_SFG_coords.append(atlas['coords'][i])
            node_colors.append('g')
            L_SFG_idxs.append(idx)
            idx += 1
        elif '_R' in roi and ('SFG' in roi or 'IFG' in roi or 'MFG' in roi or
                              'OrG' in roi or 'ACC' in roi):
            R_IFG_coords.append(atlas['coords'][i])
            node_colors.append('g')
            R_IFG_idxs.append(idx)
            idx += 1

    coords = list(L_SFG_coords) + list(R_IFG_coords)
    n_rois = len(coords)
    mat = np.zeros((n_rois, n_rois))
    mesh = np.meshgrid(L_SFG_idxs, R_IFG_idxs)
    mat[mesh] = 1
    mesh = np.meshgrid(R_IFG_idxs, L_SFG_idxs)
    mat[mesh] = 1
    plotting.plot_connectome(mat, coords,
                             edge_threshold=0.1,
                             edge_kwargs={'linewidth': 0.1,
                                          'color': 'k'},
                             node_size=10,
                             node_color=node_colors,
                             # node_kwargs={'c': 'k',
                             #              'size': 10},
                             # title=f'L SFG x R IFG'
                             title = ''
                             )
    plotting.show()