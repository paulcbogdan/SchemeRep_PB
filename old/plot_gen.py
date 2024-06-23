from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import image
from nilearn import plotting

from atlas_utils import get_combined_BNA, get_atlas


def plot_connectivity(conn, ticks=None, tick_labels=None, tick_lows=None,
                      atlas=None, title='', fp=None, ax=None,
                      t=False, no_avg=True, cbar_label='', vmin=None, vmax=None,
                      xlabel=None, ylabel=None, tile=.001,
                      tick_low=None, tick_high=None, minimal=False,
                      colorbar=True, cmap='turbo', adjust_HC_AMY=True):

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
        tick_labels = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG',
                       'MTG', 'ITG', 'FuG', 'PhG', 'pSTS',
                       'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'PCC', 'ACC', 'EVC',
                       'LOC', 'sOcG', 'Amyg', 'Hipp', 'Str',
                       'Tha']

    # tick_labels = tick_labels[::-1]


    plt.imshow(M_connect, vmin=vmin, vmax=vmax,
               cmap=cmap,#'turbo' if vmin < -1 else 'CMRmap',
               interpolation='none')

    fontsize = 20 if len(tick_labels) < 20 else 14

    for low in tick_lows:
        # print(f'{low=}')
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
        plt.show()
        return
    plt.title(title, fontsize=fontsize * 1.75, pad=12)
    plt.yticks(ticks, tick_labels, fontsize=fontsize)
    plt.xticks(ticks, tick_labels, fontsize=fontsize, rotation=90)
    if colorbar:
        cbar = plt.colorbar(shrink=0.77, aspect=20*0.7, label=cbar_label,
                            pad=0.04,
                            )

        # test = cbar.ax.get_yticklabels()
        # print(f'{test=}')
        cbar.ax.tick_params(labelsize=fontsize * 1.2)

        tick_tests = cbar.ax.get_yticklabels()[1:-1]  # ends aren't shown for some reason
        lowest_val = tick_tests[0]._y
        not_neg = lowest_val >= 0
        cbar.set_label(cbar_label, rotation=-90, labelpad=15 + not_neg * 15,
                       fontsize=fontsize * 1.5)

    if tick_low is not None or tick_high is not None:
        ticks = list(cbar.get_ticks())[1:-1]
        tick_labels = [f'{t:.2f}' for t in ticks]
        tick_labels[0] = f'{tick_labels[0]} {tick_low}'
        tick_labels[-1] = f'{tick_labels[-1]} {tick_high}'
        cbar.set_ticks(ticks, labels=tick_labels)

    # print(f'{test=}')
    # print(vars(test[0]))
    # print(test[0]._y)
    # quit()
    # cbar.ax.tick_params(rotation=45, fontsize=12)
    plt.gca().invert_yaxis()
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        # plt.tight_layout()
        plt.savefig(fp)
    if ax is None:
        # plt.tight_layout()
        plt.show()



# def plot_surf(combine_regions=True, bilateral=False):
#     atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)

def get_split_cmap(vabs, thresh, cmap, blue_half=False):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    n = 1000
    prop_black = thresh / vabs
    if prop_black > 1:
        prop_black = 1
        # raise ValueError(f'{prop_black=}')
    n_black = int(n * prop_black)
    # vals_black = np.repeat(np.array([0, 0, 0, 1])[:, None], n_black,
    #                      axis=1).T
    vals_black = np.repeat(np.array([0.5, 0.5, 0.5, 1])[:, None], n_black,
                         axis=1).T
    prop_colored = 1 - prop_black
    # print(f'{prop_colored=}')
    n_colored = int(n * prop_colored)
    print(f'{n_colored=}')
    if blue_half:
        vals_colored = cmap(np.linspace(0.05, .9, n_colored))
    else:
        vals_colored = cmap(np.linspace(0.05, .9, n))
    #I don't like the purple...
    vals_low = vals_colored[:int(n_colored/2)]
    # print(len(vals_low))
    # quit()
    vals_high = vals_colored[-int(n_colored/2):]

    # vals_low = vals_colored[:500]
    # vals_high = vals_colored[500:]


    # vals_low = vals_colored[int(n/10):int(n/2)]
    # vals_high = vals_colored[int(n/2):n-int(n/10)]
    vals = np.concatenate([vals_low, vals_black, vals_high])
    if blue_half:
        vals = vals[vals.shape[0] // 2:, :]

    # plt.plot(vals[:, 0])
    # plt.show()
    # print(vals[:, 0])
    # quit()

    cmap = ListedColormap(vals)

    # print(vals)
    return cmap



def my_plot_surf(Ms, atlas, title, fp_out=None,
                 neg='', pos='', thresh=1.65, vmax=4,
                 cmap='hot_cold'):
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
        # if val > 1:
        #     print(f'{i} | {val}')
        if i == 102:
            # fix bad part of mesh, FuG appears in Hipp or PhG or w/e
            idxs = np.indices(img_data.shape)
            slicer = idxs[0] > 28
            slicer1 = idxs[0] < 32
            img_data[(atlas_data == (i + 1)) & slicer & slicer1] = 0
        elif i == 103:
            idxs = np.indices(img_data.shape)
            slicer = idxs[0] == 67
            img_data[(atlas_data == (i + 1)) & slicer] = 0
        # above_thresh = np.abs(val) > thresh
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



    cmap = get_split_cmap(vabs, thresh, 'rainbow_r')
    # cmap = get_split_cmap(vabs, thresh, cmap)

    # cmap = 'cold_hot'

    img_data[img_data < thresh] = thresh - 0.1
    img_data[img_data < 0] = 0

    img = image.new_img_like(atlas['maps'], img_data)
    print(f'{thresh=}')
    print(f'{vmax=}')
    fig, axs = plotting.plot_img_on_surf(img, threshold=thresh,
                                         cmap=cmap, title=title,
                                         vmin=-vmax, vmax=vmax,
                                         inflate=True,
                                         surf_mesh='fsaverage7',
                                         avg_method='median')

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
        print(f'Saving fig: {fp_out=}')
        plt.clf()
    quit()

    plotting.plot_stat_map(img,
                           vmin=-vmax, vmax=vmax,
                           threshold=thresh, draw_cross=False,
                           display_mode='x',)
                           # cut_coords=[-22, -24, -26, -28])
    plt.show()
    # plotting.plot_stat_map(img,
    #                        vmin=-vmax, vmax=vmax,
    #                        threshold=thresh, draw_cross=False,
    #                        display_mode='x',
    #                        cut_coords=[-30, -32, -34, -36])
    # plt.show()
    quit()



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