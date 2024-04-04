from atlas_utils import get_atlas
from nilearn import plotting

from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from ttest_mat import get_stats_graphs
from utils import pickle_wrap

import os
from utils import pickle_wrap, stdize
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

def get_beta_graph(sn_inc_conn):
    sn_inc_conn = (sn_inc_conn -
                   np.nanmean(sn_inc_conn, axis=1)[:, None, :, :])
    n_sn = sn_inc_conn.shape[0]
    sn_conn = sn_inc_conn.reshape(-1, 246, 246)
    trils = np.tril_indices(246, k=-1)
    sn_flat = sn_conn[:, trils[0], trils[1]]
    # print(f'{sn_flat.shape=}')
    sn_flat = stdize(sn_flat, axis=0)
    regressors = np.array([[-1, 0, 1] * n_sn]).T

    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    betas = np.dot(XTX_invX, sn_flat)

    Y_pred = np.dot(regressors, betas)
    residual = sn_flat - Y_pred
    sigma_s = np.sum(residual ** 2, axis=0) / (n_sn * 2 - 2)
    ss_x = np.sum(regressors ** 2, axis=0)
    var_beta = sigma_s / ss_x

    z = betas / np.sqrt(var_beta)
    z = -z


    z_graph = np.full((246, 246), np.nan)
    z_graph[trils] = z
    z_graph[trils[1], trils[0]] = z
    return z_graph


def plot_hub_spoke(fp='obj7_fMRI', combine_regions=False, regr=False,
                   all_black=False, only_cortical=True):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              }
    if combine_regions:
        kwargs['combine_regions'] = True
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1,
                    cache_dir='cache')

    atlas = get_atlas(combine_regions=combine_regions)
    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}

        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        atlas['coords'] = [coord for j, coord in enumerate(atlas['coords'])
                             if j not in bad_j]
        nroi = len(atlas['ROI_regions'])
        n_bads = len(bad_j)
        print(f'{nroi=}, {n_bads=}, {nroi - n_bads=}')
        # quit()

        good_j = [j for j in range(nroi) if j not in bad_j]

        sn_inc_conn = sn_inc_conn[:, :, good_j, :]
        sn_inc_conn = sn_inc_conn[:, :, :, good_j]

        # sn_inc_conn[..., bad_j, :] = np.nan
        # sn_inc_conn[..., :, bad_j] = np.nan
        print('Pruned subcortical')

        for key in ['ticks', 'tick_lows', 'tick_labels', ]:
            atlas[key] = [val for val, region in
                          zip(atlas[key], atlas['tick_labels'])
                          if region not in bad_rois]

    if regr:
        z_graph = get_beta_graph(sn_inc_conn)
    else:
        _, _, _, _, _, _, z_graph = \
            get_stats_graphs(sn_inc_conn[:, 0, :, :],
                             sn_inc_conn[:, 2, :, :])
    # upper_thresh = np.nanpercentile(z_graph, 90)
    # z_graph_high = z_graph.copy()
    # z_graph_high[z_graph < upper_thresh] = np.nan
    # plot_connectivity(z_graph_high, atlas=atlas, vmin=-4, vmax=4, minimal=True)
    #
    # z_graph_low = z_graph.copy()
    # lower_thresh = np.nanpercentile(z_graph, 10)
    #
    # z_graph_low[z_graph > lower_thresh] = np.nan
    # plot_connectivity(z_graph_low, atlas=atlas, vmin=-4, vmax=4, minimal=True)
    #
    # plot_connectivity(z_graph, atlas=atlas, vmin=-4, vmax=4, minimal=True)
    #
    # quit()

    # 16 MFG
    # 142 IPL
    # 198 LOC
    # 77 ATL
    i = 77

    # 30 = IFG
    # 76 = ATL
    # 142 = IPL
    # 188 = EVC

    # for i in [30, 76, 142, 202]:
    for i in [78, 79]:
        all_black = False
        fig = plt.figure(figsize=(3.5, 3.5))

        z_graph_ = z_graph.copy()
        if all_black:
            z_graph_[:, :] = 1
            # z_graph_[:, ::3] = 1
            z_graph_[:i] = 0
            z_graph_[i+1:] = 0
            z_graph_[:, i] = z_graph_[i, :]
            vabs = 1
            edge_kw = {'linewidth': 1.5, 'alpha': 0.25}
        else:
            z_graph_[:i] = 0
            z_graph_[i+1:] = 0
            z_graph_[:, i] = z_graph_[i, :]
            vabs = 4
            edge_kw = {'linewidth': 1.5,}

        plotting.plot_connectome(
            z_graph_,
            atlas['coords'],
            edge_threshold=1.,
            edge_vmin=-vabs, edge_vmax=vabs,
            colorbar=False,
            edge_cmap='bone_r' if all_black else 'turbo',
            node_size=1 if all_black else 1.5,
            node_color='k',
            edge_kwargs=edge_kw,
            display_mode='x',
            figure=fig,
            # title=atlas['ROIs'][i] + f' {atlas["coords"][i]}',
        )
        fp_out = fr'result_pics/other/hub_spoke/{atlas["ROIs"][i]}.png'
        plt.savefig(fp_out, dpi=300)
        plt.show()




if __name__ == '__main__':
    plot_hub_spoke()






