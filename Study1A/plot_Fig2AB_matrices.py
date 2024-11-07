import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np

from Utils.atlas_funcs import get_atlas
from Study1A.load_Study1A_funcs import load_FC
from Study1A.plot_Fig2CD_partitions import get_1sample_ttest_matrix
from Utils.plotting_funcs import plot_connectivity

from utils import stdize
from Utils.pickle_wrap_funcs import pickle_wrap


def get_beta_graph(sn_inc_conn):
    sn_inc_conn = (sn_inc_conn -
                   np.nanmean(sn_inc_conn, axis=1)[:, None, :, :])
    n_sn = sn_inc_conn.shape[0]
    n_roi = sn_inc_conn.shape[2]

    sn_conn = sn_inc_conn.reshape(-1, n_roi, n_roi)
    trils = np.tril_indices(n_roi, k=-1)
    sn_flat = sn_conn[:, trils[0], trils[1]]
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

    z_graph = np.full((n_roi, n_roi), np.nan)
    z_graph[trils] = z
    z_graph[trils[1], trils[0]] = z
    return z_graph


def plot_matrix_Fig3(combine_regions=True, only_cortical=True, regr=True,
                     rotate=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }

    if combine_regions:
        kwargs['combine_regions'] = True

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, _ = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1,
                    cache_dir='cache')

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False, lifu_labels=True)
    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}

        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        atlas['coords'] = [coord for j, coord in enumerate(atlas['coords'])
                           if j not in bad_j]
        nroi = len(atlas['ROI_regions'])
        n_bads = len(bad_j)
        print(f'{nroi=}, {n_bads=}, {nroi - n_bads=}')

        good_j = [j for j in range(nroi) if j not in bad_j]

        sn_inc_conn = sn_inc_conn[:, :, good_j, :]
        sn_inc_conn = sn_inc_conn[:, :, :, good_j]

        print('Pruned subcortical')

        for key in ['ticks', 'tick_lows', 'tick_labels', ]:
            atlas[key] = [val for val, region in
                          zip(atlas[key], atlas['tick_labels'])
                          if region not in bad_rois]

    ticks_new = []
    tick_lows_new = []
    for tick, tick_low in zip(atlas['ticks'], atlas['tick_lows'], ):
        ticks_new.append(tick)
        ticks_new.append(tick + 1)
        tick_lows_new.append(tick_low)
        tick_lows_new.append(tick_low + 1)
    atlas['ticks'] = ticks_new
    atlas['tick_lows'] = tick_lows_new

    if regr:
        z_graph = get_beta_graph(sn_inc_conn)
    else:
        _, _, _, _, _, _, z_graph = \
            get_1sample_ttest_matrix(sn_inc_conn[:, 0, :, :],
                                     sn_inc_conn[:, 2, :, :])

    if rotate:
        z_graph = z_graph[::-1, ::-1]
        atlas['ticks'] = atlas['ticks'][::-1]
        atlas['tick_lows'] = atlas['tick_lows'][::-1]
        atlas['tick_labels'] = atlas['tick_labels'][::-1]

    z_graph_high = z_graph.copy()
    upper_thresh = np.nanpercentile(z_graph, 85)
    z_graph_high[z_graph < upper_thresh] = np.nan
    fp = 'result_pics/Fig2/Fig2B_upper_matrix.png'
    plot_connectivity(z_graph_high, atlas=atlas, vmin=-3, vmax=3, minimal=True,
                      fp=fp)

    z_graph_low = z_graph.copy()
    lower_thresh = np.nanpercentile(z_graph, 15)
    z_graph_low[z_graph > lower_thresh] = np.nan
    fp = 'result_pics/Fig2/Fig2B_lower_matrix.png'
    plot_connectivity(z_graph_low, atlas=atlas, vmin=-3, vmax=3, minimal=True,
                      fp=fp)

    atlas['ticks'] = atlas['ticks'][1::2]
    atlas['tick_labels'] = atlas['tick_labels'][:23]
    atlas['tick_lows'] = atlas['tick_lows'][1::2]

    fp = 'result_pics/Fig2/Fig2A_matrix.png'
    plot_connectivity(z_graph, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], vmin=-3, vmax=3, minimal=False,
                      fp=fp)


if __name__ == '__main__':
    plot_matrix_Fig3()
