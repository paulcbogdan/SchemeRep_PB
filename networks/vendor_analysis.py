# import os
# os.chdir(r'C:\PycharmProjects_C\SchemeRep\networks')
# import sys
# sys.path.extend([r'C:\PycharmProjects_C\SchemeRep'])

from collections import defaultdict

from fin_plot_conn_matrix import plot_conn_matrix
from network_IRAF import get_formula_cols

from functools import cache

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from old.modularity import get_partition_cross
from old.network_funcs import load_FC_for_Lifu
from vendor_partitioning import get_vendor_partitions
from utils import timing, pickle_wrap, stdize
from collections import Counter
from copy import copy
import scipy.stats as stats


def get_module_cross_trialwise_z(conn_trials, p_mod0, p_mod1):
    # conn_trials = sn_inc_activity_std[..., None, :] * \
    #                      sn_inc_activity_std[..., None, :, :]
    # conn_trials = sn_inc_activity_std[..., None, :] + \
    #                      sn_inc_activity_std[..., None, :, :]
    conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)
    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    agg_cross = np.nanmean(flat_cross, axis=-1)
    agg_cross = np.nanmean(agg_cross, axis=1) # omit inc axis
    return agg_cross

def scrub_plot_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, plot=False, scrub=False,
                 anat=False):
    atlas = get_atlas()
    from nichord import plot_glassbrain
    idx_to_quadrant = {i: 'PD' for i in p_d_pos}
    idx_to_quadrant.update({i: 'PV' for i in p_v_pos})
    idx_to_quadrant.update({i: 'AD' for i in p_d_ant})
    idx_to_quadrant.update({i: 'AV' for i in p_v_ant})
    node_sizes = []
    quadrant2labels = defaultdict(list)
    for i in range(len(atlas['labels'])):
        if i not in idx_to_quadrant:
            idx_to_quadrant[i] = 'N/A'
            node_sizes.append(0)
        else:
            quadrant = idx_to_quadrant[i]
            label = atlas['labels'][i].split(' ')[1].split('_')[0]
            quadrant2labels[quadrant].append(label)
            # print(f'{i}, {quadrant}: {label}')
            node_sizes.append(5)
    for k, v in quadrant2labels.items():
        print(f'{k}: {Counter(v)}')

    valid_labels = {'AD': {'IFG', 'SFG', 'MFG', 'OrG', 'ACC', 'PrG'},
                    'AV': {'ATL', 'PhG', 'Hipp', 'STG', 'ITG', 'MTG', 'FuG'}, # INS? AMY?
                    'PD': {'IPL', 'Pcun', 'PCC', 'SPL', 'sOcG'},
                    'PV': {'EVC', 'LOC', 'ITG', 'FuG', 'MTG'}}
    for i, quadrant in idx_to_quadrant.items():
        if quadrant == 'N/A':
            continue
        label = atlas['labels'][i].split(' ')[1].split('_')[0]
        if label not in valid_labels[quadrant]:
            idx_to_quadrant[i] = f'mislabeled_{quadrant}'
            node_sizes[i] = 1

    if plot:
        anat_str = '_anat' if anat else ''
        dir_out = r'C:\PycharmProjects_C\SchemeRep\nichord_plots\vendor'
        fn_glass = fr'glass_first_scrub_colored2{anat_str}.png'
        fp_glass = fr'{dir_out}\{fn_glass}'
        # fp_glass = fr'{dir_out}\glass_first_scrub_colored.png'
        coords = atlas['coords']
        edges = [(i, i) for i in range(len(coords))]
        edge_weights = [0] * len(edges)
        network_colors = {'AD': 'red', 'AV': 'dodgerblue',
                          'PD': 'limegreen', 'PV': 'orange',
                          'mislabeled_AD': 'darkred',
                          'mislabeled_AV': 'darkblue',
                          'mislabeled_PD': 'darkgreen',
                          'mislabeled_PV': 'darkgoldenrod',
                          'N/A': 'black'}
        # network_order = ['PV', 'PD', 'AV', 'AD',
        #                  'mislabeled_PV', 'mislabeled_PD',
        #                  'mislabeled_AV', 'mislabeled_AD',
        #                  'N/A']
        # glass_kwargs = {'node_size': node_sizes, 'linewidths': 15,
        #                 'network_colors': network_colors}
        # plot_and_combine(dir_out, fn_glass, idx_to_quadrant, edges,
        #                  coords=coords, network_order=network_order,
        #                  network_colors=network_colors,
        #                  glass_kwargs=glass_kwargs)
        # quit()
        plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                        coords, node_size=node_sizes, linewidths=15,
                        network_colors=network_colors,)

        for key in ['AD', 'AV', 'PD', 'PV']:
            network_colors[f'mislabeled_{key}'] = network_colors[key]
        node_sizes_copy = copy(node_sizes)
        for i, size in enumerate(node_sizes):
            if size < max(node_sizes) and size > 0:
                node_sizes[i] = 0
                node_sizes_copy[i] = max(node_sizes)

        fp_glass = fr'{dir_out}\glass_first_original2{anat_str}.png'
        plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                        coords, node_size=node_sizes_copy, linewidths=15,
                        network_colors=network_colors,)


        fp_glass = fr'{dir_out}\glass_first_scrubbed2{anat_str}.png'
        plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                        coords, node_size=node_sizes, linewidths=15,
                        network_colors=network_colors,)

    # if scrub:
    f = lambda i: 'mislabeled' not in idx_to_quadrant[i]
    p_d_ant_new = list(filter(f, p_d_ant))
    p_d_pos_new = list(filter(f, p_d_pos))
    p_v_ant_new = list(filter(f, p_v_ant))
    p_v_pos_new = list(filter(f, p_v_pos))
    return p_d_ant_new, p_d_pos_new, p_v_ant_new, p_v_pos_new
    # else:
    #     return p_d_ant, p_d_pos, p_v_ant, p_v_pos



def get_anat_vendor_partitions():
    def labels2idxs(target):
        return [i for i, label in enumerate(labels) if
                any([l in label for l in target])]

    atlas = get_atlas()
    coords, labels = atlas['coords'], atlas['labels']
    split_keys = ['PhG', 'ITG', 'FuG']
    split2l = defaultdict(list)
    for coord, label in zip(coords, labels):
        for key in split_keys:
            if key in label:
                split2l[key].append(coord)
    split2median = {}
    for key, l in split2l.items():
        split2l[key] = np.array(l)
        split2median[key] = np.median(split2l[key][:, 1])
    for i, (coord, label) in enumerate(zip(coords, labels)):
        for key in split_keys:
            if key in label:
                if coord[1] > split2median[key]:
                    labels[i] = f'{key}_a'
                else:
                    labels[i] = f'{key}_p'

    p_d_ant_labels = ['IFG', 'SFG', 'MFG', 'OrG']
    p_d_pos_labels = ['IPL', 'Pcun', 'PCC']
    p_v_ant_labels = ['ATL', 'PhG_a', 'ITG_a']
    p_v_pos_labels = ['EVC', 'LOC', 'ITG_p', 'FuG_p', 'PhG_p']
    p_d_ant = labels2idxs(p_d_ant_labels)
    p_d_pos = labels2idxs(p_d_pos_labels)
    p_v_ant = labels2idxs(p_v_ant_labels)
    p_v_pos = labels2idxs(p_v_pos_labels)
    return p_d_ant, p_d_pos, p_v_ant, p_v_pos

@timing
@cache
def get_trialwise_vendor(fp='obj7_fMRI', thr=2.0, scrub=False, anat=True):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    if anat:
        p_d_ant, p_d_pos, p_v_ant, p_v_pos = get_anat_vendor_partitions()
    else:
        p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
            get_vendor_partitions(age=2, flip=True)

    if scrub:
        p_d_ant, p_d_pos, p_v_ant, p_v_pos = scrub_plot_p(p_d_ant, p_d_pos,
                                                          p_v_ant, p_v_pos,
                                                          scrub=scrub,
                                                          plot=True,
                                                          anat=anat)
    print(f'{p_d_ant=}')
    print(f'{p_d_pos=}')
    print(f'{p_v_ant=}')
    print(f'{p_v_pos=}')

    # plot_ROI_scores(ts, results_coords, fp_out='trash.png', show=True,
    #                 vmin=0, vmax=2, title='all')

    assert set(p_d_pos).intersection(p_d_ant) == set()
    assert set(p_d_pos).intersection(p_v_ant) == set()
    assert set(p_d_pos).intersection(p_v_pos) == set()

    n_rois = sn_inc_activity.shape[2]
    matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]
                            # idk why I can't just broadcast matrix_mask

    sn_inc_act_M_pos_d = np.nanmean(sn_inc_activity[:, :, p_d_pos, :],
                                    axis=(1, 2))
    sn_inc_act_M_ant_d = np.nanmean(sn_inc_activity[:, :, p_d_ant, :],
                                    axis=(1, 2))
    sn_inc_act_M_pos_v = np.nanmean(sn_inc_activity[:, :, p_v_pos, :],
                                    axis=(1, 2))
    sn_inc_act_M_ant_v = np.nanmean(sn_inc_activity[:, :, p_v_ant, :],
                                    axis=(1, 2))
    sn_inc_act_M_overall = np.nanmean(sn_inc_activity, axis=(1, 2))

    sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                    p_d_pos, p_d_ant)
    sn_agg_trials_vv = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                    p_v_pos, p_v_ant)
    # plt.hist(sn_agg_trials_dd.flatten(), bins=25, range=(-0.5, 0.5))
    # plt.show()
    # print(sn_agg_trials_vv.shape)
    # quit()

    sn_agg_trials_dv_ant = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                        p_d_ant, p_v_ant)
    sn_agg_trials_dv_pos = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                        p_d_pos, p_v_pos)
    for i, df_sn in enumerate(df_sns_l):
        df_sn['pd_M'] = sn_inc_act_M_pos_d[i, :]
        df_sn['ad_M'] = sn_inc_act_M_ant_d[i, :]
        df_sn['d_M'] = df_sn['pd_M'] + df_sn['ad_M']
        df_sn['pv_M'] = sn_inc_act_M_pos_v[i, :]
        df_sn['av_M'] = sn_inc_act_M_ant_v[i, :]
        df_sn['v_M'] = df_sn['pv_M'] + df_sn['av_M']
        df_sn['all_M'] = df_sn['d_M'] + df_sn['v_M']
        df_sn['brain_M'] = sn_inc_act_M_overall[i, :]

        df_sn['dd'] = sn_agg_trials_dd[i, :]
        df_sn['vv'] = sn_agg_trials_vv[i, :]
        df_sn['dv_ant'] = sn_agg_trials_dv_ant[i, :]
        df_sn['dv_pos'] = sn_agg_trials_dv_pos[i, :]
        df_sn['dd_vv'] = df_sn['dd'] + df_sn['vv'] #- \
                       # df_sn['dv_ant'] - df_sn['dv_pos']
        df_sn['cross'] = df_sn['dv_ant'] + df_sn['dv_pos']
        df_sn['vendor'] = df_sn['dd_vv'] - df_sn['cross']
        # df_sn['dd_minus_vv']
    df_sns = pd.concat(df_sns_l)
    df_sns['age'] = df_sns['sn'].apply(lambda x: int(x[0]))
    return df_sns

def vendor_lmer():
    df = pickle_wrap(None, get_trialwise_vendor, kwargs={'fp': 'obj7_fMRI',
                                                         'scrub': True},
                     cache_dir='cache', easy_override=True)

    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    df['brain_M'] = stats.zscore(df['brain_M'], nan_policy='omit')
    formula_gen = 'vv ~ 1 + dd*age + inc + brain_M + dv_ant + dv_pos + ' \
                  '(1 + dd*age + inc + brain_M + dv_ant + dv_pos | sn)'
    cols = get_formula_cols(df, formula_gen)
    df_vals = df[cols].dropna()
    from pymer4 import Lmer
    model = Lmer(formula_gen, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

def plot_meta_corr_matrix():
    df_vdr = pickle_wrap(None, get_trialwise_vendor, kwargs={'fp': 'obj7_fMRI'},
                         cache_dir='cache', easy_override=False)
    cols = ['dd', 'vv', 'dv_ant', 'dv_pos']
    # TODO: for the DV_pos/ant x dd/vv, make sure that there are no overlapping
    #   edges
    corr = df_vdr[cols].corr()
    np.set_printoptions(edgeitems=10)
    np.set_printoptions(linewidth=200)
    ar_str = np.full((len(cols), len(cols)), '', dtype=object)
    for i, row in enumerate(corr.values):
        for j, r in enumerate(row):
            if r > .99999:
                ar_str[i][j] = '-'
                continue
            z = np.arctanh(r)
            z_std = 1 / np.sqrt(len(df_vdr) - 3)
            z_low = z - 1.96 * z_std
            z_high = z + 1.96 * z_std
            r_low = np.tanh(z_low)
            r_high = np.tanh(z_high)
            ar_str[i][j] = f'{r:.2f} ({r_low:.2f}, {r_high:.2f})'
    print(ar_str)
    print('-'*300)


if __name__ == '__main__':
    # plot_conn_matrix()
    vendor_lmer()
    # get_trialwise_vendor()
    # plot_meta_corr_matrix()