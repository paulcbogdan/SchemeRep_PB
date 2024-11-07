from collections import defaultdict

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from Study1A.modularity_funcs import get_partition_matrix, get_FC_between_ROIs
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
from Study2A.load_Study2A_funcs import load_rs_BOLD
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap


def get_network_partitions():
    from nichord.coord_labeler import get_idx_to_label
    atlas = get_atlas()
    idx_to_label = pickle_wrap(get_idx_to_label, None,
                               kwargs={'coords': atlas['coords'],
                                       'atlas': 'yeo'})
    network2p = defaultdict(list)
    for i, network in idx_to_label.items():
        network2p[network].append(i)
    return network2p


def get_hemi_ps(p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    atlas = get_atlas()
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos, }
    ps_hemi = defaultdict(list)
    for key, p in ps.items():
        for i in p:
            coord = atlas['coords'][i]
            if coord[0] < 0:
                ps_hemi[f'L{key}'].append(i)
            else:
                ps_hemi[f'R{key}'].append(i)
    return ps_hemi


def get_df_networks(zscore=False, anat_ver=3, add_hemi=True,
                    combine_regions=False):
    sn_roi_act, sns, conn_trials = load_rs_BOLD()
    print('Onto get_df_networks...')

    network2p = pickle_wrap(get_network_partitions)

    key2conn = {}
    for network, p in network2p.items():
        key2conn[network] = get_module_trialwise_z(conn_trials[:, None], p)

    networks = list(key2conn)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                             anat_ver=anat_ver,
                             combine_regions=combine_regions)

    if add_hemi:
        ps_hemi = get_hemi_ps(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
        keys_both = []
        for key, p0 in ps_hemi.items():
            for key1, p1 in ps_hemi.items():
                key_both = f'{key}_{key1}'
                key2conn[key_both] = (
                    get_FC_between_ROIs(conn_trials[:, None],
                                        ps_hemi[key], ps_hemi[key1]))
                keys_both.append(key_both)
        networks += keys_both

    conn_keys, conn_ps = prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos,
                                      p_v_ant, p_v_pos)
    for key, (p0, p1) in zip(conn_keys, conn_ps):
        key2conn[key] = get_FC_between_ROIs(conn_trials[:, None],
                                            p0, p1)

    key2conn['FC_all'] = get_module_trialwise_z(conn_trials[:, None],
                                                list(range(246)))

    act_keys = ['dp', 'da', 'vp', 'va']
    act_p = [p_d_pos, p_d_ant, p_v_pos, p_v_ant]
    key2p_M = {}
    for key, p in zip(act_keys, act_p):
        key2p_M[key] = np.nanmean(sn_roi_act[:, p, :], axis=1)
    act_keys += 'no'
    p_no = [i for i in range(246) if i not in p_dorsal + p_ventral]
    key2p_M['no'] = np.nanmean(sn_roi_act[:, p_no, :], axis=1)

    df_as_d = defaultdict(list)
    n_TRs = sn_roi_act.shape[-1]
    for i, sn in enumerate(sns):
        for key, conn in key2conn.items():
            vals = conn[i, :]
            if zscore: vals = stats.zscore(vals)
            df_as_d[key].extend(vals)
            n_TRs = len(vals)
        for key, M in key2p_M.items():
            vals = M[i, :]
            if zscore: vals = stats.zscore(vals)
            df_as_d[key].extend(vals)

        df_as_d['sn'].extend([sn] * n_TRs)
    df = pd.DataFrame(df_as_d)
    networks += ['dd', 'vv', 'dv_ant', 'dv_pos']
    return df, networks


def partial_corr_df(df, cols, cov, verbose=1):
    cols = [col for col in cols if col not in cov]
    ar = np.full((len(cols), len(cols)), np.nan, dtype=float)
    print(df[cols])
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i < j:
                try:
                    out = (pg.partial_corr(data=df, x=col_i, y=col_j,
                                           covar=cov).
                           round(3))

                    ar[i, j] = out['r'].values[0]
                    ar[j, i] = ar[i, j]
                except AssertionError as e:
                    ar[i, j] = np.nan
                    ar[j, i] = np.nan

    df_result = pd.DataFrame(ar, index=cols, columns=cols)
    if verbose:
        print(df_result)
    return ar


def prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    ad_else = list(set(range(246)) - set(p_d_ant))
    pd_else = list(set(range(246)) - set(p_d_pos))
    av_else = list(set(range(246)) - set(p_v_ant))
    pv_else = list(set(range(246)) - set(p_v_pos))

    no_match = list(set(range(246)) -
                    set(p_d_ant + p_d_pos + p_v_ant + p_v_pos))

    dd_else = list(set(range(246)) - set(p_d_ant + p_d_pos))
    dv_ant_else = list(set(range(246)) - set(p_d_ant + p_v_ant))
    vv_else = list(set(range(246)) - set(p_v_ant + p_v_pos))
    dv_pos_else = list(set(range(246)) - set(p_d_pos + p_v_pos))

    p_ant = list(set(p_d_ant + p_v_ant))
    p_pos = list(set(p_d_pos + p_v_pos))

    # allow diagonal
    pd_no = list(set(range(246)) - set(p_d_pos + p_d_ant + p_v_pos))
    ad_no = list(set(range(246)) - set(p_d_ant + p_d_pos + p_v_ant))
    pv_no = list(set(range(246)) - set(p_v_pos + p_v_ant + p_d_pos))
    av_no = list(set(range(246)) - set(p_v_ant + p_v_pos + p_d_ant))

    conn_keys = ['dd', 'vv',
                 'dv_ant', 'dv_pos',
                 'dpva', 'vpda',
                 'pd_else', 'ad_else',
                 'pv_else', 'av_else',
                 'dd_else', 'vv_else',
                 'dv_ant_else', 'dv_pos_else',
                 'pd_no', 'ad_no',
                 'pv_no', 'av_no',
                 'dd_no', 'vv_no',
                 'dv_ant_no', 'dv_pos_no',
                 'pd_no2', 'ad_no2',
                 'pv_no2', 'av_no2',
                 'no_no',

                 'pd_no_L', 'pd_no_R',
                 'ad_no_L', 'ad_no_R',
                 'pv_no_L', 'pv_no_R',
                 'av_no_L', 'av_no_R',
                 ]

    p_d_pos_L = [i for i in p_d_pos if i % 2 == 0]
    p_d_pos_R = [i for i in p_d_pos if i % 2 == 1]
    p_d_ant_L = [i for i in p_d_ant if i % 2 == 0]
    p_d_ant_R = [i for i in p_d_ant if i % 2 == 1]
    p_v_pos_L = [i for i in p_v_pos if i % 2 == 0]
    p_v_pos_R = [i for i in p_v_pos if i % 2 == 1]
    p_v_ant_L = [i for i in p_v_ant if i % 2 == 0]
    p_v_ant_R = [i for i in p_v_ant if i % 2 == 1]
    conn_ps = [(p_d_pos, p_d_ant), (p_v_pos, p_v_ant),
               (p_d_ant, p_v_ant), (p_d_pos, p_v_pos),
               (p_d_pos, p_v_ant), (p_v_pos, p_d_ant),
               (p_d_pos, pd_else), (p_d_ant, ad_else),
               (p_v_pos, pv_else), (p_v_ant, av_else),
               (p_dorsal, dd_else), (p_ventral, vv_else),
               (p_ant, dv_ant_else), (p_pos, dv_pos_else),
               (p_d_pos, no_match), (p_d_ant, no_match),
               (p_v_pos, no_match), (p_v_ant, no_match),
               (p_dorsal, no_match), (p_ventral, no_match),
               (p_ant, no_match), (p_pos, no_match),
               (p_d_pos, pd_no), (p_v_pos, pv_no),
               (p_d_ant, ad_no), (p_v_ant, av_no),
               (no_match, no_match),
               (p_d_pos_L, no_match), (p_d_pos_R, no_match),
               (p_d_ant_L, no_match), (p_d_ant_R, no_match),
               (p_v_pos_L, no_match), (p_v_pos_R, no_match),
               (p_v_ant_L, no_match), (p_v_ant_R, no_match),
               ]
    return conn_keys, conn_ps


def get_module_trialwise_z(sn_inc_conn_trials, p_module):
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1)  # omit inc axis
    return sn_agg_trials_dd
