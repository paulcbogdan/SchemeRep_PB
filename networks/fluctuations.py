from analyze_rs import load_act_conn, prep_conn_ps
from atlas_utils import get_atlas
from old.plot_gen import plot_connectivity
from utils import pickle_wrap, stdize
from collections import defaultdict

from vendor_lmers import get_module_trialwise_z, get_dfs_conn_trials, get_module_cross_trialwise_z
from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg

import matplotlib.pyplot as plt
from vendor_partitioning import get_vendor_partitions
from analyze_rs import load_resting_data
from functools import partial
import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

# TODO: set up windows backups

# Test for fluctuation by seeing mean distance from t[i] to t[i+n] for all n
#   for each subject
# Test also the relationship between T[i+1] - T[i] and T[i+n] - T[i+n-1]
#    make a histogram (given that T[n] is at -1 z,
#    what is the distribution of T[n+1]


# #1 priority:
# In the resting-state data, look up the dFC correlations between DMN-FPCN-etc.
#   make a triangle matrix. Evaluate also: In terms of dFC, do the networks
#   actually operate in sync at all?
# Verify that defining the networks as the average connectivity among pairs
#   indeed yields above zero.

# TODO: double-check that i'm never subtracting a TR mean across all voxels
#   Calculate FC_all based on non-quadrant voxels

# Hypothesis: These networks are static, a baseline
#    I am researching fluctuations within this baseline

# TODO: Look at thalamo-cortical loops, which Garret mentioned in 2023

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

def compare_sanity_vs_pb_lss(norm_std=False):
    f = partial(load_resting_data, raw_enc=False, lss_enc=True, lsa=False,
                YA_only=True)
    sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                 easy_override=True)

    f = partial(load_resting_data, YA_only=True, sanity=True)
    sn_roi_act_s, sns_s, conn_trials_s = load_act_conn(norm_std, f=f,
                                                 easy_override=False)




def get_df_networks(fp='pb_lss', norm_std=False, zscore=False, f=None,
                    tight_anat=True):

    if f is not None:
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=False)
    elif fp == 'sanity':
        f  = partial(load_resting_data, YA_only=False, sanity=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'raw_enc':
        f = partial(load_resting_data, raw_enc=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=False)
    elif fp == 'pb_lsa':
        f = partial(load_resting_data, raw_enc=False, lss_enc=True, lsa=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'pb_lss':
        f = partial(load_resting_data, raw_enc=False, lss_enc=True, lsa=False,
                    YA_only=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'rs':
        print('load act conn')
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     )
    elif fp == 'rs_noclean':
        f = partial(load_resting_data, compcor=False, clean=False)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=False)
    elif fp == 'rs_nocc':
        f = partial(load_resting_data, compcor=False)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=False)
    elif fp == 'rs_light':
        f = partial(load_resting_data, compcor=True, light=True, clean=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=True)
    elif fp == 'rs_medium':
        f = partial(load_resting_data, medium=True, clean=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=False,
                                                     f=f, YA_only=False)
    elif fp == 'rs_trad':
        f = partial(load_resting_data, clean=True, trad=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=True)
    else:
        sn_roi_act, conn_trials, sns = \
            pickle_wrap(get_dfs_conn_trials, kwargs={'fp': fp, 'single': False,
                                                     'w_activity': True,
                                                     'squeeze': True})
    print('Onto get_df_networks...')
    sn_roi_act = stdize(sn_roi_act, axis=2)
    sn_roi_act = stdize(sn_roi_act, axis=1)
    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]

    network2p = pickle_wrap(get_network_partitions)

    key2conn = {}
    for network, p in network2p.items():
        key2conn[network] = get_module_trialwise_z(conn_trials[:, None], p)

    networks = list(key2conn)

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_version=1)
    print(f'{len(p_d_pos)=}')
    print(f'{len(p_d_ant)=}')
    print(f'{len(p_v_pos)=}')
    print(f'{len(p_v_ant)=}')
    # key2pair = {'dd': (p_d_pos, p_d_ant), 'vv': (p_v_pos, p_v_ant),
    #             'dv_ant': (p_d_ant, p_v_ant), 'dv_pos': (p_d_pos, p_v_pos),}

    conn_keys, conn_ps = prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos,
                                      p_v_ant, p_v_pos)
    for key, (p0, p1) in zip(conn_keys, conn_ps):
    # for key, (p0, p1) in key2pair.items():
        key2conn[key] = get_module_cross_trialwise_z(conn_trials[:, None],
                                                     p0, p1)
    key2conn['FC_all'] = get_module_trialwise_z(conn_trials[:, None],
                                                list(range(246)))

    act_keys = ['dp', 'da', 'vp', 'va']
    act_p = [p_d_pos, p_d_ant, p_v_pos, p_v_ant]
    key2p_M = {}
    for key, p in zip(act_keys, act_p):
        key2p_M[key] = np.nanmean(sn_roi_act[:, p, :], axis=1)

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

def partial_corr_df(df, cols, cov):
    cols = [col for col in cols if col not in cov]
    ar = np.full((len(cols), len(cols)), np.nan, dtype=float)
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if col_i == 'dv_ant' and col_j == 'dv_pos':
                pass
            elif (col_i.split('_')[0] in col_j or
                    col_j.split('_')[0] in col_i):
                continue
            if i < j:
                # print(df[[col_i, col_j]])
                try:
                    out = (pg.partial_corr(data=df, x=col_i, y=col_j,
                                           covar=cov).
                           round(3))
                except AssertionError as e:
                    ar[i, j] = np.nan

                ar[i, j] = out['r'].values[0]
                ar[j, i] = ar[i, j]
    df_result = pd.DataFrame(ar, index=cols, columns=cols)
    print(df_result)


def analyze_networks(fp='rs_medium'):
    df, networks = pickle_wrap(get_df_networks,
                               kwargs={'fp': fp,
                                       'norm_std': False,
                                       'zscore': True},
                               easy_override=True)


    df['da_dp'] = df['da'] + df['dp']
    df['va_vp'] = df['va'] + df['vp']

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    networks = networks[:-4]
    conn = df[networks].corr()
    # print(conn)
    conn = np.array(conn)
    conn[np.diag_indices_from(conn)] = np.nan
    ticks = list(np.arange(len(networks)))
    tick_labels = networks
    tick_lows = np.arange(len(networks))
    title = 'Resting-state avg. network correlations'

    # plot_connectivity(conn, ticks, tick_labels, tick_lows, title=title, no_avg=True, cbar_label='Pearson\'s r',
    #                   vmin=-0.5, vmax=0.5)


    networks = ['dd', 'vv', 'dv_ant', 'dv_pos',
                'dd_vv', 'dv_dv']

    networks += ['no_no']
    print(df[networks].corr())


    # 'no_no', 'pd_no', 'ad_no', 'av_no', 'pv_no'
    partial_corr_df(df, networks, cov=[#'FC_all',
                                       'pd_no', 'ad_no', 'av_no', 'pv_no'])
    quit()

    # 'dd', 'vv', 'dv_ant', 'dv_pos'

    from pymer4.models import Lmer
    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    model = Lmer('dd_vv ~ dv_dv + FC_all + (1 | sn)', data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())

if __name__ == '__main__':
    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)

    analyze_networks()
    # get_dfs_conn_trials(fp='obj7_fMRI', single=False)
    # analyze_networks()

