# import os
# os.chdir(r'C:\PycharmProjects_C\SchemeRep\networks')
# import sys
from atlas_utils import get_atlas
# from corr_RSA_x_vendor import get_module_trialwise_z

from functools import cache
from collections import defaultdict

import numpy as np
import pandas as pd

from old.modularity import get_partition_cross, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from vendor_partitioning import get_vendor_partitions
from utils import timing, pickle_wrap, stdize, get_formula_cols
import scipy.stats as stats
from warnings import filterwarnings
import os

os.environ['R_HOME'] = r'C:\Users\Paul\anaconda3\envs\py312\Lib\R'

filterwarnings('ignore', category=UserWarning)

def get_module_cross_trialwise_z(conn_trials, p_mod0, p_mod1, trialwise=True,
                                 transpose=True):
    if transpose:
        conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)
    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    if trialwise:
        agg_zs = np.nanmean(flat_cross, axis=-1)
        agg_zs = np.nanmean(agg_zs, axis=1) # omit inc axis
        return agg_zs
    else:
        rs = np.nanmean(flat_cross, axis=2)
        agg_rs = np.nanmean(rs, axis=-1)
        return agg_rs

def get_module_trialwise_z(sn_inc_conn_trials, p_module, add_dim=False):
    # sn_inc_conn_trials = sn_inc_activity_std[..., None, :] * \
    #                      sn_inc_activity_std[..., None, :, :]
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1) # omit inc axis
    return sn_agg_trials_dd

def get_memory_p():
    atlas = get_atlas()
    ROIs_mem = ['Hipp']
    p_mem = []
    for i, roi in enumerate(atlas['ROIs']):
        for key_mem in ROIs_mem:
            if key_mem in roi:
                p_mem.append(i)
                break
    return p_mem


@timing
def get_dfs_conn_trials(fp='obj7_fMRI', single=False, w_activity=False,
                        squeeze=False):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='cache')
    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    if squeeze:
        sn_roi_act = np.nanmean(sn_inc_activity, axis=1)
        conn_trials = np.nanmean(conn_trials, axis=1)
        sns = age2idxs[1] + age2idxs[2]
        return sn_roi_act, conn_trials, sns

    if single:
        return conn_trials[[0]], [df_sns_l[0]]
    else:
        return conn_trials, df_sns_l

def get_df_p_x_p(p0, p1, key, fp='obj7_fMRI'):
    conn_trials, df_sns_l = get_dfs_conn_trials(fp)

    if p1 is None:
        sn_agg_trials_dd = get_module_trialwise_z(conn_trials, p0)
    else:
        sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials, p0, p1)
    for i, df_sn in enumerate(df_sns_l):
        df_sn[key] = sn_agg_trials_dd[i, :]

    df_sns = pd.concat(df_sns_l)
    df_sns = df_sns[['sn', 'obj', key]]
    return df_sns

def get_hemi_vendor_df(fp='obj7_fMRI', scrub=False, anat=False):

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat, scrub=scrub)

    atlas = get_atlas()
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos,}
    ps_hemi = defaultdict(list)
    for key, p in ps.items():
        for i in p:
            coord = atlas['coords'][i]
            if coord[0] < 0:
                ps_hemi[f'L{key}'].append(i)
            else:
                ps_hemi[f'R{key}'].append(i)
    ps_hemi.update(ps)

    conn_trials, df_sns_l = get_dfs_conn_trials(fp)

    new_cols = []

    for i, p0 in enumerate(ps_hemi):
        for j, p1 in enumerate(ps_hemi):
            # if i >= j:
            #     continue
            sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials,
                                                            ps_hemi[p0],
                                                            ps_hemi[p1])
            # if p0 == 'dp' and p1 == 'da':
                # print(sn_agg_trials_dd[:, 5])
                # print(sn_agg_trials_dd)
                # quit()
            for k, df_sn in enumerate(df_sns_l):
                df_sn[f'{p0}_{p1}'] = sn_agg_trials_dd[k, :]
                df_sn[f'{p0}_{p1}'] = stats.zscore(df_sn[f'{p0}_{p1}'],
                                                   nan_policy='omit')
            new_cols.append(f'{p0}_{p1}')
    df_sns = pd.concat(df_sns_l)
    return df_sns, new_cols

def get_hemi_cross_vendor_df(fp='obj7_fMRI', scrub=False, anat=False):
    df, cols = pickle_wrap(get_hemi_vendor_df, None, kwargs={'fp': fp, 'scrub': scrub, 'anat': anat},
                           easy_override=True, cache_dir='cache')
    cross_cols = ['Ldp_Rdp', 'Lda_Rda', 'Lvp_Rvp', 'Lva_Rva',]
    rename = {'Ldp_Rdp': 'DP_hemi',
              'Lda_Rda': 'DA_hemi',
              'Lvp_Rvp': 'VP_hemi',
              'Lva_Rva': 'VA_hemi',}
    df_cross = df[['sn', 'obj'] + cross_cols]
    df_cross.rename(rename, axis=1, inplace=True)
    return df_cross, list(rename.values())

@timing
@cache
def get_vendor_df(fp='obj7_fMRI', scrub=False, anat=False,
                  hemis=True, roiwise=False):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='cache')

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat, scrub=scrub)
    assert set(p_d_pos).intersection(p_d_ant) == set()
    assert set(p_d_pos).intersection(p_v_ant) == set()
    assert set(p_d_pos).intersection(p_v_pos) == set()

    n_rois = sn_inc_activity.shape[2]
    matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)

    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]

    diag = np.diag_indices(conn_trials.shape[3])
    conn_trials[:, :, diag[0], diag[1], :] = np.nan

    sn_inc_act_M_p_d = np.nanmean(sn_inc_activity[:, :, p_d_pos, :], axis=(1, 2))
    sn_inc_act_M_a_d = np.nanmean(sn_inc_activity[:, :, p_d_ant, :], axis=(1, 2))
    sn_inc_act_M_p_v = np.nanmean(sn_inc_activity[:, :, p_v_pos, :], axis=(1, 2))
    sn_inc_act_M_a_v = np.nanmean(sn_inc_activity[:, :, p_v_ant, :], axis=(1, 2))
    sn_inc_act_M_overall = np.nanmean(sn_inc_activity, axis=(1, 2))

    ad_else = list(set(range(246)) - set(p_d_ant))
    pd_else = list(set(range(246)) - set(p_d_pos))
    av_else = list(set(range(246)) - set(p_v_ant))
    pv_else = list(set(range(246)) - set(p_v_pos))
    dd_else = list(set(range(246)) - set(p_d_ant + p_d_pos))
    vv_else = list(set(range(246)) - set(p_v_ant + p_v_pos))

    conn_keys = ['dd', 'vv',
                 'dv_ant', 'dv_pos',
                 'dpva', 'vpda',
                 'pd_else', 'ad_else',
                 'pv_else', 'av_else',
                 'dd_else', 'vv_else' ]
    conn_ps = [(p_d_pos, p_d_ant), (p_v_pos, p_v_ant),
               (p_d_ant, p_v_ant), (p_d_pos, p_v_pos),
               (p_d_pos, p_v_ant), (p_v_pos, p_d_ant),
               (p_d_pos, pd_else), (p_d_ant, ad_else),
               (p_v_pos, pv_else), (p_v_ant, av_else),
               (p_dorsal, dd_else), (p_ventral, vv_else)]
    key2conn = {}
    for key, (p0, p1) in zip(conn_keys, conn_ps):
        key2conn[key] = get_module_cross_trialwise_z(conn_trials, p0, p1)
    key2conn['FC_all'] = get_module_trialwise_z(conn_trials, list(range(246)))

    if roiwise:
        quads = ['dp', 'da', 'vp', 'va']
        quad2p = {'dp': p_d_pos, 'da': p_d_ant, 'vp': p_v_pos, 'va': p_v_ant}
        for i in range(246):
            for quad in quads:
                key2conn[f'{quad}_{i}'] = \
                    get_module_cross_trialwise_z(conn_trials, [i],
                                                 quad2p[quad])

    all_new_cols = set()
    for i, df_sn in enumerate(df_sns_l):
        old_cols = set(df_sn.columns)
        df_sn['dp'] = df_sn['pd_M'] = sn_inc_act_M_p_d[i, :]
        df_sn['da'] = df_sn['ad_M'] = sn_inc_act_M_a_d[i, :]
        df_sn['d_M'] = df_sn['pd_M'] + df_sn['ad_M']
        df_sn['vp'] = df_sn['pv_M'] = sn_inc_act_M_p_v[i, :]
        df_sn['va'] = df_sn['av_M'] = sn_inc_act_M_a_v[i, :]
        df_sn['v_M'] = df_sn['pv_M'] + df_sn['av_M']

        df_sn['all_M'] = df_sn['d_M'] + df_sn['v_M']
        df_sn['brain_M'] = sn_inc_act_M_overall[i, :]

        for key, conn in key2conn.items():
            df_sn[key] = conn[i, :]

        df_sn['dd_vv'] = df_sn['dd'] + df_sn['vv']
        df_sn['cross'] = df_sn['dv_ant'] + df_sn['dv_pos']
        df_sn['vendor'] = df_sn['dd_vv'] - df_sn['cross']
        new_cols = list(set(df_sn.columns) - old_cols)
        all_new_cols.update(new_cols)
        for col in new_cols:
            df_sn[col] = stats.zscore(df_sn[col], nan_policy='omit')
    df_sns = pd.concat(df_sns_l)
    df_sns['age'] = df_sns['sn'].apply(lambda x: int(x[0]))

    if hemis:
        print('Onto hemi-cross vendor df')
        df_hemi, cols_hemi = pickle_wrap(get_hemi_cross_vendor_df, None, kwargs={'fp': fp,
                                                                                 'anat': anat,
                                                                                 'scrub': scrub}, easy_override=True,
                                         cache_dir='cache')
        df_sns = df_sns.merge(df_hemi, on=['sn', 'obj'])
        all_new_cols.update(cols_hemi)

    return df_sns, all_new_cols
def vendor_lmer_Feb12(fp='obj7_fMRI'):
    df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                kwargs={'fp': fp, 'scrub': False,
                                        'anat': True},
                                easy_override=True, cache_dir='cache')
    p_mem = get_memory_p()
    df_mem = get_df_p_x_p(p_mem, None, 'hc', fp=fp)
    df = df.merge(df_mem, on=['sn', 'obj'], how='left')

    for col in ['age', 'hit_hit', 'hc']: # 'con_hit', 'vis_hit',
        df[col] = stats.zscore(df[col], nan_policy='omit')

    formula = ('dd ~ vv + dv_ant + dv_pos + '
               'pd_else + ad_else + pv_else + av_else + ' # pv_else + av_else + 
               'FC_all + ' #  dd_else + # pd_else + ad_else +
               'dp + da + vp + va + '
               '(1 | sn)')

    formula = ('vv ~ dd + dv_ant + dv_pos + '
               'pd_else + ad_else + pv_else + av_else + ' # pv_else + av_else + 
               'FC_all + ' #  dd_else + # pd_else + ad_else +
               'dp + da + vp + va + '
               '(1 | sn)')

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    formula = ('dd_vv ~ dv_dv + FC_all + '
               'dp + da + vp + va + '
               '(1 | sn)')

    formula = ('dd ~ dp * da + FC_all +'
               '(1 | sn)')

    from pymer4 import Lmer
    cols = get_formula_cols(df, formula)
    df_vals = df[cols].dropna()
    model = Lmer(formula, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    # iv = 'hit_hit'
    # brain = 'vv'
    # dvs = ['dd_vv', 'cross', 'dd', 'vv']
    # for dv in dvs:
    #     formula_gen = f'{dv} ~ 1 + {brain}*{iv} + age*{iv} + ' \
    #                   f'(1 + {brain}*{iv} + age*{iv} | sn)'
    #     cols = get_formula_cols(df, formula_gen)
    #     df_vals = df[cols].dropna()
    #     model = Lmer(formula_gen, data=df_vals)
    #     model.fit(REML=True, verbose=False, summary=False)
    #     print(model.summary())
    #     print('-'*10)
    #     print(f'{dv=}')
    #     print('-'*50)


if __name__ == '__main__':
    # pass
    # hemi_vendor_corr()
    # meta_corr_triangle()
    # plot_conn_matrix()
    # vendor_lmer()
    vendor_lmer_Feb12()
    # get_trialwise_vendor()
    # plot_meta_corr_matrix()