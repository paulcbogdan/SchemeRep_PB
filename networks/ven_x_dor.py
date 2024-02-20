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

def get_module_cross_trialwise_z(conn_trials, p_mod0, p_mod1, trialwise=True):
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

def get_module_trialwise_z(sn_inc_conn_trials, p_module):
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

def get_DMN_p(exclude_ps=None):
    atlas = get_atlas()

    from nichord.coord_labeler import get_idx_to_label
    idx_to_label = pickle_wrap(None, get_idx_to_label,
                               kwargs={'coords': atlas['coords'],
                                       'atlas': 'yeo'},
                               )
    DMN_idxs = [idx for idx, label in idx_to_label.items() if 'DMN' in label]
    # print(DMN_idxs)


    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=False, scrub=False)
    # exclude_ps = p_dorsal + p_ventral

    for i in DMN_idxs:
        BNA_label = atlas['ROIs'][i]
        if i in p_d_pos:
            print(f'{BNA_label} ({i}): In: Dor-Pos')
        elif i in p_d_ant:
            print(f'{BNA_label} ({i}): In: Dor-Ant')
        elif i in p_v_pos:
            print(f'{BNA_label} ({i}): In: Ven-Pos')
        elif i in p_v_ant:
            print(f'{BNA_label} ({i}): In: Ven-Ant')
        else:
            print(f'{BNA_label} ({i}): Not in any')

    print('--------')

    cnt = defaultdict(lambda: 0)
    for i in p_d_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_d_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Ant: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Ant: {cnt}')


@timing
def get_dfs_conn_trials(fp='obj7_fMRI', single=False):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
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
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos,
          }
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
    df, cols = pickle_wrap(None, get_hemi_vendor_df,
                           kwargs={'fp': fp, 'scrub': scrub, 'anat': anat},
                           cache_dir='cache', easy_override=True)
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
                  hemis=True):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

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
    # conn_trials = np.repeat(matrix_mask[None, None, ..., None],
    #                         conn_trials.shape[-1], axis=4) * conn_trials[..., :]
    #                         idk why I can't just broadcast matrix_mask

    sn_inc_act_M_p_d = np.nanmean(sn_inc_activity[:, :, p_d_pos, :],axis=(1, 2))
    sn_inc_act_M_a_d = np.nanmean(sn_inc_activity[:, :, p_d_ant, :], axis=(1, 2))
    sn_inc_act_M_p_v = np.nanmean(sn_inc_activity[:, :, p_v_pos, :], axis=(1, 2))
    sn_inc_act_M_a_v = np.nanmean(sn_inc_activity[:, :, p_v_ant, :], axis=(1, 2))
    sn_inc_act_M_overall = np.nanmean(sn_inc_activity, axis=(1, 2))
    sn_trials_dd = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_d_ant)
    sn_trials_vv = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_v_ant)
    sn_trials_dv_a = get_module_cross_trialwise_z(conn_trials, p_d_ant, p_v_ant)
    sn_trials_dv_p = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_pos)
    sn_trials_dpva = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_ant)
    sn_trials_vpda = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_d_ant)

    all_new_cols = set()
    for i, df_sn in enumerate(df_sns_l):
        old_cols = set(df_sn.columns)
        df_sn['pd_M'] = sn_inc_act_M_p_d[i, :]
        df_sn['ad_M'] = sn_inc_act_M_a_d[i, :]
        df_sn['d_M'] = df_sn['pd_M'] + df_sn['ad_M']
        df_sn['pv_M'] = sn_inc_act_M_p_v[i, :]
        df_sn['av_M'] = sn_inc_act_M_a_v[i, :]
        df_sn['v_M'] = df_sn['pv_M'] + df_sn['av_M']

        df_sn['all_M'] = df_sn['d_M'] + df_sn['v_M']
        df_sn['brain_M'] = sn_inc_act_M_overall[i, :]

        df_sn['dd'] = sn_trials_dd[i, :]
        df_sn['vv'] = sn_trials_vv[i, :]
        df_sn['dv_ant'] = sn_trials_dv_a[i, :]
        df_sn['dv_pos'] = sn_trials_dv_p[i, :]
        df_sn['dpva'] = sn_trials_dpva[i, :]
        df_sn['vpda'] = sn_trials_vpda[i, :]

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
        df_hemi, cols_hemi = pickle_wrap(None, get_hemi_cross_vendor_df,
                                         kwargs={'fp': fp,
                                                 'anat': anat,
                                                 'scrub': scrub},
                                         cache_dir='cache',
                                         easy_override=True)
        df_sns = df_sns.merge(df_hemi, on=['sn', 'obj'])
        all_new_cols.update(cols_hemi)

    return df_sns, all_new_cols

def vendor_lmer(fp='obj7_fMRI'):
    df, vndr_cols = pickle_wrap(None, get_vendor_df, kwargs={'fp': fp,
                                                  'anat': False,
                                                  'scrub': False,
                                                  'hemis': False},
                     cache_dir='cache', easy_override=True)

    p_mem = get_memory_p()
    df_mem = get_df_p_x_p(p_mem, None, 'hc', fp=fp)
    df = df.merge(df_mem, on=['sn', 'obj'], how='left')


    # df['age'] = stats.zscore(df['age'], nan_policy='omit')
    # df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    # df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    # df['brain_M'] = stats.zscore(df['brain_M'], nan_policy='omit')
    # formula_gen = 'dd ~ 1 + inc + vv + dv_ant + dv_pos + brain_M + ' \
    #               '(1 | sn)'
    print('About to lmer...')
    formula_gen = 'dd_vv ~ 1 + inc*all_M + inc*cross + ' \
                  '(1 + inc*all_M + inc*cross | sn)'

    cols = get_formula_cols(df, formula_gen)
    for col in cols:
        if col in ['sn']: continue
        print(f'z-scoring: {col}')
        df[col] = stats.zscore(df[col], nan_policy='omit')
    df_vals = df[cols].dropna()
    from pymer4 import Lmer
    model = Lmer(formula_gen, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    print(df[['dd', 'vv', 'dv_ant', 'dv_pos', 'brain_M']].corr())
    quit()

def vendor_lmer_Feb12(fp='obj7_fMRI'):
    df, vndr_cols = pickle_wrap(None, get_vendor_df, kwargs={'fp': fp,
                                                  'scrub': False,
                                                  'anat': True},
                     cache_dir='cache', easy_override=False)

    p_mem = get_memory_p()
    df_mem = get_df_p_x_p(p_mem, None, 'hc', fp=fp)
    df = df.merge(df_mem, on=['sn', 'obj'], how='left')

    for col in ['age', 'dd_vv', 'cross', 'dd', 'vv',
                'hit_hit', 'hc']: # 'con_hit', 'vis_hit',
        df[col] = stats.zscore(df[col], nan_policy='omit')

    iv = 'hit_hit'
    brain = 'vv'
    dvs = ['dd_vv', 'cross', 'dd', 'vv']
    for dv in dvs:
        formula_gen = f'{dv} ~ 1 + {brain}*{iv} + age*{iv} + ' \
                      f'(1 + {brain}*{iv} + age*{iv} | sn)'
        cols = get_formula_cols(df, formula_gen)
        df_vals = df[cols].dropna()
        from pymer4 import Lmer
        model = Lmer(formula_gen, data=df_vals)
        model.fit(REML=True, verbose=False, summary=False)
        print(model.summary())
        print('-'*10)
        print(f'{dv=}')
        print('-'*50)


if __name__ == '__main__':
    # pass
    # hemi_vendor_corr()
    # meta_corr_triangle()
    # plot_conn_matrix()
    # vendor_lmer()
    vendor_lmer_Feb12()
    # get_trialwise_vendor()
    # plot_meta_corr_matrix()