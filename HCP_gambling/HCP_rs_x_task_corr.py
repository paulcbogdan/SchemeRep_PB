import os

from HCP_gambling.HCP_vendor import get_sn_roi_ar, make_conn, get_combo
from atlas_utils import get_atlas
from networks.sn_anat_fluc import get_quads
import copy

import pandas as pd


from networks.old.network_funcs import load_FC_for_Lifu
from networks.vendor_partitioning import get_vendor_partitions, do_regression
from utils import pickle_wrap, stdize
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, spatial
from functools import cache
from numba import njit, config, jit
from time import time
from tqdm import tqdm
from pingouin import partial_corr
import itertools

from scipy import sparse

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

os.chdir(r'C:\PycharmProjects\SchemeRep')

# @cache
def get_sn_roi_ar_std(**kw):
    ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, RAM_cache=False)
    assert len(ar.shape) == 2
    ar = stats.zscore(ar, axis=1)
    return ar

@cache
def get_sns_roi_ar_std(sns, **kw):
    ars = []
    for sn in sns:
        ar = get_sn_roi_ar_std(sn=sn, **kw)
        if ar.shape[1] < 1200:
            ar = np.pad(ar, ((0, 0), (0, 1200 - ar.shape[1])), 'constant',
                        constant_values=np.nan)
        assert ar.shape[1] == 1200, f'{sn=}, {ar.shape=}'
        ars.append(ar)
    return np.array(ars)

@cache
def get_rs_conn_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos, **kw):
    # ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, RAM_cache=True)
    # t_st = time()
    ar = get_sns_roi_ar_std(tuple(sns), **kw)
    print(f'{ar.shape=}')
    # print(f'RS time: {time() - t_st:.5f} s')
    # rs_conn = np.full((len(sns), ar.shape[1], ar.shape[1], ar.shape[2]), np.nan)
    # n_roi = ar.shape[1]
    #
    # rs_conn[:, *np.ix_(p_d_ant, p_d_pos), :] = (
    #         ar[:, p_d_ant, None, :] * ar[:, None, p_d_pos, :])
    # rs_conn[:, *np.ix_(p_v_ant, p_v_pos), :] = (
    #         ar[:, p_v_ant, None, :] * ar[:, None, p_v_pos, :])
    # rs_conn[:, *np.ix_(p_d_ant, p_v_ant), :] = (
    #         ar[:, p_d_ant, None, :] * ar[:, None, p_v_ant, :])
    # rs_conn[:, *np.ix_(p_d_pos, p_v_pos), :] = (
    #         ar[:, p_d_pos, None, :] * ar[:, None, p_v_pos, :])
    # print(f'RS time: {time() - t_st:.5f} s')

    p_all = list(p_d_ant) + list(p_d_pos) + list(p_v_ant) + list(p_v_pos)
    p_d_ant_new = np.arange(len(p_d_ant))
    p_d_pos_new = np.arange(len(p_d_pos)) + len(p_d_ant)
    p_v_ant_new = np.arange(len(p_v_ant)) + len(p_d_ant) + len(p_d_pos)
    p_v_pos_new = np.arange(len(p_v_pos)) + len(p_d_ant) + len(p_d_pos) + len(p_v_ant)
    # rs_conn_compressed = rs_conn[:, *np.ix_(p_all, p_all), :]
    # print(rs_conn_compressed.shape)
    # quit()

    map2new = {}
    for i, p in enumerate(p_all):
        map2new[p] = i

    rs_conn_compressed = np.full((len(sns), len(p_all),
                                  len(p_all), ar.shape[2]), np.nan)
    rs_conn_compressed[:, *np.ix_(p_d_ant_new, p_d_pos_new), :] = (
            ar[:, p_d_ant, None, :] * ar[:, None, p_d_pos, :])
    rs_conn_compressed[:, *np.ix_(p_v_ant_new, p_v_pos_new), :] = (
            ar[:, p_v_ant, None, :] * ar[:, None, p_v_pos, :])
    rs_conn_compressed[:, *np.ix_(p_d_ant_new, p_v_ant_new), :] = (
            ar[:, p_d_ant, None, :] * ar[:, None, p_v_ant, :])
    rs_conn_compressed[:, *np.ix_(p_d_pos_new, p_v_pos_new), :] = (
            ar[:, p_d_pos, None, :] * ar[:, None, p_v_pos, :])
    # rs_conn_compressed

    return rs_conn_compressed, map2new

def uncompress_conn(conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                     n_roi):
    p_all = list(p_d_ant) + list(p_d_pos) + list(p_v_ant) + list(p_v_pos)
    conn_uncompressed = np.full((conn.shape[0], n_roi, n_roi, conn.shape[3]), np.nan)
    conn_uncompressed[:, *np.ix_(p_all, p_all), :] = conn
    return conn_uncompressed

# def get_rs_conn(p_d_ant, p_d_pos, p_v_ant, p_v_pos, **kw):
#     # ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, RAM_cache=True)
#     ar = get_sn_roi_ar_std(**kw)
#
#     rs_conn = np.full((ar.shape[0], ar.shape[0], ar.shape[1]), np.nan)
#     rs_conn[*np.ix_(p_d_ant, p_d_pos), :] = ar[p_d_ant, None, :] * ar[None, p_d_pos, :]
#     rs_conn[*np.ix_(p_v_ant, p_v_pos), :] = ar[p_v_ant, None, :] * ar[None, p_v_pos, :]
#     rs_conn[*np.ix_(p_d_ant, p_v_ant), :] = ar[p_d_ant, None, :] * ar[None, p_v_ant, :]
#     rs_conn[*np.ix_(p_d_pos, p_v_pos), :] = ar[p_d_pos, None, :] * ar[None, p_v_pos, :]
#     # rs_conn = ar[:, None, :] * ar[None, :, :]
#     return rs_conn

def get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                   all_pda, all_pdp, all_pva, all_pvp, lr='LR',
                   combine_regions=False, bilateral=False,
                   reg_global=True, no_compcor=True):
    kw = {'lr': lr, 'combine_regions': combine_regions,
          'bilateral': bilateral,
          'reg_global': reg_global, 'no_compcor': no_compcor,
          'rs': True}
    t_st = time()
    rs_conn, map2new = get_rs_conn_sns(tuple(sns),
                              tuple(all_pda), tuple(all_pdp),
                              tuple(all_pva), tuple(all_pvp), **kw)
    print(f'Make compressed time: {time() - t_st:.5f} s')
    p_d_ant = [map2new[p] for p in p_d_ant]
    p_d_pos = [map2new[p] for p in p_d_pos]
    p_v_ant = [map2new[p] for p in p_v_ant]
    p_v_pos = [map2new[p] for p in p_v_pos]

    # rs_conn = uncompress_conn(rs_conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
    #                           n_roi)
    # print(f'RS time: {time() - t_st:.5f} s')

    dd = rs_conn[:, *np.ix_(p_d_ant, p_d_pos), :]
    dd = np.nanmean(dd, axis=(1, 2))
    dd_ = stats.zscore(dd, axis=1)
    vv = rs_conn[:, *np.ix_(p_v_ant, p_v_pos), :]
    vv = np.nanmean(vv, axis=(1, 2))
    vv_ = stats.zscore(vv, axis=1)
    dd_vv = dd_ + vv_
    dv_ant = rs_conn[:, *np.ix_(p_d_ant, p_v_ant), :]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_ant_ = stats.zscore(dv_ant, axis=1)
    dv_pos = rs_conn[:, *np.ix_(p_d_pos, p_v_pos), :]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    dv_pos_ = stats.zscore(dv_pos, axis=1)
    dv_dv = dv_ant_ + dv_pos_

    ef = np.nanmean(np.abs(dd_vv - dv_dv), axis=1)
    return ef

    # return dd_vv, dv_dv, dd, vv, dv_ant, dv_pos

# def get_HCP_rs(sn, p_d_ant, p_d_pos, p_v_ant, p_v_pos, lr='LR',
#                combine_regions=False, bilateral=False,
#                reg_global=True, no_compcor=True):
#     # atlas = get_atlas(combine_regions=combine_regions,
#     #                   combine_bilateral=bilateral,
#     #                   HCP=True)
#     kw = {'sn': sn, 'lr': lr, 'combine_regions': combine_regions,
#           'bilateral': bilateral,
#           'reg_global': reg_global, 'no_compcor': no_compcor,
#           'rs': True}
#
#     rs_conn = get_rs_conn(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
#                           **kw)
#
#     # p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
#     #     get_vendor_partitions(age='healthy', anat=True, weighted=False,
#     #                           flip=True, thr=.9, scrub=False, anat_ver=anat_ver,
#     #                           combine_regions=combine_regions)
#
#     dd = rs_conn[*np.ix_(p_d_ant, p_d_pos), :]
#     dd = np.nanmean(dd, axis=(0, 1))
#     dd_ = stats.zscore(dd)
#     vv = rs_conn[*np.ix_(p_v_ant, p_v_pos), :]
#     vv = np.nanmean(vv, axis=(0, 1))
#     vv_ = stats.zscore(vv)
#     dd_vv = dd_ + vv_
#     dv_ant = rs_conn[*np.ix_(p_d_ant, p_v_ant), :]
#     dv_ant = np.nanmean(dv_ant, axis=(0, 1))
#     dv_ant_ = stats.zscore(dv_ant)
#     dv_pos = rs_conn[*np.ix_(p_d_pos, p_v_pos), :]
#     dv_pos = np.nanmean(dv_pos, axis=(0, 1))
#     dv_pos_ = stats.zscore(dv_pos)
#     dv_dv = dv_ant_ + dv_pos_
#
#     return dd_vv, dv_dv, dd, vv, dv_ant, dv_pos
#
# def get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
#                    lr='LR', combine_regions=False, bilateral=False,
#                    reg_global=True, no_compcor=True):
#     dd_vv_l, dv_dv_l, dd_l, vv_l, dv_ant_l, dv_pos_l = [], [], [], [], [], []
#     efs = []
#     for sn in sns:
#         dd_vv, dv_dv, dd, vv, dv_ant, dv_pos = get_HCP_rs(
#             sn, p_d_ant, p_d_pos, p_v_ant, p_v_pos, lr=lr,
#             combine_regions=combine_regions, bilateral=bilateral,
#             reg_global=reg_global, no_compcor=no_compcor)
#         ef = np.nanmean(np.abs(dd_vv - dv_dv))
#         efs.append(ef)
#     ef = np.array(efs)
#     return ef
    # return dd_vv_l, dv_dv_l, dd_l, vv_l, dv_ant_l, dv_pos_l

def get_HCP_task_conn(sns, combine_regions, bilateral,
                      reg_global, no_compcor):
    # kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
    #       'neut_as_PE': None, 'drop_neut': False, 'only': None,
    #       'num_sns': 1000, 'cont_PE': 0.3, 'cont_PE_by_event': True,
    #       'regr_M': True, 'lr_separate': False,
    #       'reg_global': reg_global, 'no_compcor': no_compcor,
    #       'sns_set': list(sns)}

    # kw = {'combine_regions': combine_regions, 'bilateral': bilateral, 'neut_as_PE': None,
    #       'drop_neut': True, 'only': 'combo', 'num_sns': 1000, 'cont_PE': 0.3,
    #       'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True,
    #       'reg_global': reg_global, 'no_compcor': no_compcor, 'median_split': True,
    #       'drop_first': True, 'both_bhv': True, 'reset_trial0': True}

    kw = {'combine_regions': combine_regions, 'bilateral': False, 'neut_as_PE': None,
          'drop_neut': True, 'only': 'combo', 'num_sns': 1000, 'cont_PE': 0.3,
          'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True,
          'reg_global': reg_global, 'no_compcor': no_compcor, 'median_split': True,
          'drop_first': False, 'both_bhv': True, 'reset_trial0': True}
    print(f'{kw=}')

    kw['sns_set'] = list(sns)

    if kw['neut_as_PE']:
        kw['drop_neut'] = False
        kw['only'] = None
        kw['cont_PE'] = None
        kw['cont_PE_by_event'] = False

    if kw['only'] == 'combo':
        conn_highs, conn_lows, sns = get_combo(kw)
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    sn2conns = {}
    for sn, conn_high, conn_low in zip(sns, conn_highs, conn_lows):
        sn2conns[sn] = conn_high, conn_low
    return sn2conns

def get_dd_etc(conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos):

    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))
    itr = dd + vv - dv_ant - dv_pos
    return itr, dd, vv, dv_ant, dv_pos, M_overall

def get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                 combine_regions=False, bilateral=False,
                 reg_global=True, no_compcor=True):
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'sns': sns, 'reg_global': reg_global, 'no_compcor': no_compcor}
    sn2conns = pickle_wrap(get_HCP_task_conn, kwargs=kw, easy_override=True,
                           RAM_cache=True)

    assert set(sns) - set(sn2conns) == set(), f'{set(sns) - set(sn2conns)=}'
    conn_high = [sn2conns[sn][0] for sn in sns]
    conn_high = np.array(conn_high)
    itr_h, dd_h, vv_h, dv_ant_h, dv_pos_h, M_overall_h = (
        get_dd_etc(conn_high, p_d_ant, p_d_pos, p_v_ant, p_v_pos))

    conn_low = [sn2conns[sn][1] for sn in sns]
    conn_low = np.array(conn_low)
    itr_l, dd_l, vv_l, dv_ant_l, dv_pos_l, M_overall_l = (
        get_dd_etc(conn_low, p_d_ant, p_d_pos, p_v_ant, p_v_pos))
    return itr_l - itr_h


def get_HCP_rs_efs():
    pass

def find_overlapping_sns(reg_global_task=True, no_compcor_task=True,
                         reg_global=False, no_compcor=False):

    fns_task = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    if reg_global_task:
        fns_task = [fn for fn in fns_task if 'global' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'global' not in fn]
    if no_compcor_task:
        fns_task = [fn for fn in fns_task if 'nocc' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'nocc' not in fn]
    fns_task_LR = [fn for fn in fns_task if 'LR' in fn]
    fns_task_RL = [fn for fn in fns_task if 'RL' in fn]
    sns_task_LR = {fn.split('_')[0] for fn in fns_task_LR}
    print(f'LR task: N = {len(sns_task_LR)}')
    sns_task_RL = {fn.split('_')[0] for fn in fns_task_RL}
    print(f'RL task: N = {len(sns_task_RL)}')
    sns_task = sns_task_RL.intersection(sns_task_LR)

    fns = os.listdir(r'C:\HCP_RS_clean')
    if reg_global:
        fns = [fn for fn in fns if 'global' in fn]
    else:
        fns = [fn for fn in fns if 'global' not in fn]
    if no_compcor:
        fns = [fn for fn in fns if 'nocc' in fn]
    else:
        fns = [fn for fn in fns if 'nocc' not in fn]
    fns_LR = [fn for fn in fns if 'LR_clean' in fn]
    print(f'C:\ LR RS: N = {len(fns_LR)}')
    fns_RL = [fn for fn in fns if 'RL_clean' in fn]
    print(f'C:\ RL RS: N = {len(fns_RL)}')
    sns_rs_LR = {fn.split('_')[0] for fn in fns_LR}
    sns_rs_RL = {fn.split('_')[0] for fn in fns_RL}
    sns_rs = sns_rs_LR.intersection(sns_rs_RL)
    # sns_overlap = sns_task.intersection(sns_rs)

    fns_E = os.listdir(r'E:\HCP_RS_clean')
    if reg_global:
        fns_E = [fn for fn in fns_E if 'global' in fn]
    else:
        fns_E = [fn for fn in fns_E if 'global' not in fn]
    if no_compcor:
        fns_E = [fn for fn in fns_E if 'nocc' in fn]
    else:
        fns_E = [fn for fn in fns_E if 'nocc' not in fn]
    fns_LR_E = [fn for fn in fns_E if 'LR_clean' in fn]
    print(f'E:\ LR RS: N = {len(fns_LR_E)}')
    fns_RL_E = [fn for fn in fns_E if 'RL_clean' in fn]
    print(f'E:\ RL RS: N = {len(fns_RL_E)}')
    sns_rs_LR_E = {fn.split('_')[0] for fn in fns_LR_E}
    sns_rs_RL_E = {fn.split('_')[0] for fn in fns_RL_E}
    sns_rs_E = sns_rs_LR_E.intersection(sns_rs_RL_E)

    sns_rs.update(sns_rs_E)
    print(f'Overall RS: N = {len(sns_rs)}')

    sns_overlap = sns_task.intersection(sns_rs)

    bad_sns = {'263436'}
    sns_overlap = sorted(list(sns_overlap - bad_sns))

    print(f'Number of overlapping non-bad sns: {len(sns_overlap)}')
    return sns_overlap

def do_analysis(num_test=1_000, ctrl_group=False,
                skip_other=True, all_roi=False, ix=True, anat_ver=3,
                combine_regions=False, n='7', std_d=True,
                shuffle_seed=None):

    task_reg_global = True
    task_no_compcor = True
    rs_reg_global = False
    rs_no_compcor = False

    sns = find_overlapping_sns()

    sns = sorted(sns)[:1000]
    print(f'{len(sns)=}')
    print(f'{skip_other=}')

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = get_quads(skip_other,
                                                         all_roi=all_roi,
                                                         anat_ver=anat_ver,
                                                         combine_regions=combine_regions)
    p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all, p_no_all = (
        get_quads(False, all_roi=all_roi, anat_ver=anat_ver,
                  combine_regions=combine_regions))

    print(f'{p_v_pos=}')
    # rs_efs_all = get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
    #                         lr='LR', combine_regions=combine_regions, bilateral=False,
    #                         reg_global=True, no_compcor=True, anat_ver=3)
    # task_efs_all = get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
    #                         combine_regions=combine_regions, bilateral=False)
    # r, p = stats.spearmanr(rs_efs_all, task_efs_all, nan_policy='omit')
    # print(f'Overall: {r=:.2f}, {p=:.2f}')


    p_no = tuple(p_no)
    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'{num_pos=}')

    np.random.seed(0)
    print('Itertools...')
    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    combos = list(combos)
    np.random.shuffle(combos)

    combos_ = []
    for (a, b, c, d) in combos:
        if len(combos_) >= num_test:
            continue
        if len({a, b, c, d}) < 4:
            continue
        combos_.append((a, b, c, d))
    combos = combos_

    # print('Running...')

    task_x_rs = []
    efs = []
    rs_efs_l = []
    task_efs_l = []
    for (a, b, c, d) in tqdm(combos):
        if skip_other:
            pda_i = [a, a + 1]
            pdp_i = [b, b + 1]
            pva_i = [c, c + 1]
            pvp_i = [d, d + 1]
        else:
            pda_i = [a]
            pdp_i = [b]
            pva_i = [c]
            pvp_i = [d]

        t_st = time()
        rs_efs = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                lr='LR', combine_regions=combine_regions, bilateral=False,
                                reg_global=rs_reg_global, no_compcor=rs_no_compcor)
        rs_efs2 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                lr='RL', combine_regions=combine_regions, bilateral=False,
                                reg_global=rs_reg_global, no_compcor=rs_no_compcor)
        # print(f'{len(rs_efs)=}')
        # print(f'{len(rs_efs2)=}')
        rs_efs = np.nanmean([rs_efs, rs_efs2], axis=0)
        # print(rs_efs.shape)
        # quit()
        print(f'Resting time: {time() - t_st:.5f} s')
        # combined regions within-subject effects depend on reg_global?

        t_st = time()
        task_efs = get_HCP_task(sns, pda_i, pdp_i, pva_i, pvp_i,
                                combine_regions=combine_regions, bilateral=False,
                                reg_global=task_reg_global, no_compcor=task_no_compcor)
        # print(f'{len(task_efs)=}')
        print(f'\tTask time: {time() - t_st:.5f} s')
        # for rs_ef, sn in zip(rs_efs, sns):
        #     print(f'{rs_ef}: {sn}')
        # print(rs_efs)
        # print(task_efs)
        num_nans = np.sum(np.isnan(rs_efs))
        assert num_nans == 0
        # r, p = stats.spearmanr(rs_efs, task_efs)
        # task_x_rs.append(r)
        # if len(task_x_rs) > 1:
        #     t, p = stats.ttest_1samp(task_x_rs, 0)
        #     N = len(task_x_rs)
        #     print(f't[{N-1}] = {t:.2f}, {p=:.3f}')

        task_efs_l.append(task_efs)
        rs_efs_l.append(rs_efs)

        test_corrs(task_efs_l, rs_efs_l)

def test_corrs(task_efs_l, rs_efs_l):
    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(np.array(task_efs_l).T,
                                               np.array(rs_efs_l).T)):
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)
    N = len(t_l)
    M_r = np.nanmean(t_l)
    t, p = stats.ttest_1samp(t_l, 0)
    print(f'Within-subj: Mean r = {M_r:.3f}, '
          f't[{N - 1}] = {t:.2f}, {p=:.3f}')

    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(task_efs_l, rs_efs_l)):
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)
    N = len(t_l)
    M_r = np.nanmean(t_l)
    t, p = stats.ttest_1samp(t_l, 0)
    print(f'Across-subject: Mean r = {M_r:.3f}, '
          f't[{N - 1}] = {t:.2f}, {p=:.3f}')

if __name__ == '__main__':
    do_analysis()
