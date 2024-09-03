import copy

import pandas as pd

from atlas_utils import get_atlas
from load_more import load_a
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import partial_corr_df
from utils import pickle_wrap, stdize
from vendor_partitioning import do_regression, get_vendor_partitions
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy import stats, spatial
from functools import cache
from numba import njit, config, jit
from time import time
from data_driven_fluc import load_rs
from time import time
from tqdm import tqdm
from pingouin import partial_corr
from random import random

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)


#
CTRL = False
COMBINE_REGIONS = False
print(f'{CTRL=}')

def partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no,
                      ctrl=CTRL):
    if ctrl:
        df = pd.DataFrame({'dd_vv': dd_vv, 'dv_dv': dv_dv,
                           'pd_no': pd_no, 'ad_no': ad_no,
                           'av_no': av_no, 'pv_no': pv_no})
        r = partial_corr(df, x='dd_vv', y='dv_dv',
                         covar=['pd_no', 'ad_no', 'av_no', 'pv_no'],)
        r = r['r'].values[0]
        return r
    else:
        r, p = stats.spearmanr(dd_vv, dv_dv)
        return r

@cache
def get_quads(skip_other=False, all_roi=False, anat_ver=5):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_ver=anat_ver,
                              combine_regions=COMBINE_REGIONS)
    # n_roi = 54 if COMBINE_REGIONS else 246
    print(f'{anat_ver=}')
    n_roi = 48 if COMBINE_REGIONS else 246
    p_no = [i for i in range(n_roi) if i not in p_d_ant + p_d_pos +
                                              p_v_ant + p_v_pos]

    if all_roi:
        p_d_ant = list(range(n_roi))
        p_d_pos = list(range(n_roi))
        p_v_ant = list(range(n_roi))
        p_v_pos = list(range(n_roi))
    if skip_other:
        p_d_ant = p_d_ant[::2]
        p_d_pos = p_d_pos[::2]
        p_v_ant = p_v_ant[::2]
        p_v_pos = p_v_pos[::2]
    # p_no = []
    return p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no

def get_rs_fluc_OLD(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn):
    dd = rs_conn[pda_i, pdp_i, :]
    vv = rs_conn[pva_i, pvp_i, :]
    dd_vv = dd + vv
    dv_ant = rs_conn[pda_i, pva_i, :]
    dv_pos = rs_conn[pdp_i, pvp_i, :]
    dv_dv = dv_ant + dv_pos

    pd_no = np.nanmean(rs_conn[pda_i, p_no, :], axis=0)
    ad_no = np.nanmean(rs_conn[pdp_i, p_no, :], axis=0)
    av_no = np.nanmean(rs_conn[pva_i, p_no, :], axis=0)
    pv_no = np.nanmean(rs_conn[pvp_i, p_no, :], axis=0)
    r = partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no)
    return r

def get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn):
    # dd = rs_conn[*np.ix_(pda_i, pdp_i), :]
    dd = rs_conn[pda_i, pdp_i, :]
    # dd = np.nanmean(dd, axis=(0, 1))
    dd = np.nanmean(dd, axis=0)
    # vv = rs_conn[*np.ix_(pva_i, pvp_i), :]
    vv = rs_conn[pva_i, pvp_i, :]
    # vv = np.nanmean(vv, axis=(0, 1))
    vv = np.nanmean(vv, axis=0)
    dd_vv = dd + vv

    # dv_ant = rs_conn[*np.ix_(pda_i, pva_i), :]
    dv_ant = rs_conn[pda_i, pva_i, :]
    # dv_ant = np.nanmean(dv_ant, axis=(0, 1))
    dv_ant = np.nanmean(dv_ant, axis=0)

    # dv_pos = rs_conn[*np.ix_(pdp_i, pvp_i), :]
    dv_pos = rs_conn[pdp_i, pvp_i, :]
    # dv_pos = np.nanmean(dv_pos, axis=(0, 1))
    dv_pos = np.nanmean(dv_pos, axis=0)
    dv_dv = dv_ant + dv_pos


    pd_no = np.nanmean(rs_conn[*np.ix_(pda_i, p_no), :], axis=(0, 1))
    ad_no = np.nanmean(rs_conn[*np.ix_(pdp_i, p_no), :], axis=(0, 1))
    av_no = np.nanmean(rs_conn[*np.ix_(pva_i, p_no), :], axis=(0, 1))
    pv_no = np.nanmean(rs_conn[*np.ix_(pvp_i, p_no), :], axis=(0, 1))
    r = partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no)
    return r

def get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no, inc_conn):
    # inc_dd = inc_conn[0, *np.ix_(pda_i, pdp_i)]
    inc_dd = inc_conn[0, pda_i, pdp_i]
    inc_dd = np.nanmean(inc_dd)
    # inc_vv = inc_conn[0, *np.ix_(pva_i, pvp_i)]
    inc_vv = inc_conn[0, pva_i, pvp_i]
    inc_vv = np.nanmean(inc_vv)
    # inc_dv_ant = inc_conn[0, *np.ix_(pda_i, pva_i)]
    inc_dv_ant = inc_conn[0, pda_i, pva_i]
    inc_dv_ant = np.nanmean(inc_dv_ant)
    # inc_dv_pos = inc_conn[0, *np.ix_(pdp_i, _pvp_i)]
    inc_dv_pos = inc_conn[0, pdp_i, pvp_i]
    inc_dv_pos = np.nanmean(inc_dv_pos)

    # conn_dd = inc_conn[2, *np.ix_(pda_i, pdp_i)]
    conn_dd = inc_conn[2, pda_i, pdp_i]
    conn_dd = np.nanmean(conn_dd)
    # conn_vv = inc_conn[2, *np.ix_(pva_i, pvp_i)]
    conn_vv = inc_conn[2, pva_i, pvp_i]
    conn_vv = np.nanmean(conn_vv)
    # conn_dv_ant = inc_conn[2, *np.ix_(pda_i, pva_i)]
    conn_dv_ant = inc_conn[2, pda_i, pva_i]
    conn_dv_ant = np.nanmean(conn_dv_ant)
    # conn_dv_pos = inc_conn[2, *np.ix_(pdp_i, pvp_i)]
    conn_dv_pos = inc_conn[2, pdp_i, pvp_i]
    conn_dv_pos = np.nanmean(conn_dv_pos)

    conn_vendor = conn_dd + conn_vv - conn_dv_ant - conn_dv_pos
    inc_vendor = inc_dd + inc_vv - inc_dv_ant - inc_dv_pos
    ef = conn_vendor - inc_vendor
    return ef

@cache
def get_group_inc_conn():
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': COMBINE_REGIONS,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_roi_act, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=-1, cache_dir='cache',
                    RAM_cache=True)
    inc_conn = np.nanmean(sn_inc_conn, axis=0)
    return inc_conn

@cache
def get_group_level_efs(pda_i, pdp_i, pva_i, pvp_i, p_no):
    inc_conn = get_group_inc_conn()
    ef = get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no, inc_conn)
    return ef


# get_group_level_efs((1, 2), (3, 4), (5, 6), (7, 8), (9, 10, 11))


@cache
def get_group_rs(pda_i, pdp_i, pva_i, pvp_i, p_no):
    conn_trials, sns = load_rs()
    rs = []
    for i in range(conn_trials.shape[0]):
        rs_conn = conn_trials[i]
        r = get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn)
        rs.append(r)
    return np.nanmean(rs)


def do_sn(rs_conn, inc_conn, num_test=25_000, ctrl_group=False,
          skip_other=False, all_roi=False):
    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = get_quads(skip_other,
                                                         all_roi=all_roi)
    p_no = tuple(p_no)


    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)

    st = time()

    rs = []
    efs = []
    for a in tqdm(p_d_ant):
        for b in p_d_pos:
            for c in p_v_ant:
                for d in p_v_pos:

                    if random() > num_test / num_pos:
                        continue

                    if len({a, b, c, d}) < 4:
                        continue

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

                    pda_i = tuple(pda_i)
                    pdp_i = tuple(pdp_i)
                    pva_i = tuple(pva_i)
                    pvp_i = tuple(pvp_i)


                    r = get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn)
                    ef = get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                         inc_conn)

                    if ctrl_group:
                        group_ef = get_group_level_efs(pda_i, pdp_i, pva_i,
                                                       pvp_i, p_no)
                        group_r = get_group_rs(pda_i, pdp_i, pva_i, pvp_i, p_no)
                        r -= group_r
                        ef -= group_ef
                    if np.isnan(ef):
                        continue
                    rs.append(r)
                    efs.append(ef)
    num_nans = sum(np.isnan(rs))
    # print(f'{num_nans=}')
    # print(f'{len(rs)=}')
    num_nans_efs = sum(np.isnan(efs))
    # print(f'{num_nans_efs=}')

    print(f'{num_pos=}: {num_test=} ({len(efs)=}): '
          f'{ctrl_group=}, {CTRL=}, {all_roi=}, {skip_other=}')

    plt.scatter(rs, efs, color='red' if CTRL else 'dodgerblue',
                alpha=.25)
    plt.show()
    rho, p = stats.spearmanr(rs, efs)
    print(f'\t{rho=:.6f}, {p=:.4f}')

    # print(time() - st)
    return rho




def do_all_sn():
    conn_trials, sns = load_rs(combine_regions=COMBINE_REGIONS)

    assert all(sns[i] <= sns[i+1] for i in range(len(sns) - 1))
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': False,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_roi_act, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=-1, cache_dir='cache',
                    RAM_cache=True)
    sns_task = [df_sn['sn'].iloc[0] for df_sn in df_sns]
    assert all(sns_task[i] <= sns_task[i+1] for i in range(len(sns_task) - 1))
    bool_overlap = [sn in sns for sn in sns_task]
    sn_inc_conn = sn_inc_conn[bool_overlap]

    rhos = []
    for i in range(sn_inc_conn.shape[0]):
        rho = do_sn(conn_trials[i], sn_inc_conn[i])
        if np.isnan(rho): continue
        rhos.append(rho)
        t, p = stats.ttest_1samp(rhos, 0)
        if np.isnan(t):
            print('nan t')
            continue
        M = np.mean(rhos)
        print(f'{i}: {M=:.3f}, {t=:.2f}, {p=:.4f}')

if __name__ == '__main__':
    do_all_sn()

    #     num_pos = 11440: num_test = 5000(
    #         len(efs) = 5022): ctrl_group = False, CTRL = False, all_roi = False, skip_other = True
    #     rho = -0.043849, p = 0.0019
    # 63: M = -0.029, t = -1.79, p = 0.0781