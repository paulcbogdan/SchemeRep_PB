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
from data_driven_fluc import load_rs, shuffle_remake
from time import time
from tqdm import tqdm
from pingouin import partial_corr
from random import random, shuffle
import itertools

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

CTRL = True
COMBINE_REGIONS = False
print(f'{CTRL=}')

def partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no,
                      ctrl=CTRL):
    if ctrl:
        df = pd.DataFrame({'dd_vv': dd_vv, 'dv_dv': dv_dv,
                           'pd_no': pd_no, 'ad_no': ad_no,
                           'av_no': av_no, 'pv_no': pv_no})
        try:
            r = partial_corr(df, x='dd_vv', y='dv_dv',
                             covar=['pd_no', 'ad_no', 'av_no', 'pv_no'],)
            r = r['r'].values[0]
            return r
        except AssertionError: # NaN
            return np.nan
    else:
        r, p = stats.spearmanr(dd_vv, dv_dv)
        return r

@cache
def get_quads(skip_other=False, all_roi=False, anat_ver=3,
              combine_regions=COMBINE_REGIONS, p_no_override=False):
    # print(combine_regions)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_ver=anat_ver, combine_regions=combine_regions)
    # print(p_d_pos)
    # print(combine_regions)
    # n_roi = 54 if COMBINE_REGIONS else 246
    n_roi = 54 if combine_regions else 246
    p_no = [i for i in range(n_roi) if i not in p_d_ant + p_d_pos +
                                                p_v_ant + p_v_pos]
    # print(f'{len(p_no)=}')
    # quit()
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

    if p_no_override:
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                                  anat_ver=3, combine_regions=combine_regions)
        p_no = [i for i in range(n_roi) if i not in p_d_ant_ + p_d_pos_ +
                p_v_ant_ + p_v_pos_]
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

def get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn, ix=True,
                ctrl=CTRL):
    if ix:
        dd = rs_conn[*np.ix_(pda_i, pdp_i), :]
        dd = np.nanmean(dd, axis=(0, 1))
        vv = rs_conn[*np.ix_(pva_i, pvp_i), :]
        vv = np.nanmean(vv, axis=(0, 1))
        dv_ant = rs_conn[*np.ix_(pda_i, pva_i), :]
        dv_ant = np.nanmean(dv_ant, axis=(0, 1))
        dv_pos = rs_conn[*np.ix_(pdp_i, pvp_i), :]
        dv_pos = np.nanmean(dv_pos, axis=(0, 1))
    else:
        dd = rs_conn[pda_i, pdp_i, :]
        dd = np.nanmean(dd, axis=0)
        vv = rs_conn[pva_i, pvp_i, :]
        vv = np.nanmean(vv, axis=0)
        dv_ant = rs_conn[pda_i, pva_i, :]
        dv_ant = np.nanmean(dv_ant, axis=0)
        dv_pos = rs_conn[pdp_i, pvp_i, :]
        dv_pos = np.nanmean(dv_pos, axis=0)

    dd = stats.zscore(dd)
    vv = stats.zscore(vv)
    dv_ant = stats.zscore(dv_ant)
    dv_pos = stats.zscore(dv_pos)

    dd_vv = dd + vv

    dv_dv = dv_ant + dv_pos
    pd_no = np.nanmean(rs_conn[*np.ix_(pdp_i, p_no), :], axis=(0, 1))
    ad_no = np.nanmean(rs_conn[*np.ix_(pda_i, p_no), :], axis=(0, 1))
    av_no = np.nanmean(rs_conn[*np.ix_(pva_i, p_no), :], axis=(0, 1))
    pv_no = np.nanmean(rs_conn[*np.ix_(pvp_i, p_no), :], axis=(0, 1))
    r = partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no,
                          ctrl=ctrl)

    return r

def get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no, inc_conn, ix=True):
    if ix:
        inc_dd = inc_conn[0, *np.ix_(pda_i, pdp_i)]
        inc_dd = np.nanmean(inc_dd)
        inc_vv = inc_conn[0, *np.ix_(pva_i, pvp_i)]
        inc_vv = np.nanmean(inc_vv)
        inc_dv_ant = inc_conn[0, *np.ix_(pda_i, pva_i)]
        inc_dv_ant = np.nanmean(inc_dv_ant)
        inc_dv_pos = inc_conn[0, *np.ix_(pdp_i, pvp_i)]
        inc_dv_pos = np.nanmean(inc_dv_pos)

        conn_dd = inc_conn[2, *np.ix_(pda_i, pdp_i)]
        conn_dd = np.nanmean(conn_dd)
        conn_vv = inc_conn[2, *np.ix_(pva_i, pvp_i)]
        conn_vv = np.nanmean(conn_vv)
        conn_dv_ant = inc_conn[2, *np.ix_(pda_i, pva_i)]
        conn_dv_ant = np.nanmean(conn_dv_ant)
        conn_dv_pos = inc_conn[2, *np.ix_(pdp_i, pvp_i)]
        conn_dv_pos = np.nanmean(conn_dv_pos)
    else:
        inc_dd = inc_conn[0, pda_i, pdp_i]
        inc_dd = np.nanmean(inc_dd)
        inc_vv = inc_conn[0, pva_i, pvp_i]
        inc_vv = np.nanmean(inc_vv)
        inc_dv_ant = inc_conn[0, pda_i, pva_i]
        inc_dv_ant = np.nanmean(inc_dv_ant)
        inc_dv_pos = inc_conn[0, pdp_i, pvp_i]
        inc_dv_pos = np.nanmean(inc_dv_pos)

        conn_dd = inc_conn[2, pda_i, pdp_i]
        conn_dd = np.nanmean(conn_dd)
        conn_vv = inc_conn[2, pva_i, pvp_i]
        conn_vv = np.nanmean(conn_vv)
        conn_dv_ant = inc_conn[2, pda_i, pva_i]
        conn_dv_ant = np.nanmean(conn_dv_ant)
        conn_dv_pos = inc_conn[2, pdp_i, pvp_i]
        conn_dv_pos = np.nanmean(conn_dv_pos)

    conn_vendor = conn_dd + conn_vv - conn_dv_ant - conn_dv_pos
    inc_vendor = inc_dd + inc_vv - inc_dv_ant - inc_dv_pos
    ef = conn_vendor - inc_vendor
    return ef

def get_task_effect_std(pda_i, pdp_i, pva_i, pvp_i, p_no, inc_conn, ix=True):
    if ix:
        inc_dd = inc_conn[0, *np.ix_(pda_i, pdp_i)]
        inc_dd = np.nanmean(inc_dd, axis=(0, 1))
        inc_vv = inc_conn[0, *np.ix_(pva_i, pvp_i)]
        inc_vv = np.nanmean(inc_vv, axis=(0, 1))
        inc_dv_ant = inc_conn[0, *np.ix_(pda_i, pva_i)]
        inc_dv_ant = np.nanmean(inc_dv_ant, axis=(0, 1))
        inc_dv_pos = inc_conn[0, *np.ix_(pdp_i, pvp_i)]
        inc_dv_pos = np.nanmean(inc_dv_pos, axis=(0, 1))

        conn_dd = inc_conn[2, *np.ix_(pda_i, pdp_i)]
        conn_dd = np.nanmean(conn_dd, axis=(0, 1))
        conn_vv = inc_conn[2, *np.ix_(pva_i, pvp_i)]
        conn_vv = np.nanmean(conn_vv, axis=(0, 1))
        conn_dv_ant = inc_conn[2, *np.ix_(pda_i, pva_i)]
        conn_dv_ant = np.nanmean(conn_dv_ant, axis=(0, 1))
        conn_dv_pos = inc_conn[2, *np.ix_(pdp_i, pvp_i)]
        conn_dv_pos = np.nanmean(conn_dv_pos, axis=(0, 1))
    else:
        inc_dd = inc_conn[0, pda_i, pdp_i]
        inc_dd = np.nanmean(inc_dd, axis=0)
        inc_vv = inc_conn[0, pva_i, pvp_i]
        inc_vv = np.nanmean(inc_vv, axis=0)
        inc_dv_ant = inc_conn[0, pda_i, pva_i]
        inc_dv_ant = np.nanmean(inc_dv_ant, axis=0)
        inc_dv_pos = inc_conn[0, pdp_i, pvp_i]
        inc_dv_pos = np.nanmean(inc_dv_pos, axis=0)

        conn_dd = inc_conn[2, pda_i, pdp_i]
        conn_dd = np.nanmean(conn_dd, axis=0)
        conn_vv = inc_conn[2, pva_i, pvp_i]
        conn_vv = np.nanmean(conn_vv, axis=0)
        conn_dv_ant = inc_conn[2, pda_i, pva_i]
        conn_dv_ant = np.nanmean(conn_dv_ant, axis=0)
        conn_dv_pos = inc_conn[2, pdp_i, pvp_i]
        conn_dv_pos = np.nanmean(conn_dv_pos, axis=0)

    conn_vendor = conn_dd + conn_vv - conn_dv_ant - conn_dv_pos
    inc_vendor = inc_dd + inc_vv - inc_dv_ant - inc_dv_pos
    ef = conn_vendor - inc_vendor

    M1 = np.nanmean(conn_vendor)
    n1 = np.sum(~np.isnan(conn_vendor))
    M2 = np.nanmean(inc_vendor)
    n2 = np.sum(~np.isnan(inc_vendor))
    denom = np.sqrt(1 / n1 + 1 / n2)
    sd = np.sqrt((np.nanvar(conn_vendor) * (n1 - 1) +
                  np.nanvar(inc_vendor) * (n2 - 1)) / (n1 + n2 - 3))
    t = (M1 - M2) / (sd * denom)
    d = t * np.sqrt(1 / n1 + 1 / n2)
    return d, (M1 - M2) / d


@cache
def get_group_inc_conn(n='7'):
    kwargs = {'fp': f'obj{n}_fMRI',
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
def get_group_level_efs(pda_i, pdp_i, pva_i, pvp_i, p_no, ix=True,
                        n='7'):
    inc_conn = get_group_inc_conn(n=n)
    ef = get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no, inc_conn,  ix=ix)
    return ef


@cache
def get_group_rs(pda_i, pdp_i, pva_i, pvp_i, p_no, ix=False):
    conn_trials, sns = load_rs()
    rs = []
    for i in range(conn_trials.shape[0]):
        rs_conn = conn_trials[i]
        r = get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn, ix=ix)
        rs.append(r)
    return np.nanmean(rs)


def do_sn(rs_conn, inc_conn, num_test=2500, ctrl_group=False,
          skip_other=False, all_roi=False, ix=True, anat_ver=5,
          n='XXX'):
    if len(inc_conn.shape) == 4:
        std_d = True
    else:
        std_d = False

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = get_quads(skip_other,
                                                         all_roi=all_roi,
                                                         anat_ver=anat_ver)
    p_no = tuple(p_no)
    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)

    np.random.seed(0)
    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    # print('prod')
    combos = list(combos)
    # print('list')
    skipper = int(num_pos / num_test)
    # print(f'skip: {skipper}')
    # print(f'{len(combos)=}')
    skipper = max(1, skipper)
    combos = combos[::skipper]
    np.random.shuffle(combos)
    # print('shuffled')
    # print(f'{len(combos)=}')
    # print(combos)

    combos_ = []
    for (a, b, c, d) in combos:
        if len({a, b, c, d}) < 4:
            continue
        # if np.random.uniform(0, 1) > num_test / num_pos:
        #     continue
        combos_.append((a, b, c, d))
    combos = combos_
    # print(f'{len(combos)=}')
    # st = time()

    rs = []
    efs = []
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

        pda_i = tuple(pda_i)
        pdp_i = tuple(pdp_i)
        pva_i = tuple(pva_i)
        pvp_i = tuple(pvp_i)

        r = get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn,
                        ix=ix)
        if std_d:
            ef, div = get_task_effect_std(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                     inc_conn, ix=ix)
        else:
            ef = get_task_effect(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                 inc_conn, ix=ix)
        if ctrl_group:
            group_ef = get_group_level_efs(pda_i, pdp_i, pva_i,
                                           pvp_i, p_no, ix=ix,
                                           n=n)
            if std_d:
                group_ef /= div
            group_r = get_group_rs(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                   ix=ix)
            r -= group_r
            ef -= group_ef
        if all_roi:
            ef = np.abs(ef)
        # print(f'{ef=}, {r=}')
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
          f'{ctrl_group=}, {CTRL=}, {all_roi=}, {skip_other=}, {ix=},'
          f'{anat_ver=}, {n=}, {std_d=}')

    plt.scatter(rs, efs, color='red' if CTRL else 'dodgerblue',
                alpha=.25)
    plt.show()
    rho, p = stats.spearmanr(rs, efs)
    print(f'\t{rho=:.6f}, {p=:.4f}')

    # print(time() - st)
    return rho

def get_inc_conn_t(inc_roi_act):
    inc_roi_act = stats.zscore(inc_roi_act, axis=-1, nan_policy='omit')
    inc_roi_act0 = inc_roi_act[:, :, None, :]
    inc_roi_act1 = inc_roi_act[:, None, :, :]
    inc_conn = inc_roi_act0 * inc_roi_act1
    return inc_conn


def do_all_sn(do_t=False, n='7', shuffle_seed=None):
    conn_trials, sns = load_rs(combine_regions=COMBINE_REGIONS)

    assert all(sns[i] <= sns[i+1] for i in range(len(sns) - 1))
    kwargs = {'fp': f'obj{n}_fMRI',
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

    if shuffle_seed is not None:
        print(f'Shuffled: {shuffle_seed}')
        np.random.seed(shuffle_seed)
        if do_t:
            sn_inc_roi_act = shuffle_remake(sn_inc_roi_act, just_rows=True)
        else:
            sn_inc_conn = shuffle_remake(sn_inc_roi_act)


    sns_task = [df_sn['sn'].iloc[0] for df_sn in df_sns]
    assert all(sns_task[i] <= sns_task[i+1] for i in range(len(sns_task) - 1))
    bool_overlap = [sn in sns for sn in sns_task]
    sn_inc_conn = sn_inc_conn[bool_overlap]
    sn_inc_roi_act = sn_inc_roi_act[bool_overlap]

    rhos = []
    t = None
    for i in range(sn_inc_conn.shape[0]):
        if do_t:
            inc_conn = get_inc_conn_t(sn_inc_roi_act[i])
        else:
            inc_conn = sn_inc_conn[i]
        rho = do_sn(conn_trials[i], inc_conn, n=n)
        if np.isnan(rho): continue
        rhos.append(rho)
        t, p = stats.ttest_1samp(rhos, 0)
        if np.isnan(t):
            print('nan t')
            continue
        M = np.mean(rhos)
        print(f'{i}: {M=:.3f}, {t=:.2f}, {p=:.4f}')
    return t

if __name__ == '__main__':
    do_all_sn()

    #     num_pos = 11440: num_test = 5000(
    #         len(efs) = 5022): ctrl_group = False, CTRL = False, all_roi = False, skip_other = True
    #     rho = -0.043849, p = 0.0019
    # 63: M = -0.029, t = -1.79, p = 0.0781_