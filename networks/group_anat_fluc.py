import copy
import itertools

import pandas as pd

from atlas_utils import get_atlas
from load_more import load_a
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import partial_corr_df
from sn_anat_fluc import get_rs_fluc, get_quads
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

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

COMBINE_REGIONS = True
CTRL = True

@cache
def get_sn_inc_conn_cache(combine_regions=False, n='7', shuffle_seed=None):
    conn_trials, sns = load_rs(combine_regions=combine_regions)

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
        np.random.seed(shuffle_seed)
        sn_inc_conn = shuffle_remake(sn_inc_roi_act)


    sns_task = [df_sn['sn'].iloc[0] for df_sn in df_sns]
    assert all(sns_task[i] <= sns_task[i+1] for i in range(len(sns_task) - 1))
    bool_overlap = [sn in sns for sn in sns_task]
    sn_inc_conn = sn_inc_conn[bool_overlap]
    return conn_trials, sn_inc_conn

@cache
def get_group_avg_rs_r(pda_i, pdp_i, pva_i, pvp_i, p_no, ix=True,
                       ctrl=True, combine_regions=False, n='7',
                       ):
    conn_trials, _ = get_sn_inc_conn_cache(combine_regions=combine_regions,
                                           n=n)
    corrs = []
    for i in range(conn_trials.shape[0]):
        rs_conn = conn_trials[i]

        r = get_rs_fluc(pda_i, pdp_i, pva_i, pvp_i, p_no, rs_conn,
                        ix=ix, ctrl=ctrl)

        corrs.append(r)

    return np.nanmean(corrs)

def get_group_level_d(pda_i, pdp_i, pva_i, pvp_i, p_no, ix=True,
                      combine_regions=False, n='7', std_d=False,
                      shuffle_seed=None):

    _, sn_inc_conn = get_sn_inc_conn_cache(combine_regions=combine_regions,
                                           n=n, shuffle_seed=shuffle_seed)
    pdp_i = list(pdp_i)
    pda_i = list(pda_i)
    pvp_i = list(pvp_i)
    pva_i = list(pva_i)
    p_no = list(p_no)

    if ix:
        inc_dd = sn_inc_conn[:, 0, *np.ix_(pda_i, pdp_i)]
        inc_dd = np.nanmean(inc_dd, axis=(1, 2))
        inc_vv = sn_inc_conn[:, 0, *np.ix_(pva_i, pvp_i)]
        inc_vv = np.nanmean(inc_vv, axis=(1, 2))
        inc_dv_ant = sn_inc_conn[:, 0, *np.ix_(pda_i, pva_i)]
        inc_dv_ant = np.nanmean(inc_dv_ant, axis=(1, 2))
        inc_dv_pos = sn_inc_conn[:, 0, *np.ix_(pdp_i, pvp_i)]
        inc_dv_pos = np.nanmean(inc_dv_pos, axis=(1, 2))

        conn_dd = sn_inc_conn[:, 2, *np.ix_(pda_i, pdp_i)]
        conn_dd = np.nanmean(conn_dd, axis=(1, 2))
        conn_vv = sn_inc_conn[:, 2, *np.ix_(pva_i, pvp_i)]
        conn_vv = np.nanmean(conn_vv, axis=(1, 2))
        conn_dv_ant = sn_inc_conn[:, 2, *np.ix_(pda_i, pva_i)]
        conn_dv_ant = np.nanmean(conn_dv_ant, axis=(1, 2))
        conn_dv_pos = sn_inc_conn[:, 2, *np.ix_(pdp_i, pvp_i)]
        conn_dv_pos = np.nanmean(conn_dv_pos, axis=(1, 2))
    else:
        inc_dd = sn_inc_conn[:, 0, pda_i, pdp_i]
        inc_dd = np.nanmean(inc_dd, axis=1)
        inc_vv = sn_inc_conn[:, 0, pva_i, pvp_i]
        inc_vv = np.nanmean(inc_vv, axis=1)
        inc_dv_ant = sn_inc_conn[:, 0, pda_i, pva_i]
        inc_dv_ant = np.nanmean(inc_dv_ant, axis=1)
        inc_dv_pos = sn_inc_conn[:, 0, pdp_i, pvp_i]
        inc_dv_pos = np.nanmean(inc_dv_pos, axis=1)

        conn_dd = sn_inc_conn[:, 2, pda_i, pdp_i]
        conn_dd = np.nanmean(conn_dd, axis=1)
        conn_vv = sn_inc_conn[:, 2, pva_i, pvp_i]
        conn_vv = np.nanmean(conn_vv, axis=1)
        conn_dv_ant = sn_inc_conn[:, 2, pda_i, pva_i]
        conn_dv_ant = np.nanmean(conn_dv_ant, axis=1)
        conn_dv_pos = sn_inc_conn[:, 2, pdp_i, pvp_i]
        conn_dv_pos = np.nanmean(conn_dv_pos, axis=1)

    conn_vendor = conn_dd + conn_vv - conn_dv_ant - conn_dv_pos
    inc_vendor = inc_dd + inc_vv - inc_dv_ant - inc_dv_pos
    ef = conn_vendor - inc_vendor
    if std_d:
        return np.nanmean(ef) / np.nanstd(ef)
    else:
        return np.nanmean(ef)# / np.nanstd(ef)

def get_dists(al, bl, cl, dl, combine_regions=True):
    atlas = get_atlas(combine_regions=combine_regions)
    all_totals = []
    for a in al:
        coord_a = atlas['coords'][a]
        for b in bl:
            coord_b = atlas['coords'][b]
            ab_dist = spatial.distance.euclidean(coord_a, coord_b)
            for c in cl:
                coord_c = atlas['coords'][c]
                ac_dist = spatial.distance.euclidean(coord_a, coord_c)
                bc_dist = spatial.distance.euclidean(coord_b, coord_c)
                for d in dl:
                    coord_d = atlas['coords'][d]
                    ad_dist = spatial.distance.euclidean(coord_a, coord_d)
                    bd_dist = spatial.distance.euclidean(coord_b, coord_d)
                    cd_dist = spatial.distance.euclidean(coord_c, coord_d)
                    total_dist = (ab_dist + ac_dist + bc_dist + ad_dist +
                                  bd_dist + cd_dist)
                    all_totals.append(total_dist)
    all_total = np.mean(all_totals)
    return all_total
    # print(f'{all_total=}')
    # quit()
                    # print(f'{ab_dist=:.2f}, {ac_dist=:.2f}, {bc_dist=:.2f}, '
                    #       f'{ad_dist=:.2f}, {bd_dist=:.2f}, {cd_dist=:.2f}')



def do_group(num_test=10_000, ctrl_group=False,
             skip_other=True, all_roi=True, ix=True, anat_ver=5,
             combine_regions=True, n='8', std_d=True,
             shuffle_seed=None):

    p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, p_no_ = (
        get_quads(False, all_roi=False, anat_ver=3,
                  combine_regions=combine_regions))
    # M all: og_r, og_d = -.18, -.11
    p_d_ant_ = tuple(p_d_ant_)
    p_d_pos_ = tuple(p_d_pos_)
    p_v_ant_ = tuple(p_v_ant_)
    p_v_pos_ = tuple(p_v_pos_)

    get_dists(p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_,)

    p_no_ = tuple(p_no_)
    purp_combos = itertools.product(p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_)
    purp_combos = list(purp_combos)
    og_r = get_group_avg_rs_r(p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, p_no_,
                              ix=ix, combine_regions=combine_regions, n=n)
    og_d = get_group_level_d(p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, p_no_,
                             ix=ix, combine_regions=combine_regions,
                             n=n, std_d=std_d, shuffle_seed=shuffle_seed)

    print(f'OG all: {og_r=:.2f}, {og_d=:.2f}\n')

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other, all_roi=all_roi, anat_ver=anat_ver,
                  combine_regions=combine_regions, p_no_override=True))


    p_no = tuple(p_no)
    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'{num_pos=}')

    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    combos = list(combos)
    shuffle(combos)

    combos_ = []
    for (a, b, c, d) in combos:
        if random() > num_test / num_pos:
            continue
        combos_.append((a, b, c, d))
    #     print(f'{dist=}')
    # quit()
    combos = combos_
    print(f'{len(combos)=}')
    print(f'{len(purp_combos)=}')
    combos = purp_combos + combos


    rs = []
    efs = []
    dists = []
    colors = []
    for (a, b, c, d) in combos:
        # if random() > num_test / num_pos:
        #     continue
        if len({a, b, c, d}) < 4:
            continue
        dist = get_dists([a], [b], [c], [d], combine_regions=combine_regions)
        dists.append(dist)

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

        if (a in p_d_ant_ and b in p_d_pos_ and c in p_v_ant_ and
                d in p_v_pos_):
            color = 'purple'
        else:
            color = 'green'


        pda_i = tuple(pda_i)
        pdp_i = tuple(pdp_i)
        pva_i = tuple(pva_i)
        pvp_i = tuple(pvp_i)

        r = get_group_avg_rs_r(pda_i, pdp_i, pva_i, pvp_i, p_no,
                               ix=ix, n=n)
        d = get_group_level_d(pda_i, pdp_i, pva_i, pvp_i, p_no,
                              ix=ix, n=n, std_d=std_d,
                              shuffle_seed=shuffle_seed)
        rs.append(r)
        efs.append(d)
        colors.append(color)

        if len(efs) % 200 == 0:
            plot_rs_efs(rs, efs, colors, og_d, og_r, dists,
                        combine_regions=combine_regions,
                        all_roi=all_roi, n=n, std_d=std_d)

            if all_roi:
                plot_rs_efs(rs, np.abs(efs), colors, og_d, og_r, dists,
                            combine_regions=combine_regions,
                            all_roi=all_roi, n=n + f'seed: {shuffle_seed}',
                            std_d=std_d)

    print(f'{num_pos=}: {num_test=} ({len(efs)=}): '
          f'{ctrl_group=}, {CTRL=}, {all_roi=}, {skip_other=}, {ix=},'
          f'{anat_ver=}')

    plot_rs_efs(rs, efs, colors, og_d, og_r, dists,
                combine_regions=combine_regions,
                all_roi=all_roi, std_d=std_d,
                n=n + f'seed: {shuffle_seed}')
    rho, p = stats.spearmanr(rs, efs)
    print(f'\t{rho=:.6f}, {p=:.4f}')
    return rho

def plot_rs_efs(rs, efs, colors, og_d, og_r, dists,
                combine_regions=False, all_roi=False, n='7', std_d=True):
    rho, p = stats.spearmanr(rs, efs)
    fig, axs = plt.subplots(3, 1, figsize=(5, 9))
    plt.sca(axs[0])

    cmap = plt.get_cmap('turbo')
    max_dist = max(dists) / 1.2
    min_dist = min(dists)

    def get_color(d, c):
        if c == 'purple':
            return c
        else:
            return cmap((d - min_dist) / (max_dist - min_dist))

    # colors = [cmap((d - min_dist) / (max_dist - min_dist)) for d in dists]
    colors = [get_color(d, c) for d, c in zip(dists, colors)]


    plt.title(f'Number: {len(rs)}, {rho=:.3f}, {p=:.4f}\n{combine_regions=},'
              f' {all_roi=}, {n=}, {std_d=}')
    alphas = [.75 if color == 'purple' else .25 for color in colors]

    plt.scatter(rs, efs, color=colors,
                alpha=alphas)
    plt.scatter([og_r], [og_d], color='red', alpha=.9, marker='s')
    plt.xlabel('Resting correlation')
    plt.ylabel('Task effect')

    # low = min(rs)
    # high = max(rs)
    # rs_ = [r for r, c in zip(rs, colors) if c != 'purple']
    plt.sca(axs[1])
    r, p = stats.spearmanr(dists, efs)
    plt.title(f'Dist x ef: {r=:.3f}, {p=:.4f}')
    plt.scatter(dists, efs, color=colors,
                alpha=alphas)
    # plt.hist(rs_, bins=100, cumulative=True, range=(low, high))
    plt.xlabel('dists')
    plt.ylabel('Task effect')

    plt.sca(axs[2])
    r, p = stats.spearmanr(dists, rs)
    plt.title(f'Dist x corr: {r=:.3f}, {p=:.4f}')
    plt.scatter(dists, rs, color=colors,
                alpha=alphas)
    # plt.hist(rs_, bins=100, cumulative=True, range=(low, high))
    plt.xlabel('dists')
    plt.ylabel('Resting correlation')
    plt.tight_layout()
    plt.show()



def shuffle_test():
    for seed in range(100):
        rho = do_group(shuffle_seed=seed)

def simple_shuffle_og():
    p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, p_no_ = (
        get_quads(False, all_roi=False, anat_ver=3,
                  combine_regions=combine_regions))
    # M all: og_r, og_d = -.18, -.11
    p_d_ant_ = tuple(p_d_ant_)
    p_d_pos_ = tuple(p_d_pos_)
    p_v_ant_ = tuple(p_v_ant_)
    p_v_pos_ = tuple(p_v_pos_)
    og_d = get_group_level_d(p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, p_no_,
                             ix=ix, combine_regions=combine_regions,
                             n=n, std_d=std_d, shuffle_seed=shuffle_seed)

    print(f'OG all: {og_r=:.2f}, {og_d=:.2f}\n')

if __name__ == '__main__':
    do_group()
    shuffle_test()

