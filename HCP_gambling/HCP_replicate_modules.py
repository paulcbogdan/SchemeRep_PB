import os

from networks.old.modularity import get_partition_matrix
from utils import pickle_wrap

os.chdir(r'C:\PycharmProjects\SchemeRep')

from HCP_gambling.HCP_rs_x_task_corr import get_sn_roi_ar_std, find_overlapping_sns
from networks.sn_anat_fluc import get_quads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg
from old.plot_gen import plot_connectivity

def partial_corr_df_2(df, cols, cov, verbose=0):
    cols = [col for col in cols if col not in cov]
    ar = np.full((len(cols), len(cols)), np.nan, dtype=float)
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

def get_modules_2(matrix_thresh):
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(matrix_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=0)
    return part

def get_corr_matrix(rs, pda, pdp, pva, pvp, p_no, no_ctrl=True):
    cols_order = ['Lda_Ldp', 'Rda_Rdp',
                  'Lva_Lvp', 'Rva_Rvp',
                  'Ldp_Lvp', 'Rdp_Rvp',
                  'Lda_Lva', 'Rda_Rva',
                  'Ldp_Rdp', 'Lvp_Rvp',
                  'Lda_Rda', 'Lva_Rva', ]

    lda_ldp = rs[*np.ix_(pda[::2], pdp[::2]), :].mean(axis=(0, 1))
    rda_rdp = rs[*np.ix_(pda[1::2], pdp[1::2]), :].mean(axis=(0, 1))
    lva_lvp = rs[*np.ix_(pva[::2], pvp[::2]), :].mean(axis=(0, 1))
    rva_rvp = rs[*np.ix_(pva[1::2], pvp[1::2]), :].mean(axis=(0, 1))

    ldp_lvp = rs[*np.ix_(pdp[::2], pvp[::2]), :].mean(axis=(0, 1))
    rdp_rvp = rs[*np.ix_(pdp[1::2], pvp[1::2]), :].mean(axis=(0, 1))
    lda_lva = rs[*np.ix_(pda[::2], pva[::2]), :].mean(axis=(0, 1))
    rda_rva = rs[*np.ix_(pda[1::2], pva[1::2]), :].mean(axis=(0, 1))

    ldp_rdp = rs[*np.ix_(pdp[::2], pdp[1::2]), :].mean(axis=(0, 1))
    lvp_rvp = rs[*np.ix_(pvp[::2], pvp[1::2]), :].mean(axis=(0, 1))
    lda_rda = rs[*np.ix_(pda[::2], pda[1::2]), :].mean(axis=(0, 1))
    lva_rva = rs[*np.ix_(pva[::2], pva[1::2]), :].mean(axis=(0, 1))

    cols_ctrl = ['pd_no_L', 'ad_no_L', 'av_no_L', 'pv_no_L',
                 'pd_no_R', 'ad_no_R', 'av_no_R', 'pv_no_R',]

    pd_no_L = rs[*np.ix_(pdp[::2], p_no), :].mean(axis=(0, 1))
    pv_no_L = rs[*np.ix_(pvp[::2], p_no), :].mean(axis=(0, 1))
    ad_no_L = rs[*np.ix_(pda[::2], p_no), :].mean(axis=(0, 1))
    av_no_L = rs[*np.ix_(pva[::2], p_no), :].mean(axis=(0, 1))

    pd_no_R = rs[*np.ix_(pdp[1::2], p_no), :].mean(axis=(0, 1))
    pv_no_R = rs[*np.ix_(pvp[1::2], p_no), :].mean(axis=(0, 1))
    ad_no_R = rs[*np.ix_(pda[1::2], p_no), :].mean(axis=(0, 1))
    av_no_R = rs[*np.ix_(pva[1::2], p_no), :].mean(axis=(0, 1))

    cols = cols_order + cols_ctrl

    df = pd.DataFrame(np.array([lda_ldp, rda_rdp, lva_lvp, rva_rvp,
                                ldp_lvp, rdp_rvp, lda_lva, rda_rva,
                                ldp_rdp, lvp_rvp, lda_rda, lva_rva,
                                pd_no_L, pv_no_L, ad_no_L, av_no_L,
                                pd_no_R, pv_no_R, ad_no_R, av_no_R,
                                ]).T, columns=cols)

    if no_ctrl:
        cov = []
    else:
        cov = ['pd_no_L', 'ad_no_L', 'av_no_L', 'pv_no_L',
               'pd_no_R', 'ad_no_R', 'av_no_R', 'pv_no_R',]
    corr = partial_corr_df_2(df.copy(), cols_order,
                             cov=cov)
    return corr, cols_order

def analyze_rs_sn(sn, lr, pda, pdp, pva, pvp, p_no,
                  no_ctrl=False):
    kw = {'lr': lr, 'combine_regions': False,
          'bilateral': False, 'reg_global': False,
          'no_compcor': False, 'rs': True,}

    ar = get_sn_roi_ar_std(sn=sn, **kw)
    rs = ar[:, None, :] * ar[None, :, :]
    mat, cols_order = get_corr_matrix(rs, pda, pdp, pva, pvp, p_no,
                                      no_ctrl=no_ctrl)
    return mat, cols_order



def replicate_vendor():
    task_reg_global = True
    task_no_compcor = True
    rs_reg_global = False
    rs_no_compcor = False

    sns = find_overlapping_sns()
    sns = sorted(list(sns))

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(False, all_roi=False, anat_ver=4,
                  combine_regions=False))

    mats_all = []
    for sn in sns[:500]:
        sn_mats = []
        for lr in ['LR', 'RL']:
            kw = {'sn': sn, 'lr': lr,
                  'pda': p_d_ant, 'pdp': p_d_pos,
                  'pva': p_v_ant, 'pvp': p_v_pos,
                  'p_no': p_no, 'no_ctrl': False}
            mat, cols_order = pickle_wrap(analyze_rs_sn, kwargs=kw,)
            # mat, cols_order = analyze_rs_sn(sn, lr, p_d_ant,
            #                                 p_d_pos, p_v_ant,
            #                                 p_v_pos, p_no)
            sn_mats.append(mat)
            break
        sn_mat = np.nanmean(sn_mats, axis=0)
        mats_all.append(sn_mat)
    corr = np.nanmean(mats_all, axis=0)
    # plt.imshow(corr)
    # plt.colorbar()
    # plt.show()
    # quit()

    tick_lows = np.arange(0, len(cols_order))
    ticks = tick_lows
    tick_labels = cols_order
    vmax = np.nanquantile(corr, .85)
    vmin = np.nanquantile(corr, .15)
    plot_connectivity(corr, ticks, tick_labels, tick_lows,
                      title=None, no_avg=True,
                      vmax=vmax, vmin=vmin,
                      minimal=False)

    bool_ar = np.zeros(corr.shape)
    for i in range(corr.shape[0]):
        row = corr[i]
        median = np.nanmedian(row)
        bool_ar[i, row > median - .0001] += 1
        bool_ar[row > median - .0001, i] += 1
    bool_ar[bool_ar > 1.5] = 1
    corr = bool_ar
    # bool_ar[np.diag_indices_from(bool_ar)] = np.nan
    # M = np.nanmean(bool_ar)
    # print(M)
    # plt.imshow(bool_ar)
    # plt.colorbar()
    # plt.show()
    # quit()

    tick_lows = np.arange(0, len(cols_order))
    ticks = tick_lows
    tick_labels = cols_order
    plot_connectivity(corr, ticks, tick_labels, tick_lows,
                      title=None, no_avg=True, #vmin=-0.15, vmax=0.15,
                      tile=.4, minimal=False)


    partitions = get_modules_2(corr)

    corr_v0 = get_partition_matrix(np.ones(corr.shape), partitions[0],
                                   w_zeros=True)

    for i, p in enumerate(partitions):
        p_named = [cols_order[j] for j in p]
        print(f'{i}: {p=} ({p_named})')

    corr_v1 = get_partition_matrix(np.ones(corr.shape), partitions[1],
                                   w_zeros=True)

    plot_connectivity(corr_v0, ticks, tick_labels, tick_lows,
                      title='HCP', no_avg=True, vmin=-0.3, vmax=0.3)

    plot_connectivity(corr_v1, ticks, tick_labels, tick_lows,
                      title='HCP', no_avg=True, vmin=-0.3, vmax=0.3)


    quit()

if __name__ == '__main__':
    replicate_vendor()
