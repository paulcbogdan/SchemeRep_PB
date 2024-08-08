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

config.CACHE_DIR = r'H:\PycharmProjects_H\SchemeRep\cache\numba_test'
CACHE_NUMBA = True

@cache
def load_rs():
    sn_roi_act, sns, conn_trials = load_a(fp='rs_medium', norm_std=True)
    return conn_trials

def load_task(combine_regions=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
    atlas = get_atlas(combine_regions=combine_regions)
    bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
             if roi in bad_rois]
    sn_inc_conn[..., bad_j, :] = np.nan
    sn_inc_conn[..., :, bad_j] = np.nan
    z_both = do_regression(sn_inc_conn, flip=False, nans=True)
    partitions_VD, _ = get_main_partitions(z_both)


    z_both = do_regression(sn_inc_conn, flip=True, nans=True)
    partitions_PA, _ = get_main_partitions(z_both)

    partitions_VD = [set(partitions_VD[0]), set(partitions_VD[1])]
    partitions_PA = [set(partitions_PA[0]), set(partitions_PA[1])]

    quad0 = partitions_VD[0].intersection(partitions_PA[0])
    quad1 = partitions_VD[0].intersection(partitions_PA[1])
    quad2 = partitions_VD[1].intersection(partitions_PA[0])
    quad3 = partitions_VD[1].intersection(partitions_PA[1])

    quads = [list(quad0), list(quad1), list(quad2), list(quad3)]

    # print(f'{len(quad0)=}')
    # print(f'{len(quad1)=}')
    # print(f'{len(quad2)=}')
    # print(f'{len(quad3)=}')
    #
    # print(f'{len(partitions_VD[0])=}')
    # print(f'{len(partitions_VD[1])=}')
    # print(f'{len(partitions_PA[0])=}')
    # print(f'{len(partitions_PA[1])=}')

    non_used_nodes = (set(range(246)) - partitions_VD[0] -  partitions_VD[1] -
                      partitions_PA[0] - partitions_PA[1])
    non_used_quad = (set(range(246)) - quad0 - quad1 - quad2 - quad3)

    partitions_VD = [list(partitions_VD[0]), list(partitions_VD[1])]
    partitions_PA = [list(partitions_PA[0]), list(partitions_PA[1])]

    return partitions_VD, partitions_PA, list(non_used_quad), quads

    # print(sn_inc_conn.shape)
    # quit()

    # sn_inc_conn[:, :, 93, :] = np.nan
    # sn_inc_conn[:, :, :, 93] = np.nan

    # sn_inc_conn =

    # for i in range(246):
    #     num_nan = np.sum(np.all(np.isnan(sn_inc_activity[:, :, i, :]),
    #                             axis=(1, 2)), axis=0)
    #     print(i, ':', num_nan)
    # quit()




    # print(sn_inc_conn[:, :, 93, 0])
    # quit()

    # plt.imshow(z_both)
    # plt.show()
    # quit()
    # print(z_both[0, 1])

    # atlas = get_atlas()

    # for k, p in enumerate(partitions):
    #     # if len(p) < 10: continue
    #     if k == 3: break
    #     mat = np.full((246, 246), 0)
    #
    #     for i in p:
    #         for j in p:
    #             mat[i, j] = 1
    #     plot_connectivity(mat, atlas=atlas, vmin=0, vmax=2, minimal=False)
    #
    # # print(partitions)
    # quit()



    # get_vendor_partitions(age='healthy', flip=True, plot=True,
    #                   scrub=True, easy_override=True, thr=THRESHOLD,
    #                   combine_regions=False, regress=REGRESS)

def get_task_triangles(combine_regions=False, strongest_efs=0.1,
                       shuffle=True, seed=0):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=-1, cache_dir='cache',
                    RAM_cache=False)
    # sn_inc_conn = copy.deepcopy(sn_inc_conn)

    if shuffle:
        np.random.seed(seed)
        idxs = np.arange(sn_inc_conn.shape[0])
        np.random.shuffle(idxs)
        sn_inc_conn = sn_inc_conn[idxs]
        start = np.random.randint(6)
        idxs_possible = [[0, 1, 2], [0, 2, 1], [1, 0, 2],
                         [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        idxs_possible = np.array(idxs_possible)
        np.random.shuffle(idxs_possible)
        # idxs = [0, 1, 2]
        for i in range(sn_inc_conn.shape[0]):
            np.random.shuffle(idxs)
            # print(idxs)
            idxs = idxs_possible[(start + i) % 6]
            sn_inc_conn[i, ] = sn_inc_conn[i, idxs, :, :]

    bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
    atlas = get_atlas(combine_regions=combine_regions)
    bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
             if roi in bad_rois]
    sn_inc_conn[..., bad_j, :] = np.nan
    sn_inc_conn[..., :, bad_j] = np.nan
    lowest_bad_j = min(bad_j)
    z_both = do_regression(sn_inc_conn, flip=False, nans=True)
    z_both = z_both[:lowest_bad_j, :lowest_bad_j]
    # print(z_both)
    # print(f'{seed=}')

    coords = np.array(atlas['coords'][:lowest_bad_j])
    pdist = spatial.distance.pdist(coords[:, 1:]) # drop x-dim
    pdist = spatial.distance.squareform(pdist)
    candidates = []
    n_roi = pdist.shape[0]
    for i in range(n_roi):
        idxs = np.argsort(pdist[i])
        candidates.append(idxs[n_roi // 4:])

    candidates = np.array(candidates)
    triangles = []
    for i in range(n_roi):
        triangles_i = []
        candidate_zs = z_both[i, candidates[i]]
        lowest_zs = np.argsort(candidate_zs)[
                    :int(strongest_efs * len(candidate_zs))]
        highest_zs = np.argsort(candidate_zs)[
                        -int(strongest_efs * len(candidate_zs)):]
        for neg in lowest_zs:
            for pos in highest_zs:
                triangles_i.append((i, candidates[i][neg], candidates[i][pos]))
        triangles.append(triangles_i)
    triangles = np.array(triangles)
    # print(triangles[:2, :2])
    return triangles



def calc_corr(combine_regions=False):
    conn_trials = load_rs()
    partitions_VD, partitions_PA, non_used_nodes, quads = (
        load_task(combine_regions=combine_regions))



    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_ver=3,
                              combine_regions=combine_regions)

    partitions_VD = [p_d_ant + p_d_pos, p_v_ant + p_v_pos]
    partitions_PA = [p_d_ant + p_v_ant, p_d_pos + p_v_pos]

    non_used_nodes = (set(range(246)) -
                      set(partitions_VD[0]) - set(partitions_VD[1]) -
                      set(partitions_PA[0]) - set(partitions_PA[1]))
    non_used_nodes = list(non_used_nodes)

    # VD0 = conn_trials[:, *np.ix_(partitions_VD[0], partitions_VD[0]), :]
    VD0 = conn_trials[:, *np.ix_(p_d_ant, p_d_pos), :]

    # VD0 = conn_trials[:, *np.ix_(quads[0], quads[1])]
    VD0 = np.nanmean(VD0, axis=(1, 2))
    # VD1 = conn_trials[:, *np.ix_(partitions_VD[1], partitions_VD[1]), :]
    VD1 = conn_trials[:, *np.ix_(p_v_ant, p_v_pos), :]
    # VD1 = conn_trials[:, *np.ix_(quads[2], quads[3])]
    VD1 = np.nanmean(VD1, axis=(1, 2))
    # PA0 = conn_trials[:, *np.ix_(partitions_PA[0], partitions_PA[0]), :]
    PA0 = conn_trials[:, *np.ix_(p_v_ant, p_d_ant), :]
    # PA0 = conn_trials[:, *np.ix_(quads[0], quads[2])]
    PA0 = np.nanmean(PA0, axis=(1, 2))
    # PA1 = conn_trials[:, *np.ix_(partitions_PA[1], partitions_PA[1]), :]
    PA1 = conn_trials[:, *np.ix_(p_v_pos, p_d_pos), :]
    # PA1 = conn_trials[:, *np.ix_(quads[1], quads[3])]
    PA1 = np.nanmean(PA1, axis=(1, 2))



    glob = np.nanmean(conn_trials, axis=(1, 2))
    # nde = conn_trials[:, *np.ix_(non_used_nodes, non_used_nodes), :]
    # nde = np.nanmean(nde, axis=(1, 2))
    #
    # # nde0 = conn_trials[:, *np.ix_(quads[0], non_used_nodes), :]
    # nde0 = conn_trials[:, *np.ix_(partitions_VD[0], non_used_nodes), :]
    nde0 = conn_trials[:, *np.ix_(p_d_ant, list(set(range(246)) )), :]
    nde0 = np.nanmean(nde0, axis=(1, 2))
    # nde1 = conn_trials[:, *np.ix_(partitions_VD[1], non_used_nodes), :]
    nde1 = conn_trials[:, *np.ix_(p_v_ant, non_used_nodes), :]
    nde1 = np.nanmean(nde1, axis=(1, 2))
    # nde2 = conn_trials[:, *np.ix_(partitions_PA[0], non_used_nodes), :]
    nde2 = conn_trials[:, *np.ix_(p_d_pos, non_used_nodes), :]
    nde2 = np.nanmean(nde2, axis=(1, 2))
    # nde3 = conn_trials[:, *np.ix_(partitions_PA[1], non_used_nodes), :]
    nde3 = conn_trials[:, *np.ix_(p_v_pos, non_used_nodes), :]
    nde3 = np.nanmean(nde3, axis=(1, 2))
    #
    df = pd.DataFrame({'VD0': VD0.flatten(), 'VD1': VD1.flatten(),
                       'PA0': PA0.flatten(), 'PA1': PA1.flatten(),
                       'global': glob.flatten(),
                       # 'NDE': nde.flatten(),
                       'NDE0': nde0.flatten(), 'NDE1': nde1.flatten(),
                       'NDE2': nde2.flatten(), 'NDE3': nde3.flatten()})

    df['dd_vv'] = df['VD0'] + df['VD1']
    df['dv_dv'] = df['PA0'] + df['PA1']

    # all_nodes = np.concatenate([quads[0], quads[1], quads[2], quads[3], ])

    # for quad in [quads[0], quads[1], quads[2], quads[3]]:
    # # for quad in [partitions_VD[0], partitions_VD[1],
    # #              partitions_PA[0], partitions_PA[1]]:
    #     atlas = get_atlas(combine_regions=combine_regions)
    #     coords = [atlas['coords'][i] for i in quad]
    #     plotting.plot_markers([1] * len(coords), coords)
    #     plt.show()
    # quit()

    # plt.scatter(df['dd_vv'], df['dv_dv'])
    # plt.show()

    # 'VD0', 'VD1', 'PA0', 'PA1'
    partial_corr_df(df, ['dd_vv', 'dv_dv'],
                        ['NDE0', 'NDE1', 'NDE2', 'NDE3', ])
    # 'NDE0', 'NDE1', 'NDE2', 'NDE3'
    # ['global']

    # ['dd_vv', 'dv_dv'], #

def calc_triangle_corr(shuffle=True, seed=0, strongest_efs=0.1,):
    conn_trials = load_rs()
    # print(conn_trials.shape)
    sns_ok = list(range(27)) + [28, 29] + list(range(31, conn_trials.shape[0]))
    conn_trials = conn_trials[sns_ok]

    triangles = get_task_triangles(shuffle=shuffle, seed=seed,
                                   strongest_efs=strongest_efs)

    edge0s = []
    edge1s = []

    edge0s_total = np.zeros((conn_trials.shape[0], 206))
    edge1s_total = np.zeros((conn_trials.shape[0], 206))
    for roi_i in range(triangles.shape[0]):
        for triangle_j in range(triangles.shape[1]):
            triangle = triangles[roi_i, triangle_j]
            edge0 = conn_trials[:, triangle[0], triangle[1], :]
            edge0s_total += edge0
            # print(edge0.shape)
            # quit()
            # edge0s.append(edge0)
            edge1 = conn_trials[:, triangle[0], triangle[2], :]
            edge1s_total += edge1
            # edge1s.append(edge1)
    t_st = time()
    edge0s = np.array(edge0s)[None, :, :]
    print(f'{edge0s.shape=}')
    edge1s = np.array(edge1s)[None, :, :]
    # edge0s = np.mean(edge0s, axis=0)[None, :, :]
    # edge1s = np.mean(edge1s, axis=0)[None, :, :]

    # edge0s = edge0s[:10, :, :]
    # edge1s = edge1s[:, :2, :]
    r_gavg = numba_corr(edge0s, edge1s)
    print(f'{shuffle} | {r_gavg=:.7f} | {time() - t_st:.2f}')
    # return r_gavg
    # edge0s = np.array(edge0s)
    # print(edge0s.shape)
    # quit()
    # edge0s = stdize(edge0s, axis=2)
    # edge1s = np.array(edge1s)
    # edge1s = stdize(edge1s, axis=2)
    # prod = edge0s * edge1s
    # r = np.mean(prod, axis=2)
    # print(r[0, 0])
    # r_avg = np.nanmean(r, axis=0)
    # r_gavg2 = np.nanmean(r_avg)
    # print(f'{shuffle} | {r_gavg=:.7f} | {r_gavg2=:.7f}')
    # quit()
    return r_gavg

@jit(cache=CACHE_NUMBA, fastmath=True, nopython=True,)
def numba_corr(edges0, edges1):
    n_triangles = edges0.shape[0]
    n_sn = edges0.shape[1]
    n_t = edges0.shape[2]

    # r_out = np.empty((n_triangles, n_sn))
    r_total = 0
    for j in range(n_sn):
        for i in range(n_triangles):
            M = edges0[i, j, :].mean()
            SD = edges0[i, j, :].std()
            for k in range(n_t):
                edges0[i, j, k] = (edges0[i, j, k] - M) / SD
            M = edges1[i, j, :].mean()
            SD = edges1[i, j, :].std()
            for k in range(n_t):
                edges1[i, j, k] = (edges1[i, j, k] - M) / SD

            prod_total = 0
            for k in range(n_t):
                prod_total += edges0[i, j, k] * edges1[i, j, k]
            # r_out[i, j] = prod_total / n_t
            r_total += prod_total / n_t
            # print(j, i, prod_total / n_t)
            # return
            # print(prod_total / n_t)
    r_total /= n_triangles * n_sn
    return r_total
    # return r_out, r_total




def calc_triangle_shuffle(strongest_efs=0.2,):
    n = 100
    rs = []
    for i in range(n):
        rs.append(calc_triangle_corr(shuffle=True, seed=i,
                                     strongest_efs=strongest_efs))
        cutoff = sorted(rs)[int((i + 1) * .05)]
        print(f'\t{cutoff=:.3f}')
    rs = np.array(rs)
    print(f'{rs=}')
    print(f'{np.mean(rs)=}')
    print(f'{np.std(rs)=}')

if __name__ == '__main__':
    calc_triangle_corr(strongest_efs=0.3, shuffle=False)
    calc_triangle_shuffle(strongest_efs=0.3,)


