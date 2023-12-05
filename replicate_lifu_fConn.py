import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from nichord.combine import plot_and_combine

from conn_utils import corr_matrix_last_two_dim, get_ROI_vecs_wrap, get_BNA_ROIs
from fMRI_proc import get_ROI_vecs
from old.analyze_ROIs import prune_bad_sns
from atlas_utils import get_atlas
from organize_bhv import get_all_sns, get_trial_info
from utils import get_RSA_fn, pickle_wrap
from time import time
import matplotlib.pyplot as plt

def get_modules(corr, threshold=.95, bonus_str=''):
    threshold = np.nanquantile(corr, threshold)
    print(f'{threshold=}')
    corr_thresh = corr.copy()
    top_edges_mat = corr > threshold
    # plt.imshow(top_edges_mat)
    # plt.show()
    # test = np.mean(top_edges_mat)
    # print(f'{test=:.3f}')
    corr_thresh[corr_thresh < threshold] = 0
    corr_thresh[corr_thresh >= threshold] = 1
    corr[corr < threshold] = 0
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(corr_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=0)
    return part, top_edges_mat

def get_partition_matrix(mat, idx, w_zeros=False):
    if w_zeros:
        mat_new = np.zeros(mat.shape)
        mat_new[np.ix_(idx, idx)] = mat[np.ix_(idx, idx)]
        return mat_new
    else:
        meshy = np.ix_(idx, idx)
        # print(f'{idx=}')
        # return mat[..., meshy[0], meshy[1]]
        slicer = tuple([slice(None)] * (mat.ndim - 2) + [meshy[0], meshy[1]])
        return mat[slicer]

def plot_nichord(corr, coords, fn, title, dir_out='nichord_plots'):
    from nichord.convert import convert_matrix
    from nichord.coord_labeler import get_idx_to_label

    edges, edge_weights = convert_matrix(corr)
    idx_to_label = get_idx_to_label(coords, atlas='yeo')

    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=coords,
                     network_order=network_order, network_colors=network_colors,
                     title=title, chord_kwargs={'alphas': .5})

def load_FC_for_Lifu(atlas_name='BNA', fp='obj3_fMRI',
                     combine_regions=False, split=False,
                     ):
    if atlas_name == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=False,
                          combine_bilateral=False,
                          split=split, split_code='xyz')
    coords = atlas['coords']
    age2sn = get_all_sns(ret=True)
    age_sn_inc_conn = []
    # age_l = []
    age2idxs = defaultdict(list)
    sn_idx = 0
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        ROIs_l = get_BNA_ROIs(code=None)
        sn_inc_conn = []
        for sn in sns:
            df_sn = get_trial_info(sn)
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=False,
                                     combine_regions=False)
            activity_ar = []
            for ROI in ROIs_l:
                activity_ar.append(np.nanmean(ROI2vecs0[ROI], axis=1))
            activity_ar = np.array(activity_ar)
            print(f'{activity_ar.shape=}')

            activity_inc_ar = np.empty((3, activity_ar.shape[0],
                                           activity_ar.shape[1] // 3))
            conns = []
            for i, inc in enumerate([1, 2, 3]):
                matching_trials = df_sn['inc'] == inc
                activity_inc_ar[i] = activity_ar[:, matching_trials]
                conn_inc = np.corrcoef(activity_ar[:, matching_trials])
                conns.append(conn_inc)
            conns = np.array(conns)
            sn_inc_conn.append(conns)
            # age_l.append(age)
            age2idxs[age].append(sn_idx)
            sn_idx += 1
        age_sn_inc_conn.append(np.array(sn_inc_conn))
    sn_inc_conn = np.concatenate(age_sn_inc_conn, axis=0)
    print(f'{sn_inc_conn.shape=}')
    return sn_inc_conn, age2idxs



def get_partitions(sn_inc_conn, coords=None, plot=False,
                   threshold=.95):
    coords = coords or [[-5, 15, 54], [7, 16, 54], [-18, 24, 53], [22, 26, 51], [-11, 49, 40], [13, 48, 40], [-18, -1, 65], [20, 4, 64], [-6, -5, 58], [7, -4, 60], [-5, 36, 38], [6, 38, 35], [-8, 56, 15], [8, 58, 13], [-27, 43, 31], [30, 37, 36], [-42, 13, 36], [42, 11, 39], [-28, 56, 12], [28, 55, 17], [-41, 41, 16], [42, 44, 14], [-33, 23, 45], [42, 27, 39], [-32, 4, 55], [34, 8, 54], [-26, 60, -6], [25, 61, -4], [-46, 13, 24], [45, 16, 25], [-47, 32, 14], [48, 35, 13], [-53, 23, 11], [54, 24, 12], [-49, 36, -3], [51, 36, -1], [-39, 23, 4], [42, 22, 3], [-52, 13, 6], [54, 14, 11], [-7, 54, -7], [6, 47, -7], [-36, 33, -16], [40, 39, -14], [-23, 38, -18], [23, 36, -18], [-6, 52, -19], [6, 57, -16], [-10, 18, -19], [9, 20, -19], [-41, 32, -9], [42, 31, -9], [-49, -8, 39], [55, -2, 33], [-32, -9, 58], [33, -7, 57], [-26, -25, 63], [34, -19, 59], [-13, -20, 73], [15, -22, 71], [-52, 0, 8], [54, 4, 9], [-49, 5, 30], [51, 7, 30], [-8, -38, 58], [10, -34, 54], [-4, -23, 61], [5, -21, 61], [-32, 14, -34], [31, 15, -34], [-54, -32, 12], [54, -24, 11], [-50, -11, 1], [51, -4, -1], [-62, -33, 7], [66, -20, 6], [-45, 11, -20], [47, 12, -20], [-55, -3, -10], [56, -12, -5], [-65, -30, -12], [65, -29, -13], [-53, 2, -30], [51, 6, -32], [-59, -58, 4], [60, -53, 3], [-58, -20, -9], [58, -16, -10], [-45, -26, -27], [46, -14, -33], [-51, -57, -15], [53, -52, -18], [-43, -2, -41], [40, 0, -43], [-56, -16, -28], [55, -11, -32], [-55, -60, -6], [54, -57, -8], [-59, -42, -16], [61, -40, -17], [-55, -31, -27], [54, -31, -26], [-33, -16, -32], [33, -15, -34], [-31, -64, -14], [31, -62, -14], [-42, -51, -17], [43, -49, -19], [-27, -7, -34], [28, -8, -33], [-25, -25, -26], [26, -23, -27], [-28, -32, -18], [30, -30, -18], [-19, -12, -30], [19, -10, -30], [-23, 2, -32], [22, 1, -36], [-17, -39, -10], [19, -36, -11], [-54, -40, 4], [53, -37, 3], [-52, -50, 11], [57, -40, 12], [-16, -60, 63], [19, -57, 65], [-15, -71, 52], [19, -69, 54], [-33, -47, 50], [35, -42, 54], [-22, -47, 65], [23, -43, 67], [-27, -59, 54], [31, -54, 53], [-34, -80, 29], [45, -71, 20], [-38, -61, 46], [39, -65, 44], [-51, -33, 42], [47, -35, 45], [-56, -49, 38], [57, -44, 38], [-47, -65, 26], [53, -54, 25], [-53, -31, 23], [55, -26, 26], [-5, -63, 51], [6, -65, 51], [-8, -47, 57], [7, -47, 58], [-12, -67, 25], [16, -64, 25], [-6, -55, 34], [6, -54, 35], [-50, -16, 43], [50, -14, 44], [-56, -14, 16], [56, -10, 15], [-46, -30, 50], [48, -24, 48], [-21, -35, 68], [20, -33, 69], [-36, -20, 10], [37, -18, 8], [-32, 14, -13], [33, 14, -13], [-34, 18, 1], [36, 18, 1], [-38, -4, -9], [39, -2, -9], [-38, -8, 8], [39, -7, 8], [-38, 5, 5], [38, 5, 5], [-4, -39, 31], [4, -37, 32], [-3, 8, 25], [5, 22, 12], [-6, 34, 21], [5, 28, 27], [-8, -47, 10], [9, -44, 11], [-5, 7, 37], [4, 6, 38], [-7, -23, 41], [6, -20, 40], [-4, 39, -2], [5, 41, 6], [-11, -82, -11], [10, -85, -9], [-5, -81, 10], [7, -76, 11], [-6, -94, 1], [8, -90, 12], [-17, -60, -6], [18, -60, -7], [-13, -68, 12], [15, -63, 12], [-31, -89, 11], [34, -86, 11], [-46, -74, 3], [48, -70, -1], [-18, -99, 2], [22, -97, 4], [-30, -88, -12], [32, -85, -12], [-11, -88, 31], [16, -85, 34], [-22, -77, 36], [29, -75, 36], [-19, -2, -20], [19, -2, -19], [-27, -4, -20], [28, -3, -20], [-22, -14, -19], [22, -12, -20], [-28, -30, -10], [29, -27, -10], [-12, 14, 0], [15, 14, -2], [-22, -2, 4], [22, -2, 3], [-17, 3, -9], [15, 8, -9], [-23, 7, -4], [22, 8, -1], [-14, 2, 16], [14, 5, 14], [-28, -5, 2], [29, -3, 1], [-7, -12, 5], [7, -11, 6], [-18, -13, 3], [12, -14, 1], [-18, -23, 4], [18, -22, 3], [-7, -14, 7], [3, -13, 5], [-16, -24, 6], [15, -25, 6], [-15, -28, 4], [13, -27, 8], [-12, -22, 13], [10, -14, 14], [-11, -14, 2], [13, -16, 7]]
    M_conn = np.nanmean(sn_inc_conn, axis=(0, 1))


    partitions, top_edges_mat = get_modules(M_conn, threshold=threshold)
    if plot:
        for i, p in enumerate(partitions):
            if len(p) < 5:
                continue
            # print(f'Partition {i}: {p}')
            # print(atlas['coords'])
            M_corr_part = get_partition_matrix(M_conn, p, w_zeros=True,
                                               )
            print(f'{M_corr_part.shape=}')
            fn = f'Dec5_p{i}_M_corr_thr{threshold}.png'
            title = f'Partition {i}'
            plot_nichord(M_corr_part, coords, fn, title,
                         dir_out='result_pics/nichord',)
    return partitions, top_edges_mat


def calculate_within_between(sn_inc_conn, age2idxs, p):
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        p_mat = get_partition_matrix(age_sn_inc_conn, p)
        p_M_within_connectivity = np.nanmean(p_mat, axis=(-2, -1))
        between_mask = np.full((246, 246), False)
        between_mask[p, :] = True
        between_mask[:, p] = True
        between_mask[np.ix_(p, p)] = False
        p_between_edges = age_sn_inc_conn[:, :, between_mask]
        p_M_between_connectivity = np.nanmean(p_between_edges, axis=-1)
        integration = p_M_between_connectivity / p_M_within_connectivity

        print(f'\tAge: {age}')
        for i in range(3):
            M_i = np.mean(integration[:, i])
            SD_i = np.std(integration[:, i])
            SE = SD_i / np.sqrt(integration.shape[0])
            print(f'Inc {i+1}: {M_i:.3f} [{SE:.3f}]')

        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            dif = integration[:, i] - integration[:, j]
            M_dif = np.mean(dif)
            SD_dif = np.std(dif)
            SE = SD_dif / np.sqrt(integration.shape[0])
            t = M_dif / SE
            print(f'Inc {i+1} - {j+1}: {M_dif:.3f} [{SE:.3f}], {t=:.3f}')
        all_inc = np.mean(integration, axis=1)
        M_inc = np.mean(all_inc)
        SD_inc = np.std(all_inc)
        SE_inc = SD_inc / np.sqrt(integration.shape[0])
        print(f'All inc: {M_inc:.3f} [{SE_inc:.3f}]')

def do_Dec5(threshold=0.9):
    fp = 'obj3_fMRI'
    sn_inc_conn, age2idxs = pickle_wrap(f'cache/FC_data_{fp}.pkl',
                                        load_FC_for_Lifu,
                                        kwargs={'fp': fp},
                                        verbose=1, easy_override=False)
    diag = np.diag_indices(sn_inc_conn.shape[-1])
    sn_inc_conn[:, :, diag[0], diag[1]] = np.nan
    partitions, top_edges_mat = get_partitions(sn_inc_conn, plot=False,
                                               threshold=threshold)
    print(partitions)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    # test = np.mean(M)
    # print(f'{test=}')
    # print(f'{np.mean(sn_inc_conn_test)=}')

    # plt.imshow(M, interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    # quit()

    # 90% thresh

def Fig15_analyses(partitions, threshold, sn_inc_conn, age2idxs):
    i2name = {0.9: {0: 'PFC', 1: 'MTL'}}
    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        if len(p) < 5:
            continue
        print('-'*50)
        print(f'Partition: {i2name[threshold][i]} ({i})')
        calculate_within_between(sn_inc_conn, age2idxs, p)


def Fig16a_analyses():
    pass

if __name__ == '__main__':
     do_Dec5()
