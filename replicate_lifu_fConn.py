import pickle
from collections import defaultdict
from pathlib import Path
import random

import matplotlib
import numpy as np
from nichord.combine import plot_and_combine

from conn_utils import corr_matrix_last_two_dim, get_ROI_vecs_wrap, get_BNA_ROIs
from fMRI_proc import get_ROI_vecs
from old.analyze_ROIs import prune_bad_sns
from atlas_utils import get_atlas
from organize_bhv import get_all_sns, get_trial_info
from utils import get_RSA_fn, pickle_wrap, stdize
from time import time
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as stats

def get_binary_matrix(matrix, threshold=.9):
    threshold = np.nanquantile(matrix, threshold)
    matrix_binary = matrix.copy()
    matrix_mask = matrix > threshold
    matrix_binary[matrix_binary < threshold] = 0
    matrix_binary[matrix_binary >= threshold] = 1
    matrix[matrix < threshold] = 0
    return matrix_binary, matrix_mask

def get_modules(matrix_thresh):
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(matrix_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=0)
    return part

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

def load_FC_for_Lifu(atlas_name='BNA', fp='obj3_fMRI', split=False, key='inc',
                     key_vals=(1, 2, 3), odd_even=False):
    if atlas_name == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=False,
                          combine_bilateral=False,
                          split=split, split_code='xyz')
    coords = atlas['coords']
    age2sn = get_all_sns(ret=True)
    # age_sn_inc_conn = []
    # age_sn_conn = []
    sn_inc_conn = []
    sn_conn = []

    # age_l = []
    age2idxs = defaultdict(list)
    sn_idx = 0
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        ROIs_l = get_BNA_ROIs(code=None)
        for sn in sns:
            print(f'Prepping FC: {sn=}')
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
            conn_no_cond = np.corrcoef(activity_ar)
            sn_conn.append(conn_no_cond)
            activity_inc_ar = []
            conns = []
            for i, inc in enumerate(key_vals):
                if key == 'rand':
                    if key_vals == (1, 2, 3):
                        matching_trials = df_sn['inc'].sample(frac=1.) == inc
                    else:
                        matching_trials = df_sn['vis_hit'].sample(frac=1.) == inc
                else:
                    matching_trials = df_sn[key] == inc
                if odd_even:
                    matching_trials_even = []
                    matching_trials_odd = []
                    cnt = 0
                    org = [False] * (sum(matching_trials) // 2) + \
                          [True] * (sum(matching_trials) // 2)
                    if sum(matching_trials) % 2:
                        org += [False]
                    random.shuffle(org)
                    for trial in matching_trials:
                        if trial:
                            matching_trials_even.append(org[cnt])
                            matching_trials_odd.append(not org[cnt])
                            cnt += 1
                        else:
                            matching_trials_even.append(False)
                            matching_trials_odd.append(False)
                    activity_inc_ar.append([activity_ar[:, matching_trials_even],
                                            activity_ar[:, matching_trials_odd]])
                    # matching_even = stdize(activity_ar[:, matching_trials_even],
                    #                    axis=1)
                    # conn_even = matching_even[:, None, :] - \
                    #             matching_even[None, :, :]
                    # conn_even = np.nanmean(conn_even, axis=-1)
                    # matching_odd = stdize(activity_ar[:, matching_trials_odd],
                    #                         axis=1)
                    # conn_odd = matching_odd[:, None, :] - \
                    #            matching_odd[None, :, :]
                    # conn_odd = np.nanmean(conn_odd, axis=-1)
                    # conns.append([conn_even, conn_odd])
                    conns.append([np.corrcoef(activity_ar[:, matching_trials_even]),
                                  np.corrcoef(activity_ar[:, matching_trials_odd])])
                else:
                    activity_inc_ar.append(activity_ar[:, matching_trials])
                    conn_inc = np.corrcoef(activity_ar[:, matching_trials])
                    conns.append(conn_inc)
            conns = np.array(conns)
            sn_inc_conn.append(conns)
            age2idxs[age].append(sn_idx)
            sn_idx += 1
    sn_inc_conn = np.array(sn_inc_conn)
    sn_conn = np.array(sn_conn)

    diag = np.diag_indices(sn_inc_conn.shape[-1])
    sn_inc_conn[..., diag[0], diag[1]] = np.nan
    sn_conn[..., diag[0], diag[1]] = np.nan

    return sn_inc_conn, sn_conn, age2idxs

def get_BNA_coords():
    coords = [[-5, 15, 54], [7, 16, 54], [-18, 24, 53], [22, 26, 51], [-11, 49, 40], [13, 48, 40], [-18, -1, 65],
              [20, 4, 64], [-6, -5, 58], [7, -4, 60], [-5, 36, 38], [6, 38, 35], [-8, 56, 15], [8, 58, 13], [-27, 43, 31], [30, 37, 36], [-42, 13, 36], [42, 11, 39], [-28, 56, 12], [28, 55, 17], [-41, 41, 16], [42, 44, 14], [-33, 23, 45], [42, 27, 39], [-32, 4, 55], [34, 8, 54], [-26, 60, -6], [25, 61, -4], [-46, 13, 24], [45, 16, 25], [-47, 32, 14], [48, 35, 13], [-53, 23, 11], [54, 24, 12], [-49, 36, -3], [51, 36, -1], [-39, 23, 4], [42, 22, 3], [-52, 13, 6], [54, 14, 11], [-7, 54, -7], [6, 47, -7], [-36, 33, -16], [40, 39, -14], [-23, 38, -18], [23, 36, -18], [-6, 52, -19], [6, 57, -16], [-10, 18, -19], [9, 20, -19], [-41, 32, -9], [42, 31, -9], [-49, -8, 39], [55, -2, 33], [-32, -9, 58], [33, -7, 57], [-26, -25, 63], [34, -19, 59], [-13, -20, 73], [15, -22, 71], [-52, 0, 8], [54, 4, 9], [-49, 5, 30], [51, 7, 30], [-8, -38, 58], [10, -34, 54], [-4, -23, 61], [5, -21, 61], [-32, 14, -34], [31, 15, -34], [-54, -32, 12], [54, -24, 11], [-50, -11, 1], [51, -4, -1], [-62, -33, 7], [66, -20, 6], [-45, 11, -20], [47, 12, -20], [-55, -3, -10], [56, -12, -5], [-65, -30, -12], [65, -29, -13], [-53, 2, -30], [51, 6, -32], [-59, -58, 4], [60, -53, 3], [-58, -20, -9], [58, -16, -10], [-45, -26, -27], [46, -14, -33], [-51, -57, -15], [53, -52, -18], [-43, -2, -41], [40, 0, -43], [-56, -16, -28], [55, -11, -32], [-55, -60, -6], [54, -57, -8], [-59, -42, -16], [61, -40, -17], [-55, -31, -27], [54, -31, -26], [-33, -16, -32], [33, -15, -34], [-31, -64, -14], [31, -62, -14], [-42, -51, -17], [43, -49, -19], [-27, -7, -34], [28, -8, -33], [-25, -25, -26], [26, -23, -27], [-28, -32, -18], [30, -30, -18], [-19, -12, -30], [19, -10, -30], [-23, 2, -32], [22, 1, -36], [-17, -39, -10], [19, -36, -11], [-54, -40, 4], [53, -37, 3], [-52, -50, 11], [57, -40, 12], [-16, -60, 63], [19, -57, 65], [-15, -71, 52], [19, -69, 54], [-33, -47, 50], [35, -42, 54], [-22, -47, 65], [23, -43, 67], [-27, -59, 54], [31, -54, 53], [-34, -80, 29], [45, -71, 20], [-38, -61, 46], [39, -65, 44], [-51, -33, 42], [47, -35, 45], [-56, -49, 38], [57, -44, 38], [-47, -65, 26], [53, -54, 25], [-53, -31, 23], [55, -26, 26], [-5, -63, 51], [6, -65, 51], [-8, -47, 57], [7, -47, 58], [-12, -67, 25], [16, -64, 25], [-6, -55, 34], [6, -54, 35], [-50, -16, 43], [50, -14, 44], [-56, -14, 16], [56, -10, 15], [-46, -30, 50], [48, -24, 48], [-21, -35, 68], [20, -33, 69], [-36, -20, 10], [37, -18, 8], [-32, 14, -13], [33, 14, -13], [-34, 18, 1], [36, 18, 1], [-38, -4, -9], [39, -2, -9], [-38, -8, 8], [39, -7, 8], [-38, 5, 5], [38, 5, 5], [-4, -39, 31], [4, -37, 32], [-3, 8, 25], [5, 22, 12], [-6, 34, 21], [5, 28, 27], [-8, -47, 10], [9, -44, 11], [-5, 7, 37], [4, 6, 38], [-7, -23, 41], [6, -20, 40], [-4, 39, -2], [5, 41, 6], [-11, -82, -11], [10, -85, -9], [-5, -81, 10], [7, -76, 11], [-6, -94, 1], [8, -90, 12], [-17, -60, -6], [18, -60, -7], [-13, -68, 12], [15, -63, 12], [-31, -89, 11], [34, -86, 11], [-46, -74, 3], [48, -70, -1], [-18, -99, 2], [22, -97, 4], [-30, -88, -12], [32, -85, -12], [-11, -88, 31], [16, -85, 34], [-22, -77, 36], [29, -75, 36], [-19, -2, -20], [19, -2, -19], [-27, -4, -20], [28, -3, -20], [-22, -14, -19], [22, -12, -20], [-28, -30, -10], [29, -27, -10], [-12, 14, 0], [15, 14, -2], [-22, -2, 4], [22, -2, 3], [-17, 3, -9], [15, 8, -9], [-23, 7, -4], [22, 8, -1], [-14, 2, 16], [14, 5, 14], [-28, -5, 2], [29, -3, 1], [-7, -12, 5], [7, -11, 6], [-18, -13, 3], [12, -14, 1], [-18, -23, 4], [18, -22, 3], [-7, -14, 7], [3, -13, 5], [-16, -24, 6], [15, -25, 6], [-15, -28, 4], [13, -27, 8], [-12, -22, 13], [10, -14, 14], [-11, -14, 2], [13, -16, 7]]
    return coords

def get_main_partitions(sn_inc_conn, coords=None, plot=False,
                        threshold=.95, fn_str=''):
    coords = coords or get_BNA_coords()
    M_conn = np.nanmean(sn_inc_conn,
                        axis=tuple(range(len(sn_inc_conn.shape[:-2]))))
    matrix_binary, matrix_mask = get_binary_matrix(M_conn,
                                                     threshold=threshold)
    M_conn_masked = M_conn * matrix_mask
    partitions = get_modules(matrix_binary)
    if plot:
        for i, p in enumerate(partitions):
            if len(p) < 5:
                continue
            print(f'Plotting partition: {i}')
            M_corr_part = get_partition_matrix(M_conn_masked, p, w_zeros=True)
            fn = f'Dec5_{fn_str}thr{threshold}_p{i}.png'
            title = f'Partition {i}'
            plot_nichord(M_corr_part, coords, fn, title,
                         dir_out='result_pics/nichord',)
    partitions = [p for p in partitions]
    return partitions, matrix_mask

def get_ylim_settings(sn_inc_conn, age2idxs, ps, do_division=True):
    y_low = 1e6
    y_high = -1e6
    for p in ps:
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
            if do_division:
                integration = p_M_between_connectivity / p_M_within_connectivity
            else:
                integration = p_M_between_connectivity
            idxs = integration.shape[1]
            for i in range(idxs):
                M_i = np.mean(integration[:, i])
                SD_i = np.std(integration[:, i])
                SE = SD_i / np.sqrt(integration.shape[0])
                candidate_high = M_i + SE * 2
                candidate_low = M_i - SE * 2
                if candidate_high > y_high:
                    y_high = candidate_high
                if candidate_low < y_low:
                    y_low = candidate_low
    return y_low, y_high



def calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                             y_low, y_high,
                             do_division=True, suptitle=None,
                             ):

    matplotlib.rc('font', **{'size': 14})
    fig, axs = plt.subplots(1, 2)
    Ms_all = []
    SDs_all = []
    SEs_all = []
    for age in [1, 2]:
        plt.sca(axs[age-1])
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

        if do_division:
            integration = p_M_between_connectivity / p_M_within_connectivity
            # plt.ylim((0.7, 1.1))
            plt.ylabel('Integration')
        else:
            integration = p_M_between_connectivity
            # plt.ylim((0.24, 0.31))
            plt.ylabel('Between connectivity')

        Ms = []
        errs = []

        print(f'\tAge: {age}')
        idxs = integration.shape[1]
        for i in range(idxs):
            M_i = np.mean(integration[:, i])
            SD_i = np.std(integration[:, i])
            SE = SD_i / np.sqrt(integration.shape[0])
            Ms.append(M_i)
            errs.append(SE)
            print(f'Mean  {i}: {M_i:.3f} [{SE:.3f}]')

        plt.bar(labels, Ms, yerr=errs, color='dodgerblue' if age == 1 else 'r',
                capsize=6)
        dist = (np.max(Ms) - np.min(Ms) + 2 * np.max(errs)) / 2
        mid = (np.max(Ms) + np.min(Ms)) / 2

        comparisons = combinations(list(range(idxs)), 2)
        height = mid + dist * 1.05
        for (i, j) in comparisons:
            dif = integration[:, i] - integration[:, j]
            M_dif = np.mean(dif)
            SD_dif = np.std(dif)
            SE = SD_dif / np.sqrt(integration.shape[0])
            t = M_dif / SE
            p_val = stats.t.sf(np.abs(t), integration.shape[0]-1)*2
            if p_val < 0.1:
                plt.plot([i, j], [height, height], color='k')
                plt.text((i+j)/2, height + dist * 0.02,
                         f'p = {p_val:.3f}', ha='center', va='bottom')
                height += dist * 0.35

            print(f't-test {i} - {j}: {M_dif:.3f} [{SE:.3f}], {t=:.3f}')
        Y_low = mid - dist * 1.3
        Y_high = height + dist * 0.15


        all_inc = np.mean(integration, axis=1)
        M_inc = np.mean(all_inc)
        SD_inc = np.std(all_inc)
        SE_inc = SD_inc / np.sqrt(integration.shape[0])
        print(f'All inc: {M_inc:.3f} [{SE_inc:.3f}]')
        Ms_all.append(M_inc)
        SDs_all.append(SD_inc)
        SEs_all.append(SE_inc)

        plt.ylim((Y_low, Y_high))
        age2str = {1: 'YA', 2: 'OA'}
        # matplotlib.use('ps')
        # matplotlib.rc('text', usetex=True)
        # matplotlib.rc('text.latex', preamble=r'\usepackage{color}')
        # plt.title(r'\textcolor{red}{Today}')
        plt.title(f'Age = {age2str[age]}', color='dodgerblue' if age == 1 else 'r')
        plt.ylim((y_low, y_high))

    plt.tight_layout()
    if suptitle is not None: plt.suptitle(suptitle)
    plt.show()

    plt.figure(figsize=(4, 4))
    t, p_val = stats.ttest_ind_from_stats(Ms_all[0], SDs_all[0], len(age2idxs[1]),
                                      Ms_all[1], SDs_all[1], len(age2idxs[2]))
    print(f'Between age t-test: {t=:.3f}, {p_val=:.3f}')
    dist = (np.max(Ms_all) - np.min(Ms_all) + 2 * np.max(SEs_all)) / 2
    mid = (np.max(Ms_all) + np.min(Ms_all)) / 2
    height = mid + dist * 1.1
    plt.plot([0, 1], [height, height], color='k')
    plt.text(0.5, height + dist * 0.05,
             f'p = {p_val:.3f}', ha='center', va='bottom')
    Y_low = mid - dist * 1.3
    Y_high = height + dist * 0.35
    # plt.ylim((Y_low, Y_high))
    plt.title(suptitle)
    plt.bar(['Young', 'Old'], Ms_all, yerr=SEs_all, color=['dodgerblue', 'r'],
            capsize=6)
    if do_division:
        # plt.ylim((0.7, 1.1))
        plt.ylabel('Integration')
    else:
        # plt.ylim((0.24, 0.31))
        plt.ylabel('Between connectivity')
    plt.ylim((y_low, y_high))
    plt.tight_layout()
    plt.show()

def reconfiguration(sn_inc_conn, age2idxs, p, title=None,
                    col0=0, col1=1, plot=False):
    matplotlib.rc('font', **{'size': 14})
    Ms = []
    SEs = []
    labels =[ 'Young', 'Old']
    plt.figure(figsize=(4, 4))
    rs_both = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        p_mat = get_partition_matrix(age_sn_inc_conn, p)
        rs = []
        for sn_i in range(p_mat.shape[0]):
            R_mat = p_mat[sn_i, col0]
            F_mat = p_mat[sn_i, col1]
            tril = np.tril_indices_from(R_mat, k=-1)
            R_flat = R_mat[tril]
            F_flat = F_mat[tril]
            r, _ = stats.spearmanr(R_flat, F_flat, nan_policy='omit')

            rs.append(1-r)

        # for sn_i in range(p_mat.shape[0]):
        #     R_mat = p_mat[sn_i, 0]
        #     F_mat = p_mat[sn_i, 1]
        #     K_mat = p_mat[sn_i, 2]
        #     tril = np.tril_indices_from(R_mat, k=-1)
        #     R_flat = R_mat[tril]
        #     F_flat = F_mat[tril]
        #     K_flat = K_mat[tril]
        #     r, _ = stats.spearmanr(R_flat, F_flat, nan_policy='omit')
        #     r2, _ = stats.spearmanr(R_flat, K_flat, nan_policy='omit')
        #     r3, _ = stats.spearmanr(F_flat, K_flat, nan_policy='omit')
        #     # rs.append(1-r)
        #     rs.append(3-r-r2-r3)

        M_reconfig = np.mean(rs)
        SD_reconfig = np.std(rs)
        SE_reconfig = SD_reconfig / np.sqrt(len(rs))
        print(f'{age=} Reconfiguration ({title}): {M_reconfig:.3f} '
              f'[{SE_reconfig:.3f}]')
        Ms.append(M_reconfig)
        SEs.append(SE_reconfig)
        rs_both.append(rs)
    super_M = np.mean(rs_both[0] + rs_both[1])
    super_SD = np.std(rs_both[0] + rs_both[1])
    rs_both[0] = np.array(rs_both[0])
    # rs_both[0] = (rs_both[0] - super_M) / super_SD
    rs_both[1] = np.array(rs_both[1])
    dif = np.mean(rs_both[0]) - np.mean(rs_both[1])
    # rs_both[1] = (rs_both[1] - super_M) / super_SD
    M0 = np.mean(rs_both[0])
    M1 = np.mean(rs_both[1])
    Ms = [M0, M1]
    SEs = [np.std(rs_both[0]) / np.sqrt(len(rs_both[0])),
           np.std(rs_both[1]) / np.sqrt(len(rs_both[1]))]

    t, p_val = stats.ttest_ind(rs_both[0], rs_both[1])
    d = t / np.sqrt(len(rs_both[0]) + len(rs_both[1]))
    print(f'\tAge-related reconfiguration ({title}): {t=:.3f}, {p_val=:.3f}, '
          f'{d=:.3f}, {dif=:.3f}')
    dist = (np.max(Ms) - np.min(Ms) + 2 * np.max(SEs)) / 2
    mid = (np.max(Ms) + np.min(Ms)) / 2
    height = mid + dist * 1.05
    if plot:
        return M0, M1, d, dif

    plt.plot([0, 1], [height, height], color='k')
    plt.text(0.5, height + dist * 0.02,
             f'p = {p_val:.3f}', ha='center', va='bottom')
    height += dist * 0.35
    plt.bar(labels, Ms, yerr=SEs, color=['dodgerblue', 'r'], capsize=6)
    Y_low = mid - dist * 1.3
    Y_high = height + dist * 0.15
    plt.ylim((Y_low, Y_high))
    # plt.ylim((0.6, 0.8))
    if 'vs' in title:
        plt.ylabel('Schema-related reconfiguration')
    else:
        plt.ylabel('Memory-related reconfiguration')
    if title is not None: plt.title(title)
    plt.tight_layout()
    plt.show()
    return M0, M1, d, dif

def Fig15_Table2_analyses(threshold=0.9):
    fp = 'obj3_fMRI'
    labels = ['Inc', 'Neu', 'Con']
    kwargs = {'fp': fp, 'split': False, 'key': 'inc', 'key_vals': (1, 2, 3)}
    sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,#f'cache/FC_data_{fp}.pkl',
                                        load_FC_for_Lifu, kwargs=kwargs,
                                        verbose=1, easy_override=False,
                                        cache_dir='cache')
    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}} # 3: 'PCun', 4: 'SM',
    ps_good = [partitions[i] for i in i2name[threshold]]
    y_low_div, y_high_div = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                      do_division=True)
    y_low, y_high = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                      do_division=False)
    # print(f'{y_low=}, {y_high=}')
    # quit()
    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        if len(p) < 5:
            continue
        print('-'*50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 do_division=True, suptitle=name,
                                 y_low=y_low_div, y_high=y_high_div)
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 do_division=False, suptitle=name,
                                 y_low=y_low, y_high=y_high)

def Fig16a_analyses(threshold=0.9, memory_type='vis_hit'):
    fp = 'obj3_fMRI'
    if memory_type == 'vis_hit':
        labels = ['Vis hit', 'Vis miss']
    elif memory_type == 'con_hit':
        labels = ['Con hit', 'Con miss']
    elif memory_type == 'hit_hit':
        labels = ['Both hit', 'Either miss']
    else:
        raise ValueError(f'Unknown memory type: {memory_type=}')
    kwargs = {'fp': fp, 'split': False, 'key': memory_type,
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,
                                        load_FC_for_Lifu, kwargs=kwargs,
                                        verbose=1, easy_override=False,
                                        cache_dir='cache')
    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}
    ps_good = [partitions[i] for i in i2name[threshold]]
    y_low, y_high = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                      do_division=True)
    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        if len(p) < 5:
            continue
        print('-'*50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        reconfiguration(sn_inc_conn, age2idxs, p,
                        title=f'{name}, {memory_type}')
        # quit()
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 y_low, y_high,
                                 do_division=True, suptitle=name)

def Fig17_analyses(threshold=0.9, plot=True):
    fp = 'obj3_fMRI'
    kwargs = {'fp': fp, 'split': False, 'key': 'vis_hit',
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,
                                        load_FC_for_Lifu, kwargs=kwargs,
                                        verbose=1, easy_override=False,
                                        cache_dir='cache')


    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    PFC_partition = partitions[0]

    sn_inc_conn = get_partition_matrix(sn_inc_conn, PFC_partition)
    sn_conn = get_partition_matrix(sn_conn, PFC_partition)
    PFC_coords = [coord for i, coord in enumerate(get_BNA_coords())
                  if i in PFC_partition]
    sub_partitions, top_edges_mat = get_main_partitions(sn_conn, plot=True,
                                                        threshold=threshold,
                                                        fn_str='PFC_')

def analyze_subject_specific(sn_inc_conn, age2idxs):
    sames_all = []
    difs_all = []
    efs_all = []
    n_elements = sn_inc_conn.shape[1] * sn_inc_conn.shape[2]
    # print(M_conn.shape)
    # quit()
    # sn_inc_conn = sn_inc_conn - M_conn[:, None, None, :, :]
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        M_conn = np.nanmean(age_sn_inc_conn, axis=(1, 2))

        sames = []
        difs = []
        efs = []

        for sn in range(age_sn_inc_conn.shape[0]):
            sames_sn = []
            difs_sn = []
            for inc0 in range(age_sn_inc_conn.shape[1]):
                for inc1 in range(age_sn_inc_conn.shape[1]):
                    if inc0 > inc1:
                        continue

                    mat0 = age_sn_inc_conn[sn, inc0, 0]
                    mat1 = age_sn_inc_conn[sn, inc1, 1]
                    mat_else = M_conn[sn]
                    mat_else = (mat_else * n_elements - mat0 - mat1) / (n_elements - 2)
                    # mat0 -= mat_else
                    # mat1 -= mat_else
                    # mat0 -= (mat_else * n_elements - mat0) / (n_elements - 1)
                    # mat1 -= (mat_else * n_elements - mat1) / (n_elements - 1)
                    trils = np.tril_indices_from(mat0, k=-1)
                    flat0_a = mat0[trils]
                    flat1_a = mat1[trils]
                    # r_a = np.nanmean(flat0_a - flat1_a)
                    r_a, _ = stats.spearmanr(flat0_a, flat1_a, nan_policy='omit')

                    if inc0 == inc1:
                        sames_sn.append(r_a)
                    else:
                        mat0 = age_sn_inc_conn[sn, inc0, 1]
                        mat1 = age_sn_inc_conn[sn, inc1, 0]
                        mat_else = M_conn[sn]
                        # mat0 -= (mat_else * n_elements - mat0) / (n_elements - 1)
                        # mat1 -= (mat_else * n_elements - mat1) / (n_elements - 1)

                        # plt.imshow(mat0)
                        # plt.show()
                        # mat_else = (mat_else * n_elements - mat0 - mat1) / (n_elements - 2)
                        # mat0 -= mat_else
                        # mat1 -= mat_else

                        # if sn == 0 and inc0 == 0:
                        #     plt.imshow(mat_else)
                        #     plt.title(f'age = {age}')
                        #     plt.show()

                        # plt.imshow(mat0)
                        # plt.show()
                        # quit()

                        trils = np.tril_indices_from(mat0, k=-1)
                        flat0_b = mat0[trils]
                        flat1_b = mat1[trils]
                        r_b, _ = stats.spearmanr(flat0_b, flat1_b, nan_policy='omit')
                        # r_b = np.nanmean(flat0_b - flat1_b)
                        difs_sn.extend([r_a, r_b])
                        # mat0 = age_sn_inc_conn[sn, inc0, 0]
                        # mat1 = age_sn_inc_conn[sn, inc1, 0]
                        # trils = np.tril_indices_from(mat0, k=-1)
                        # flat0_a = mat0[trils]
                        # flat1_a = mat1[trils]
                        # r_a, _ = stats.spearmanr(flat0_a, flat1_a, nan_policy='omit')
                        # mat0 = age_sn_inc_conn[sn, inc0, 1]
                        # mat1 = age_sn_inc_conn[sn, inc1, 1]
                        # trils = np.tril_indices_from(mat0, k=-1)
                        # flat0_b = mat0[trils]
                        # flat1_b = mat1[trils]
                        # r_b, _ = stats.spearmanr(flat0_b, flat1_b, nan_policy='omit')
                        # difs_sn.extend([r_a, r_b])
            if sn == 0:
                print(f'{len(sames_sn)=} {len(difs_sn)=}')
            M_same = np.mean(sames_sn)
            M_dif = np.mean(difs_sn)
            sames.append(M_same)
            difs.append(M_dif)
            efs.append(M_same - M_dif)
        M_age_same = np.mean(sames)
        M_age_dif = np.mean(difs)
        M_age_ef = np.mean(efs)
        SD_age_ef = np.std(efs)
        SE_age_ef = SD_age_ef / np.sqrt(len(efs))
        t_age_ef = M_age_ef / SE_age_ef
        p_age_ef = stats.t.sf(np.abs(t_age_ef), len(efs)-1)*2
        print(f'Subject specific effect ({age}) | same: {M_age_same:.3f}, '
              f'dif: {M_age_dif:.3f}, t = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        sames_all.append(sames)
        difs_all.append(difs)
        efs_all.append(efs)
    t, p = stats.ttest_ind(efs_all[0], efs_all[1])
    print(f'\tYoung vs. Old: t = {t:.3f}, p = {p:.3f}')

def subject_specific(threshold=0.9):
    fp = 'obj3_fMRI'
    fp = 'scn3_fMRI'
    # fp = 'bl3_fMRI'
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'hit_hit',
    #           'key_vals': (True, False),
    #           'odd_even': True}

    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': True}

    sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,
                                                 load_FC_for_Lifu, kwargs=kwargs,
                                                 verbose=1, easy_override=False,
                                                 cache_dir='cache')
    fp = 'cache/test.pkl'
    partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
        sn_conn, plot=False, threshold=threshold), easy_override=False)
    # partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
    #                                                 threshold=threshold)
    # partitions = [[0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 62, 164, 166, 167, 176, 177, 178, 179, 186, 187, 232], [48, 49, 68, 69, 76, 77, 78, 80, 82, 83, 86, 87, 88, 89, 92, 93, 94, 95, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 165, 210, 211, 212, 213, 214, 215, 216, 217], [6, 7, 8, 9, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 138, 139, 147, 148, 149, 154, 155, 158, 159, 160, 161, 182, 183, 184], [81, 84, 85, 90, 91, 96, 97, 98, 99, 100, 101, 104, 105, 106, 107, 134, 135, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209], [218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245], [60, 61, 70, 71, 72, 73, 74, 75, 79, 120, 121, 122, 123, 144, 145, 156, 157, 162, 163, 168, 169, 170, 171, 172, 173], [136, 137, 140, 141, 142, 143, 146, 150, 151, 152, 153, 174, 175, 180, 181, 185]]

    # partitions = [list(range(246)) for i in range(2)]

    sn_inc_conn[..., ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}

    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        print('-' * 50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        p_mat = get_partition_matrix(sn_inc_conn, p)
        analyze_subject_specific(p_mat, age2idxs)


def INC_reconfig(threshold=0.9):
    fp = 'obj3_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              # 'key': 'rand',
              'key_vals': (1, 2, 3)}
    sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,
                                                 load_FC_for_Lifu, kwargs=kwargs,
                                                 verbose=1, easy_override=False,
                                                 cache_dir='cache')
    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}
    ps_good = [partitions[i] for i in i2name[threshold]]


    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        print('-' * 50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                        title=f'{name}, Inc vs Neu',
                        col0=0, col1=1)
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                        title=f'{name}, Inc vs Con',
                        col0=0, col1=2)
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                        title=f'{name}, Neu vs Con',
                        col0=1, col1=2)



def rand_test(threshold=0.9, way3=False):
    difs = []
    ds = []
    for nsim in range(100):
        fp = 'obj3_fMRI'
        kwargs = {'fp': fp, 'split': False,
                  # 'key': 'inc',
                  'key': 'rand',
                  'key_vals': (1, 2, 3) if way3 else (False, True)}
        sn_inc_conn, sn_conn, age2idxs = pickle_wrap(None,
                                                     load_FC_for_Lifu, kwargs=kwargs,
                                                     verbose=1, easy_override=True,
                                                     cache_dir='cache')
        partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                        threshold=threshold)
        sn_inc_conn[:, :, ~top_edges_mat] = np.nan
        i2name = {0.9: {1: 'MTL+'}}
        ps_good = [partitions[i] for i in i2name[threshold]]

        for i, p in enumerate(partitions):
            if i not in i2name[threshold]:
                continue
            if len(p) < 5:
                continue
            print('-' * 50)
            name = i2name[threshold][i]
            print(f'Partition: {name} ({i})')
            M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                     title=f'{name}, Inc vs Neu',
                                     col0=0, col1=1, plot=False)
            ds.append(d)
            difs.append(dif)
            if way3:
                M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                         title=f'{name}, Inc vs Con',
                                         col0=0, col1=2, plot=False)
                ds.append(d)
                difs.append(dif)
                M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                         title=f'{name}, Neu vs Con',
                                         col0=1, col1=2, plot=False)
                ds.append(d)
                difs.append(dif)
            print(f'{len(ds)}')
            print(f'{np.mean(ds)=} [{np.std(ds)=})]')
            print(f'{np.mean(difs)=} [{np.std(difs)=})]')

if __name__ == '__main__':
    THRESHOLD = 0.9
    # Fig15_Table2_analyses(THRESHOLD)
    # quit()
    # Fig16a_analyses(THRESHOLD, memory_type='vis_hit')
    # quit()
    # Fig16a_analyses(THRESHOLD, memory_type='con_hit')
    # Fig16a_analyses(THRESHOLD, memory_type='hit_hit')

    # Fig17_analyses(THRESHOLD)
    # INC_reconfig(THRESHOLD)
    # rand_test(THRESHOLD)
    subject_specific()