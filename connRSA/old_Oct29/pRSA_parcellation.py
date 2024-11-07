from collections import defaultdict

import utils
from Utils.atlas_funcs import get_atlas
from connRSA.single_trial_conn import prep_fps, prep_vecs
from Study1A.modularity_funcs import get_partition_matrix
from Utils.plotting_funcs import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_stim_RDM
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from functools import cache
from numba import jit, njit
from tqdm import tqdm
from scipy import stats
from nilearn import image
from nilearn import plotting

NAN_NUM = -999

@cache
def prep_ventral_stream_atlas(atlas_name, num_neighbors=8):
    if atlas_name in ['whole']:
        atlas = get_atlas(lifu_labels=False)
        ventral_regions = set(atlas['ROI_regions'])
    elif atlas_name in ['OG', 'OG_fix', 'OC_IT']:
        atlas = get_atlas(lifu_labels=False)
        ventral_regions = {'Cun', 'OcG', 'ITG', 'FuG', 'PhG',}
    elif atlas_name in ['OC_IT_split']:
        atlas = get_atlas(lifu_labels=False, split=True)
        ventral_regions = {'Cun', 'OcG', 'ITG', 'FuG', 'PhG',}
    elif atlas_name in ['OC_T_split']:
        atlas = get_atlas(lifu_labels=False, split=True)
        ventral_regions = {'Cun', 'OcG', 'ITG', 'FuG', 'PhG', 'MTG', 'STG'}
    elif atlas_name in ['OG_big', 'OG_big_fix', 'OC_T']:
        atlas = get_atlas(lifu_labels=False)
        ventral_regions = {'Cun', 'OcG', 'ITG', 'FuG', 'PhG', 'MTG', 'STG'}
    else:
        # ventral_regions = {'EVC', 'LOC', 'sOcG', 'ITG', 'FuG', 'PhG',} # 'ATL', 'MTG'
        raise ValueError

    ventral_roi_idxs = [idx for idx, region in enumerate(atlas['ROI_regions'])
                        if region in ventral_regions]
    ventral_roi_idxs = np.array(ventral_roi_idxs)
    # print(ventral_roi_idxs)
    # quit()
    maps = atlas['maps'].get_fdata()
    maps_new = np.full_like(maps, np.nan)
    for i, idx in enumerate(ventral_roi_idxs):
        if atlas_name in ['OG', 'OG_big']:
            maps_new[maps == (idx)] = i
        else:
            maps_new[maps == (idx + 1)] = i # proper

    neighbors = {}
    region2new_i = defaultdict(list)
    for new_i, i in enumerate(ventral_roi_idxs):
        coord_i = atlas['coords'][i]
        region = atlas['ROI_regions'][i]
        region2new_i[region].append(new_i)
        dists_i = []
        for j in ventral_roi_idxs:
            # if i == j:
            #     continue
            coord_j = atlas['coords'][j]
            dist = distance.euclidean(coord_i, coord_j)
            dists_i.append(dist)
        closest_8 = np.argsort(dists_i)[:num_neighbors]
        neighbors[new_i] = list(closest_8)

    ventral_roi_idxs = list(range(len(ventral_roi_idxs)))

    maps_new_flat = np.reshape(maps_new, -1, order='F')
    roi2locs = {}
    for idx in ventral_roi_idxs:
        roi2locs[idx] = np.where(maps_new_flat == idx)[0]

    regions_ordered = sorted(ventral_regions, key=lambda x: atlas['ROI_regions'].index(x))
    ticks = []
    tick_labels = []
    tick_lows = []
    for region in regions_ordered:
        l = region2new_i[region]
        ticks.append(np.mean(l))
        tick_labels.append(region)
        tick_lows.append(l[0])

    return maps_new, roi2locs, ventral_roi_idxs, neighbors, ticks, tick_labels, tick_lows


def get_sns_good():
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    bad_sns += ['212'] # missing in bl7_fMRI?
    sns_good = [sn for sn in sns if sn not in bad_sns]
    # sns_good = sns_good[:3]
    return sns_good


def prep_ventral_stream(fps, sanity=False, atlas_name='OG'):
    _, roi2locs, ventral_roi_idxs, _, _, _, _ = prep_ventral_stream_atlas(atlas_name)
    sns_good = get_sns_good()

    roi2ar = {idx: [] for idx in ventral_roi_idxs}
    for i, fp in enumerate(fps):
        for idx in ventral_roi_idxs:
            roi2ar[idx].append([])

        for j, sn in tqdm(enumerate(sns_good), desc=f'Prepping ventral ({fp})'):
            df_sn = get_trial_info(sn, easy_override=False)
            df_sn.sort_values(by='obj', inplace=True)
            img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp])
            img_flat = np.reshape(img, (-1, img.shape[-1]),
                                  order='F')
            if sanity:
                assert (img[1, 0, 0] == img_flat[1, :114]).all()

            for idx in ventral_roi_idxs:
                locs = roi2locs[idx]
                ar = img_flat[locs, :]
                roi2ar[idx][i].append(ar)


    for idx, ar in roi2ar.items():
        roi2ar[idx] = np.array(ar)
    return roi2ar, ventral_roi_idxs

@jit(fastmath=True, nopython=True, cache=True, parallel=False)
def make_RSMs(ROI_trial_vec, is_na, spearman):
    num_ROIS = ROI_trial_vec.shape[0]
    num_trials = ROI_trial_vec.shape[1]
    ROI_size = ROI_trial_vec.shape[2]

    if spearman:
        ROI_trial_vec_ranked = np.full(ROI_trial_vec.shape, NAN_NUM, dtype=int)
        for i in range(num_ROIS):
            for j in range(num_trials):
                ROI_trial_vec_ranked[i, j, :ROI_size] = np.argsort(
                    np.argsort(ROI_trial_vec[i, j, :ROI_size]))
        ROI_trial_vec = ROI_trial_vec_ranked

    out = np.full((num_ROIS, num_trials, num_trials), NAN_NUM, dtype=int)
    for i in range(num_ROIS):
        for j in range(num_trials):
            for k in range(num_trials):
                if j == k:
                    out[i, j, k] = 1
                    continue
                elif j < k:
                    continue
                prod_sum = 0
                j_sum = 0
                jj_sum = 0
                k_sum = 0
                kk_sum = 0
                num_points = 0
                for l in range(ROI_size):
                    if is_na[i, j, l] or is_na[i, k, l]:
                        continue
                    prod_sum += (ROI_trial_vec[i, j, l] *
                                 ROI_trial_vec[i, k, l])
                    jj_sum += ROI_trial_vec[i, j, l] ** 2
                    kk_sum += ROI_trial_vec[i, k, l] ** 2
                    j_sum += ROI_trial_vec[i, j, l]
                    k_sum += ROI_trial_vec[i, k, l]
                    num_points += 1
                if num_points == 0: continue
                if np.abs(prod_sum) < 1e-12: # all zeroes in one
                    continue

                E_JK = prod_sum / num_points
                E_J = j_sum / num_points
                E_K = k_sum / num_points
                numerator = E_JK - (E_J * E_K)
                E_JJ = jj_sum / num_points
                E_KK = kk_sum / num_points
                denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
                if np.abs(denominator) < 1e-12: # basically zero
                    continue

                out[i, j, k] = numerator / denominator
                out[i, k, j] = numerator / denominator
    return out

def get_RSM_stims(fps, semantic=True, dist='corr'):
    d_vecs = prep_vecs(True, semantic)
    sns_good = get_sns_good()
    # fps = fps[1:]

    # fp_sn2RSM = {}
    # fp_sn2RSM
    fp_sn2RSM = []
    for k, fp in enumerate(fps):
        sess = fp.replace('7', '').replace('8', '').replace('_fMRI', '')
        l = []
        for j, sn in enumerate(sns_good):
            df_sn = get_trial_info(sn, easy_override=False)
            # df_sn_ = df_sn.sort_values(by=f'{sess}_trial')
            # print(df_sn_['obj'].values)
            df_sn.sort_values(by='obj', inplace=True)
            df_sn[f'{sess}_trial'] = np.argsort(df_sn[f'{sess}_trial'])
            block_length = (df_sn[f'{sess}_trial'].max() + 1) // 3
            df_sn[f'{sess}_block'] = (df_sn[f'{sess}_trial']) // block_length

            # make a matrix where element is 1 if same block and 0 otherwise
            block_mat = np.zeros((df_sn.shape[0], df_sn.shape[0]))
            for i in range(df_sn.shape[0]):
                block_mat[i] = df_sn[f'{sess}_block'] == df_sn[f'{sess}_block'].iloc[i]
            block_mat = block_mat.astype(int)

            RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                                    dist=dist)
            # plt.title(f'{fp}, {sn}')
            # plt.imshow(RSM_stim)
            # plt.show()
            RSM_stim[block_mat == 1] = np.nan
            l.append(RSM_stim)
            # fp_sn2RSM[(k, j)] = RSM_stim
        fp_sn2RSM.append(l)
    fp_sn2RSM = np.array(fp_sn2RSM)
    return fp_sn2RSM

@njit(parallel=False, cache=True)
def rowwise_corr(X, Y):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    corr = np.empty(n_rows)  # store the correlations

    for i in range(n_rows):  # parallel loop
        # Get current rows
        x_row = X[i, :]
        y_row = Y[i, :]

        # Filter out NaNs
        mask = np.zeros(n_cols, dtype=np.bool_)
        # can't do fastmath because that prevents the NaN checking
        for j in range(n_cols):
            if x_row[j] == x_row[j] and y_row[j] == y_row[j]:  # Check if NaN
                mask[j] = True

        x_filtered = x_row[mask]
        y_filtered = y_row[mask]

        # Compute correlation if there are enough values
        if len(x_filtered) > 1:
            corr[i] = np.corrcoef(x_filtered, y_filtered)[0, 1]
        else:
            corr[i] = np.nan  # Not enough data to compute correlation

    return corr


@njit(parallel=False, fastmath=False)
def rowwise_standardized_regression(A, B, C, Y):
    n_rows = A.shape[0]
    n_cols = A.shape[1]
    betas = np.full((n_rows, 3), np.nan)  # Store betas (3 predictors: A, B, C)

    for i in range(n_rows):
        # Get current rows
        a_row = A[i, :]
        b_row = B[i, :]
        c_row = C[i, :]
        y_row = Y[i, :]

        # Filter out NaNs
        mask = ((y_row == y_row) & (a_row == a_row) & (b_row == b_row) & (c_row == c_row) &
                (y_row != NAN_NUM) & (a_row != NAN_NUM) & (b_row != NAN_NUM) & (c_row != NAN_NUM))
        a_filtered = a_row[mask]
        b_filtered = b_row[mask]
        c_filtered = c_row[mask]
        y_filtered = y_row[mask]

        # Proceed if we have at least 4 valid data points (1 point per predictor)
        if len(y_filtered) > 3:
            # Standardize each variable (A, B, C, and Y)
            a_std = (a_filtered - np.mean(a_filtered)) / np.std(a_filtered)
            b_std = (b_filtered - np.mean(b_filtered)) / np.std(b_filtered)
            c_std = (c_filtered - np.mean(c_filtered)) / np.std(c_filtered)
            y_std = (y_filtered - np.mean(y_filtered)) / np.std(y_filtered)
            # y_std = y_filtered

            X = np.empty((len(a_filtered), 3))
            X[:, 0] = a_std
            X[:, 1] = b_std
            X[:, 2] = c_std
            # X[:, 3] = np.ones(X.shape[0])  # Intercept... not needed... zzzz

            # Calculate the betas using the normal equation: beta = (X^T X)^-1 X^T y
            XtX_inv = np.linalg.inv(X.T @ X)
            XtY = X.T @ y_std
            betas[i, :] = XtX_inv @ XtY  # Store betas for this row

    return betas

def eval_pair(idx2ar_F, idx2is_na, idx0, idx1, fp_sn2RSM, idx2RSMs,
              spearman=False, regr=True, M_ROI=False):
    ar_F0 = idx2ar_F[idx0]
    is_na0 = idx2is_na[idx0]
    if idx0 in idx2RSMs:
        RSMs0 = idx2RSMs[idx0]
    else:
        RSMs0 = make_RSMs(ar_F0, is_na0, spearman)
        idx2RSMs[idx0] = RSMs0
    tril_idxs = np.tril_indices(RSMs0.shape[-1], k=-1)
    RSMs0 = RSMs0[:, tril_idxs[0], tril_idxs[1]]

    ar_F1 = idx2ar_F[idx1]
    is_na1 = idx2is_na[idx1]
    if idx1 in idx2RSMs:
        RSMs1 = idx2RSMs[idx1]
    else:
        RSMs1 = make_RSMs(ar_F1, is_na1, spearman)
        idx2RSMs[idx1] = RSMs1
    RSMs1 = RSMs1[:, tril_idxs[0], tril_idxs[1]]

    ar_F_both = np.concatenate((ar_F0, ar_F1), axis=-1)
    print(ar_F_both.shape)
    if M_ROI:
        ar_F0_M = np.nanmean(ar_F0, axis=2, keepdims=True)
        ar_F1_M = np.nanmean(ar_F1, axis=2, keepdims=True)
        ar_F_both_M = np.concatenate((ar_F0_M, ar_F1_M), axis=-1)
        RSM_pre_i = ar_F_both_M[:, :, None, 0] > ar_F_both_M[:, None, :, 0]
        RSM_pre_j = ar_F_both_M[:, :, None, 1] > ar_F_both_M[:, None, :, 1]
        RSMs_both = (RSM_pre_i == RSM_pre_j).astype(int)
    else:
        is_na_both = np.concatenate((is_na0, is_na1), axis=-1)
        RSMs_both = make_RSMs(ar_F_both, is_na_both, spearman)
    RSMs_both = RSMs_both[:, tril_idxs[0], tril_idxs[1]]

    if regr:
        # print(RSMs_both.shape)
        # quit()
        try:
            betas = rowwise_standardized_regression(RSMs_both, RSMs0, RSMs1, fp_sn2RSM)
            return betas
        except np.linalg.LinAlgError:
            num_nans = np.sum(np.isnan(RSMs_both))
            num_non_nans = np.sum(~np.isnan(RSMs_both))
            print(f'{num_nans=}, {num_non_nans=}')
            # num_elements = np.count_nonzero()
            return np.full((RSMs0.shape[0], 3), np.nan)
        # return betas
    else:

        RSMs_M = np.nanmean(np.stack((RSMs0, RSMs1)), axis=0)
        corr_M = rowwise_corr(RSMs_M, fp_sn2RSM)
        corr_both = rowwise_corr(RSMs_both, fp_sn2RSM)
        return corr_M, corr_both

def get_idx2(fps, atlas_name):
    kw = {'fps': fps, 'atlas_name': atlas_name}
    roi2ar, ventral_roi_idxs = pickle_wrap(prep_ventral_stream,
                                           kwargs=kw, verbose=0,
                                           easy_override=False)
    idx2ar_F = {}
    idx2is_na = {}
    for idx, ar in tqdm(roi2ar.items(), desc='organizing ar_F'):
        ar = np.array(ar)
        if ar.shape[2] == 0:
            ar = np.full((ar.shape[0], ar.shape[1], 1, ar.shape[-1]), np.nan)
            # idx2ar_F[idx] = ar_F
            # idx2is_na[idx] = np.full(ar.shape[:2], True)

        ar_F = np.reshape(ar, (-1, ar.shape[-2], ar.shape[-1]),
                          order='F')
        ar_F = np.transpose(ar_F, (0, 2, 1))
        is_na = np.isnan(ar_F)
        ar_F = np.nan_to_num(ar_F, nan=NAN_NUM)
        idx2is_na[idx] = is_na
        idx2ar_F[idx] = ar_F
    return idx2ar_F, idx2is_na

# OG_big_fix
def analyze_ventral_stream(semantic=True,
                           atlas_name='OC_T',
                           # atlas_name='whole',
                           spearman=True, M_ROI=True,
                           regr=True, neighbor_mask=False):

    fps = prep_fps('7')
    kw = {'fps': fps, 'semantic': semantic}
    fp_sn2RSM = pickle_wrap(get_RSM_stims, kwargs=kw, verbose=0,
                            easy_override=False)
    fp_sn2RSM = np.reshape(fp_sn2RSM, (fp_sn2RSM.shape[0] * fp_sn2RSM.shape[1],
                                       fp_sn2RSM.shape[2], fp_sn2RSM.shape[3]))
    tril_idxs = np.tril_indices(fp_sn2RSM.shape[-1], k=-1)
    fp_sn2RSM = fp_sn2RSM[:, tril_idxs[0], tril_idxs[1]]

    kw = {'fps': fps, 'atlas_name': atlas_name}
    idx2ar_F, idx2is_na = pickle_wrap(get_idx2, kwargs=kw, verbose=0,
                                      easy_override=False)

    maps_new, _, ventral_roi_idxs, neighbors, ticks, tick_labels, tick_lows = (
        prep_ventral_stream_atlas(atlas_name, num_neighbors=32))

    st_oc = tick_lows[-2]


    idx2RSMs = {}
    conn_RSA = np.full((len(ventral_roi_idxs), len(ventral_roi_idxs)), np.nan)
    for idx0 in tqdm(ventral_roi_idxs, desc='evaluating pairs'):
        for idx1 in range(idx0):
            # if idx0 >= st_oc or idx1 >= st_oc: continue
            if neighbor_mask:
                if idx1 not in neighbors[idx0] and idx0 not in neighbors[idx1]:
                    continue
            regr_str = '_regr' if regr else ''
            M_roi_str = '_M_ROI' if M_ROI else ''
            fp_pkl = (rf'C:\PycharmProjects\SchemeRep\cache\pairRSA\{idx0}_{idx1}_{atlas_name}'
                      rf'_{spearman}_{semantic}{regr_str}{M_roi_str}.pkl')
            out = pickle_wrap(lambda: eval_pair(idx2ar_F, idx2is_na, idx0, idx1,
                                                fp_sn2RSM, idx2RSMs, spearman,
                                                regr=regr, M_ROI=M_ROI),
                                            fp_pkl, verbose=0,
                                   easy_override=False)
            if regr:
                # dif = ((out[:, 0] -  0.5 * (out[:, 1] + out[:, 2]))
                #        / (out[:, 0] + 0.5 * (out[:, 1] + out[:, 2])))
                # dif = out[:, 0] - 0.5 * (out[:, 1] + out[:, 2])
                # dif = 0.5 * (out[:, 1] + out[:, 2])
                dif = out[:, 0] - 0.5 * (out[:, 1] + out[:, 2])
            else:
                corr_M, corr_both = out[0], out[1]
                dif = corr_both# - corr_M
                # dif = corr_both
            nans = np.isnan(dif)
            dif = dif[~nans]
            # both_adv = stats.trim_mean(dif, .1) / stats.mstats.trimmed_std(dif, .1)
            both_adv = stats.trim_mean(dif, .1)


            conn_RSA[idx0, idx1] = both_adv
            conn_RSA[idx1, idx0] = both_adv


    corr_str = 'Spearman' if spearman else 'Pearson'
    title = f'Ventral stream RSA {corr_str}'

    plot_connectivity(conn_RSA, ticks, tick_labels, tick_lows, title=title,
                      tile=.1, dontoverride=True, adjust_HC_AMY=False)
    # quit()


    # p = [64, 65] are LOC

    med = np.nanquantile(conn_RSA, .8)
    print(f'{med=:.3f}')
    bool_RSA = (conn_RSA > med).astype(int)

    neighbors_mat = np.zeros_like(bool_RSA)
    for i in range(neighbors_mat.shape[0]):
        neighbors_mat[i, neighbors[i]] = 1
    plot_connectivity(neighbors_mat, ticks, tick_labels, tick_lows, title='neighbors',
                      tile=.1, dontoverride=True, adjust_HC_AMY=False)
    # quit()

    # bool_RSA *= neighbors_mat
    plot_connectivity(bool_RSA, ticks, tick_labels, tick_lows, title=title,
                      tile=.1, dontoverride=True, adjust_HC_AMY=False)


    partitions = get_modules((bool_RSA))
    print(len(partitions))
    atlas = get_atlas(lifu_labels=False)

    for p in partitions:

        # p = [24, 14, 39, 33, 0]
        if len(p) < 3:
            continue
        corr_v0 = get_partition_matrix(bool_RSA, p,
                                       w_zeros=True)
        # plot_connectivity(corr_v0, ticks, tick_labels, tick_lows, title=title,
        #                   tile=.1,  adjust_HC_AMY=False, dontoverride=True)
        # corr_v00 = corr_v0
        # corr_v00[::2] = 0
        # corr_v00[:, ::2] = 0
        # plot_connectivity(corr_v00, ticks, tick_labels, tick_lows, title=title,
        #                     tile=.1, dontoverride=True, adjust_HC_AMY=False)

        # corr_v01 = corr_v0
        # corr_v01[1::2] = 0
        # corr_v01[:, 1::2] = 0
        # plot_connectivity(corr_v01, ticks, tick_labels, tick_lows, title=title,
        #                     tile=.1, dontoverride=True)
        #
        # quit()
        img_plot = np.zeros_like(maps_new)
        for i, idx in enumerate(p):
            img_plot[maps_new == idx] = i + 1

        # plt.imshow(img_plot[:, 25, :], cmap='tab20')
        # plt.show()
        # quit()

        img_plot = image.new_img_like(atlas['maps'], img_plot)

        plotting.plot_glass_brain(img_plot, cmap='turbo',  # plt.get_cmap(cmap),
                                  black_bg=False, vmin=0.5,
                                  resampling_interpolation='nearest',
                                  )
        plt.show()
        # quit()


def get_modules(matrix_thresh):
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(matrix_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=0)
    return part


if __name__ == '__main__':
    analyze_ventral_stream()