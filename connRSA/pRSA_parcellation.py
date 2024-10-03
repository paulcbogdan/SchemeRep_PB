from collections import defaultdict

import utils
from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_fps, prep_vecs
from networks.old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_stim_RDM
from utils import pickle_wrap
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from functools import cache
from numba import jit, njit, prange
from tqdm import tqdm

NAN_NUM = -999

# atlas = get_atlas(lifu_labels=False)
# quit()
#
# def resel_pairRSA():
#     pass

@cache
def prep_ventral_stream_atlas(atlas_name, num_neighbors=8):
    if atlas_name == 'OG':
        atlas = get_atlas(lifu_labels=False)
        ventral_regions = {'Cun', 'OcG', 'ITG', 'FuG', 'PhG',} #  'MTG'
    else:
        # ventral_regions = {'EVC', 'LOC', 'sOcG', 'ITG', 'FuG', 'PhG',} # 'ATL', 'MTG'
        raise ValueError

    ventral_roi_idxs = [idx for idx, region in enumerate(atlas['ROI_regions'])
                        if region in ventral_regions]
    ventral_roi_idxs = np.array(ventral_roi_idxs)
    maps = atlas['maps'].get_fdata()
    maps_new = np.full_like(maps, np.nan)
    for i, idx in enumerate(ventral_roi_idxs):
        maps_new[maps == idx] = i

    neighbors = {}
    region2new_i = defaultdict(list)
    for new_i, i in enumerate(ventral_roi_idxs):
        coord_i = atlas['coords'][i]
        region = atlas['ROI_regions'][i]
        region2new_i[region].append(new_i)
        dists_i = []
        for j in ventral_roi_idxs:
            if i == j:
                continue
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

    return roi2locs, ventral_roi_idxs, neighbors, ticks, tick_labels, tick_lows


def get_sns_good():
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    bad_sns += ['212'] # missing in bl7_fMRI?
    sns_good = [sn for sn in sns if sn not in bad_sns]
    # sns_good = sns_good[:3]
    return sns_good


def prep_ventral_stream(fps, sanity=False, atlas_name='OG'):
    roi2locs, ventral_roi_idxs, _, _, _, _ = prep_ventral_stream_atlas(atlas_name)
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

@jit(fastmath=True, nopython=True, cache=True, parallel=True)
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
    for i in prange(num_ROIS):
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

@njit(parallel=True, cache=True)
def rowwise_corr(X, Y):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    corr = np.empty(n_rows)  # store the correlations

    for i in prange(n_rows):  # parallel loop
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


def eval_pair(idx2ar_F, idx2is_na, idx0, idx1, fp_sn2RSM, idx2RSMs,
              spearman=False):
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

    RSMs_M = np.nanmean(np.stack((RSMs0, RSMs1)), axis=0)
    corr_M = rowwise_corr(RSMs_M, fp_sn2RSM)
    corr_M_scalar = np.nanmean(corr_M)

    ar_F_both = np.concatenate((ar_F0, ar_F1), axis=-1)
    is_na_both = np.concatenate((is_na0, is_na1), axis=-1)
    RSMs_both = make_RSMs(ar_F_both, is_na_both, spearman)
    RSMs_both = RSMs_both[:, tril_idxs[0], tril_idxs[1]]
    corr_both = rowwise_corr(RSMs_both, fp_sn2RSM)
    corr_both_scalar = np.nanmean(corr_both)
    return corr_both_scalar - corr_M_scalar, idx2RSMs

def analyze_ventral_stream(semantic=True, atlas_name='OG'):
    fps = prep_fps('7')
    kw = {'fps': fps, 'semantic': semantic}
    fp_sn2RSM = pickle_wrap(get_RSM_stims, kwargs=kw, verbose=0,
                            easy_override=False)
    fp_sn2RSM = np.reshape(fp_sn2RSM, (fp_sn2RSM.shape[0] * fp_sn2RSM.shape[1],
                                       fp_sn2RSM.shape[2], fp_sn2RSM.shape[3]))
    tril_idxs = np.tril_indices(fp_sn2RSM.shape[-1], k=-1)
    fp_sn2RSM = fp_sn2RSM[:, tril_idxs[0], tril_idxs[1]]

    kw = {'fps': fps, 'atlas_name': atlas_name}
    roi2ar, ventral_roi_idxs = pickle_wrap(prep_ventral_stream,
                                           kwargs=kw, verbose=0,
                                           easy_override=False)
    idx2ar_F = {}
    idx2is_na = {}
    for idx, ar in tqdm(roi2ar.items(), desc='organizing ar_F'):
        ar = np.array(ar)
        ar_F = np.reshape(ar, (-1, ar.shape[-2], ar.shape[-1]),
                          order='F')
        ar_F = np.transpose(ar_F, (0, 2, 1))
        is_na = np.isnan(ar_F)
        ar_F = np.nan_to_num(ar_F, nan=NAN_NUM)
        idx2is_na[idx] = is_na
        idx2ar_F[idx] = ar_F

    idx2RSMs = {}
    conn_RSA = np.full((len(ventral_roi_idxs), len(ventral_roi_idxs)), np.nan)
    for idx0 in tqdm(ventral_roi_idxs, desc='evaluating pairs'):
        for idx1 in range(idx0):
            # if idx0 == idx1:
            #     continue
            both_adv, idx2RSMs = (
                eval_pair(idx2ar_F, idx2is_na, idx0, idx1, fp_sn2RSM, idx2RSMs))
            conn_RSA[idx0, idx1] = both_adv
            conn_RSA[idx1, idx0] = both_adv
        if idx0 == 5:
            break

    _, _, _, ticks, tick_labels, tick_lows = prep_ventral_stream_atlas(atlas_name)
    plot_connectivity(conn_RSA, ticks, tick_labels, tick_lows, title='Ventral Stream',
                      tile=.01, cbar_label='Pos = distributed / Neg = local')



    # plt.imshow(conn_RSA, interpolation='none')
    # plt.show()



if __name__ == '__main__':
    analyze_ventral_stream()