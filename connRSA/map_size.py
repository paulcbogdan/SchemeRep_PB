import os.path

import numpy as np
from numba import njit, prange
from time import time

from Utils.atlas_funcs import get_atlas
from connRSA.fft_funcs import get_ROIs_from_region
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from org_sns import get_sns
from organize_bhv import get_trial_info
import utils
from tqdm import tqdm
import scipy.stats as stats
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from functools import cache
from nilearn import image
from pathlib import Path

GLOBAL_NAN_VALUE = -999_999_999

    # for the less than/greater than, we just check whether in trial 1, a > b
    #    is similar to, say, trial 2 a > b

    # Note that high RSA correlation when a predictor is 1s and 0s
    #   is similar to just finding a set of 1s where the average similar is high

# @njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA_mat(voxels_mat0, voxels_mat_bool, stim_RSM, median_cond2=False,
                       just_second_vox=False):
    n_voxels0 = voxels_mat0.shape[1]
    same_Ms_mat = np.empty((n_voxels0, n_voxels0))
    stim_is_nan = np.isnan(stim_RSM)
    voxels_mat1 = voxels_mat0 if median_cond2 else voxels_mat_bool

    MI_both_mat = np.empty((n_voxels0, n_voxels0))
    MI_vox0_mat = np.empty((n_voxels0, n_voxels0))
    MI_vox1_mat = np.empty((n_voxels0, n_voxels0))
    ex_both_mat = np.empty((n_voxels0, n_voxels0))
    ex_vox0_mat = np.empty((n_voxels0, n_voxels0))
    ex_vox1_mat = np.empty((n_voxels0, n_voxels0))

    for v in tqdm(range(1, n_voxels0)):
        voxel_l = voxels_mat_bool[:, v]
        MI_both, MI_vox0, MI_vox1, ex_both, ex_vox0, ex_vox1 = (
            run_map_MI_RSA(voxel_l, voxels_mat1, stim_RSM,
                           stim_is_nan, median_cond2=median_cond2,
                           just_second_vox=just_second_vox))
        MI_both_mat[v, :] = MI_both
        MI_vox0_mat[v, :] = MI_vox0
        MI_vox1_mat[v, :] = MI_vox1
        ex_both_mat[v, :] = ex_both
        ex_vox0_mat[v, :] = ex_vox0
        ex_vox1_mat[v, :] = ex_vox1

    return MI_both_mat, MI_vox0_mat, MI_vox1_mat, ex_both_mat, ex_vox0_mat, ex_vox1_mat

@njit(fastmath=True, nopython=True, cache=True)
def run_vox_MI_RSA(voxels_mat_bool, stim_RSM, stim_is_nan):
    n_trials = voxels_mat_bool.shape[0]
    n_voxels = voxels_mat_bool.shape[1]
    same_Ms = np.empty(n_voxels)
    for v in range(n_voxels):
        voxel_l = voxels_mat_bool[:, v]
        same_total = 0
        n_same = 0
        for t0 in range(n_trials):
            for t1 in range(t0):
                if stim_is_nan[t0, t1]:
                    break
                if voxel_l[t0] == voxel_l[t1]:
                    n_same += 1
                    same_total += stim_RSM[t0, t1]
        same_M = same_total / n_same
        same_Ms[v] = same_M
    return same_Ms

def generate_RSA_vox_map(sn, fp, region, semantic, layer,
                         stdize_by_run=True):
    voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM = (
        load_img_rsm(region, fp, sn, semantic, layer,
                     stdize_by_run=stdize_by_run))

    t_st = time()
    stim_is_nan = np.isnan(stim_RSM)
    same_Ms = run_vox_MI_RSA(voxels_mat_bool, stim_RSM, stim_is_nan)
    # same_Ms = run_map_MI_RSA_mat(voxels_mat, voxels_mat, stim_RSM)
    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    same_Ms[same_Ms == GLOBAL_NAN_VALUE] = np.nan

    return same_Ms, valid_voxel_idxs

def generate_RSA_vox_vox_map(sn, fp, region, semantic, layer,
                             stdize_by_run=True,):
    same_Ms, valid_voxel_idxs = generate_RSA_vox_map(sn, fp, region, semantic, layer,
                                                     stdize_by_run=stdize_by_run)

    vox_vox_Ms = generate_RSA_size_map(sn, fp, region, semantic, layer,
                                       median_cond2=True, stdize_by_run=stdize_by_run,
                                       just_second_vox=True)
    vox_vox_Ms = vox_vox_Ms + same_Ms[:, None]
    return vox_vox_Ms


@njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA(voxel_l, voxels_mat, stim_RSM,
                   stim_is_nan, median_cond2=False,
                   just_second_vox=False):
    # break into bins
    n_trials = voxel_l.shape[0]
    assert n_trials == 114
    n_voxels = voxels_mat.shape[1]

    if median_cond2:
        # pass
        center_is_zero = np.argwhere(~voxel_l)
        cnt_zero = center_is_zero.shape[0]
        center_is_one = np.argwhere(voxel_l)
        cnt_one = center_is_one.shape[0]
        voxel_Ms = np.empty((2, n_voxels))
        for v in range(n_voxels):
            t_at_is_zero = np.full(n_trials, 0, dtype=np.float32)
            for i in range(cnt_zero):
                t = center_is_zero[i][0]
                t_at_is_zero[i] = voxels_mat[t, v]
            voxel_Ms[0, v] = np.median(t_at_is_zero[:cnt_zero])

            t_at_is_one = np.full(n_trials, 0, dtype=np.float32)
            for i in range(cnt_one):
                t = center_is_one[i][0]
                t_at_is_one[i] = voxels_mat[t, v]
            voxel_Ms[1, v] = np.median(t_at_is_one[:cnt_one])

    t2vox0 = np.empty(n_trials, dtype=np.int8)
    for t in range(n_trials):
        t2vox0[t] = np.int8(voxel_l[t])

    MI_both = np.empty(n_voxels, dtype=np.float32)
    MI_vox0 = np.empty(n_voxels, dtype=np.float32)
    MI_vox1 = np.empty(n_voxels, dtype=np.float32)

    ex_both = np.empty(n_voxels, dtype=np.int16)
    ex_vox0 = np.empty(n_voxels, dtype=np.int16)
    ex_vox1 = np.empty(n_voxels, dtype=np.int16)

    # trial2cond_cond = np.empty(n_trials, dtype=np.int8)
    # t2vox1 = np.empty(n_trials, dtype=np.int8)
    for v in range(n_voxels):
        # continue
        t2vox1 = np.empty(n_trials, dtype=np.int8)
        for t in range(n_trials):
            # continue
            # vox0 = np.int8(voxel_l[t] == 1)
            if median_cond2:
                vox0 = t2vox0[t]
                t2vox1[t] = voxels_mat[t, v] > voxel_Ms[vox0, v]
                t2vox1[t] = np.int8(t2vox1[t])
                # print(f'{vox0}: {vox1} | {voxels_mat[t, v]:.3f} < {voxel_Ms[vox0, v]:.3f}')
                # if just_second_vox:
                #     trial2cond_cond[t] = t2vox1[t]
                # else:
                #     trial2cond_cond[t] = vox0 * 2 + t2vox1[t]
            else:
                # same_Ms[v] = voxels_mat[t, v] * 5
                t2vox1[t] = np.int8(voxels_mat[t, v])
                # trial2cond_cond[t] = t2vox0[t] * 2 + t2vox1[t]


        MI_both_voxel = 0
        MI_voxel0 = 0
        MI_voxel1 = 0
        n_same = 0
        n_same0 = 0
        n_same1 = 0
        for t0 in range(n_trials):
            for t1 in range(t0):
                if stim_is_nan[t0, t1]:
                    break
                if t2vox0[t0] == t2vox0[t1]:
                    n_same0 += 1
                    MI_voxel0 += stim_RSM[t0, t1]
                    if t2vox1[t0] == t2vox1[t1]:
                        n_same1 += 1
                        n_same += 1
                        MI_voxel1 += stim_RSM[t0, t1]
                        MI_both_voxel += stim_RSM[t0, t1]
                else:
                    if t2vox1[t0] == t2vox1[t1]:
                        n_same1 += 1
                        MI_voxel1 += stim_RSM[t0, t1]

        ex_both[v] = n_same
        ex_vox0[v] = n_same0
        ex_vox1[v] = n_same1

        if n_same == 0:
            MI_both[v] = GLOBAL_NAN_VALUE
        else:
            MI_both[v] = MI_both_voxel / n_same
        if n_same0 == 0:
            MI_vox0[v] = GLOBAL_NAN_VALUE
        else:
            MI_vox0[v] = MI_voxel0 / n_same0
        if n_same1 == 0:
            MI_vox1[v] = GLOBAL_NAN_VALUE
        else:
            MI_vox1[v] = MI_voxel1 / n_same1

    return MI_both, MI_vox0, MI_vox1, ex_both, ex_vox0, ex_vox1

def get_img_region(region, fp, sn):
    if '_L' in region:
        L_R = 'L'
        region = region.replace('_L', '')
    elif '_R' in region:
        L_R = 'R'
        region = region.replace('_R', '')
    else:
        L_R = None

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp])

    ROIs = get_ROIs_from_region(region)
    if L_R:
        ROIs = [ROI for ROI in ROIs if f'_{L_R}_' in ROI]

    ROI_idxs = [int(roi.split(' ')[0]) for roi in ROIs]

    atlas = get_atlas(combine_regions=False, combine_bilateral=False,)
    atlas_data = atlas['maps'].get_fdata()
    mask = np.zeros(atlas_data.shape)
    for idx in ROI_idxs:
        mask[atlas_data == idx] = 1
    img[mask < 0.5, :] = np.nan
    return img

def load_img_rsm(region, fp, sn, semantic, layer, easy_override=False,
                 stdize_by_run=True):

    img = get_img_region(region, fp, sn)
    valid_voxel_idxs = np.argwhere(~np.isnan(img[:, :, :, 0]))

    voxels_mat = img[valid_voxel_idxs[:, 0], valid_voxel_idxs[:, 1],
                     valid_voxel_idxs[:, 2], :]
    voxels_mat = voxels_mat.T


    if stdize_by_run:
        voxels_mat = utils.stdize(voxels_mat, stdize_by_run=True, axis=0)

    voxels_mat_bool = voxels_mat > np.nanmedian(voxels_mat, axis=0)


    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    return voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM

def generate_RSA_size_map(sn, fp, region, semantic, layer,
                          median_cond2=False, stdize_by_run=True,
                          just_second_vox=False):
    voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM = (
        load_img_rsm(region, fp, sn, semantic, layer, stdize_by_run=stdize_by_run))

    has_nans = []
    for v in range(voxels_mat_bool.shape[1]):
        cnt_nan = np.sum(np.isnan(voxels_mat[:, v]))
        has_nans.append(cnt_nan > 0)
    has_nans = np.array(has_nans)

    t_st = time()
    MI_both_mat, MI_vox0_mat, MI_vox1_mat, ex_both_mat, ex_vox0_mat, ex_vox1_mat = (
        run_map_MI_RSA_mat(voxels_mat, voxels_mat_bool, stim_RSM,
                           median_cond2=median_cond2, just_second_vox=just_second_vox))
    # same_Ms = np.float32(same_Ms)
    # MI_both_mat = np.float32(MI_both_mat)
    # MI_vox0_mat = np.float32(MI_vox0_mat)
    # MI_vox1_mat = np.float32(MI_vox1_mat)
    # ex_both_mat = np.float32(ex_both_mat)
    # ex_vox0_mat = np.float32(ex_vox0_mat)
    # ex_vox1_mat = np.float32(ex_vox1_mat)

    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    MI_both_mat[MI_both_mat == GLOBAL_NAN_VALUE] = np.nan
    MI_vox0_mat[MI_vox0_mat == GLOBAL_NAN_VALUE] = np.nan
    MI_vox1_mat[MI_vox1_mat == GLOBAL_NAN_VALUE] = np.nan

    return (MI_both_mat, MI_vox0_mat, MI_vox1_mat, ex_both_mat, ex_vox0_mat, ex_vox1_mat,
            valid_voxel_idxs, has_nans)




def get_dist2MI(sn, fp, semantic, layer, region, subtract_vox, max_vox,
                subtract_other=True, median_cond2=False, stdize_by_run=True):
    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_M = np.nanmean(stim_RSM) * 10_000
    print(f'Doing: {sn}, {fp}')
    kw = {'sn': sn, 'fp': fp, 'region': region, 'semantic': semantic,
          'layer': layer, 'stdize_by_run': stdize_by_run,
          'median_cond2': median_cond2}

    (MI_both_mat, MI_vox0_mat, MI_vox1_mat, ex_both_mat, ex_vox0_mat, ex_vox1_mat,
     valid_voxel_idxs, has_nans) = utils.pickle_wrap(generate_RSA_size_map, None,
                                                     kwargs=kw, verbose=-1,
                                                     easy_override=False)

    same_Ms = MI_both_mat


    if subtract_vox:
        # check for beng extra close to zero: Doing: 138, vis7_fMRI | 4.121147867408581e-13
        kw = {'sn': sn, 'fp': fp, 'region': region, 'semantic': semantic,
              'layer': layer, 'stdize_by_run': stdize_by_run}
        vox_Ms, valid_voxel_idxs2 = utils.pickle_wrap(generate_RSA_vox_map, None,
                                                      kwargs=kw,
                                                      verbose=-1, easy_override=False)
        vox_Ms *= 10_000
        for i in range(0, valid_voxel_idxs2.shape[0], 100):
            assert tuple(valid_voxel_idxs2[i]) == tuple(valid_voxel_idxs[i]), 'Mismatch idxs'
        vox_Ms -= np.nanmean(stim_M)
        vox_Ms = vox_Ms[:, None] + vox_Ms[None, :]
        same_Ms -= vox_Ms
    elif subtract_other:
        same_Ms_m = np.nanmean(same_Ms, axis=0)
        same_Ms_m_subber = (same_Ms_m[None, :] + same_Ms_m[:, None]) / 2
        same_Ms -= same_Ms_m_subber

        # print(same_Ms.shape)
        # quit()
        # pass

    has_nan = np.any(np.isnan(same_Ms))
    if has_nan:
        M_M = np.nanmean(same_Ms)
        same_Ms[np.isnan(same_Ms)] = M_M
        has_nan_str = ' (has nan)'
    else:
        M_M = np.mean(same_Ms)
        has_nan_str = ''
    dif = M_M - stim_M
    print(f'\t{sn}, {fp}, {stim_M=:.5f} | {dif=:.5f}{has_nan_str}')
    t_st = time()
    euc_mtx = get_idx2euc_custom(valid_voxel_idxs)

    print(f'Time needed for euc_mtx: {time() - t_st:.2f} s')
    t_st = time()

    vox_dist2MI, cnter = get_voxel_dist2MI(same_Ms - stim_M, euc_mtx, max_vox=max_vox)

    vox_dist2MI[vox_dist2MI == GLOBAL_NAN_VALUE] = np.nan
    dist2MI = np.nanmean(vox_dist2MI, axis=0)
    print(f'Time needed for dist2MI: {time() - t_st:.2f} s')

    print(f'Time needed for dist2MI: {time() - t_st:.2f} s')
    return vox_dist2MI, dist2MI, valid_voxel_idxs, cnter

def run_RSA_map_all_sn(region, semantic, layer, max_vox=55, subtract_vox=True,
                       median_cond2=True, stdize_by_run=True):
    if 'OC_IT' in region:
        max_vox = 100
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI']#, 'con7_fMRI', 'vis7_fMRI']
    sns = sns[0::4]
    dist2MI_all = []
    for i, sn in enumerate(sns):
        sn_dist2MI = []
        for j, fp in enumerate(fps):
            vox_dist2MI, dist2MI, _, _ = utils.pickle_wrap(get_dist2MI, None,
                                                     kwargs={'sn': sn, 'fp': fp,
                                                             'semantic': semantic,
                                                             'layer': layer,
                                                             'region': region,
                                                             'subtract_vox': subtract_vox,
                                                             'max_vox': max_vox,
                                                             'median_cond2': median_cond2,
                                                             'stdize_by_run': True},
                                                     verbose=-1, easy_override=False)

            # vox_dist2MI, dist2MI = get_dist2MI(sn, fp, semantic, layer, region,
            #                                    subtract_vox, max_vox)

            sn_dist2MI.append(dist2MI)
        # continue
        sn_dist2MI = np.nanmean(np.array(sn_dist2MI), axis=0)
        dist2MI_all.append(sn_dist2MI)

        M = np.nanmean(dist2MI_all, axis=0)
        super_M = np.nanmean(M)
        print(f'{super_M=}')
        SE = np.nanstd(dist2MI_all, axis=0) / np.sqrt(len(dist2MI_all))
        t = M / SE
        for dist in range(max_vox):
            print(f'{dist}: {M[dist]=:.2f}, {SE[dist]=:.2f}, {t[dist]=:.2f}')

    # TODO: Use surface meshes to estimate distance


@njit(fastmath=True, nopython=True, cache=True)
def calc_dist2MI(same_Ms, euc_mtx, max_vox=100):
    result = np.zeros(max_vox)
    rows, cols = same_Ms.shape

    cnter = np.zeros(max_vox, dtype=np.int32)
    idxs = np.zeros((max_vox, rows * cols, 2), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            dist = euc_mtx[i, j]
            if dist >= max_vox: continue
            idxs[dist, cnter[dist], 0] = i
            idxs[dist, cnter[dist], 1] = j
            cnter[dist] += 1

    for dist in range(max_vox):
        cnt = cnter[dist]
        for i in range(cnt):
            idx0 = idxs[dist, i, 0]
            idx1 = idxs[dist, i, 1]
            result[dist] += same_Ms[idx0, idx1]
        if cnt > 0:
            result[dist] /= cnt
        else:
            result[dist] = GLOBAL_NAN_VALUE

    return result, cnter

@njit(fastmath=True, nopython=True, cache=True)
def get_voxel_dist2MI(same_Ms, euc_mtx, max_vox=100):
    # TODO: Increase max vox for OC_IT
    n_rows, n_cols = same_Ms.shape
    result = np.zeros((n_rows, max_vox))
    cnter = np.zeros((n_rows, max_vox), dtype=np.int32)

    for i in range(n_rows):
        # cnter = np.zeros(max_vox, dtype=np.int32)
        # idxs = np.zeros((max_vox, n_cols, 2), dtype=np.int32)
        for j in range(n_cols):
            dist = euc_mtx[i, j]
            if dist >= max_vox: continue
            result[i, dist] += same_Ms[i, j]
            cnter[i, dist] += 1
            # dist = euc_mtx[i, j]
            # if dist >= max_vox: continue
            # idxs[dist, cnter[dist], 0] = i
            # idxs[dist, cnter[dist], 1] = j
            #

        for dist in range(max_vox):
            cnt = cnter[i, dist]
            # for i in range(cnt):
            #     idx0 = idxs[dist, i, 0]
            #     idx1 = idxs[dist, i, 1]
            #     result[i, dist] += same_Ms[idx0, idx1]
            if cnt > 0:
                result[i, dist] /= cnt
            else:
                result[i, dist] = GLOBAL_NAN_VALUE

    return result, cnter


def get_idx2euc_custom(valid_voxel_idxs):
    euc_mtx = spatial.distance.cdist(valid_voxel_idxs, valid_voxel_idxs)
    euc_mtx = np.array(euc_mtx, dtype=np.int8)
    return euc_mtx

@cache
def get_idx2euc(region='ITL_L'):
    sn = 102
    fp = 'bl7_fMRI'
    semantic = True
    layer = None
    same_Ms, valid_voxel_idxs, has_nans = utils.pickle_wrap(generate_RSA_size_map, None,
                                                  kwargs={'sn': sn, 'fp': fp,
                                                          'region': region,
                                                          'semantic': semantic,
                                                          'layer': layer},
                                                  verbose=-1, easy_override=False)
    euc_mtx = spatial.distance.cdist(valid_voxel_idxs, valid_voxel_idxs)
    euc_mtx = np.array(euc_mtx, dtype=np.int8)
    return euc_mtx

    # for i in range(euc_mtx.shape[0]):
    #     for j in range(euc_mtx.shape[1]):
    #         if i > j:
    #             euc_mtx[j, i] = euc_mtx[i, j]


def plot_euc_hist(region='ITL_L'):
    sn = 102
    fp = 'bl7_fMRI'
    semantic = True
    layer = None
    same_Ms, valid_voxel_idxs, has_nans = utils.pickle_wrap(generate_RSA_size_map, None,
                                                  kwargs={'sn': sn, 'fp': fp,
                                                          'region': region,
                                                          'semantic': semantic,
                                                          'layer': layer},
                                                  verbose=-1, easy_override=False)
    euc_mtx = spatial.distance.cdist(valid_voxel_idxs, valid_voxel_idxs)
    eucs = euc_mtx[np.tril_indices_from(euc_mtx, k=-1)]
    plt.hist(eucs, bins=100, range=(1, 100), density=True, cumulative=True)
    plt.show()


if __name__ == '__main__':
    run_RSA_map_all_sn('ITL_L', True, None)
    # run_RSA_map_all_sn('Occipital_L', False, 0)
    # run_RSA_map_all_sn('OC_IT_L', True, None)
    # run_RSA_map_all_sn('OC_IT_L', False, 0)

    # run_RSA_map_all_sn('OC_IT_R', True, None)
    # run_RSA_map_all_sn('Occipital_L', True, None)

    # TODO: Still need to try median_cond2=False, because with it true then the comparison to the
    #   voxel is worse... or I could compute the voxel one based on the median's implied by
    #   the first voxel
