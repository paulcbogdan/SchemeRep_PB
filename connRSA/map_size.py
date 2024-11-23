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
def run_map_MI_RSA_mat(voxels_mat0, voxels_mat_bool, stim_RSM, median_cond2=False):
    n_voxels0 = voxels_mat0.shape[1]
    same_Ms_mat = np.empty((n_voxels0, n_voxels0))
    stim_is_nan = np.isnan(stim_RSM)
    voxels_mat1 = voxels_mat0 if median_cond2 else voxels_mat_bool

    for v in tqdm(range(n_voxels0)):
        voxel_l = voxels_mat_bool[:, v]
        same_Ms_v = run_map_MI_RSA(voxel_l, voxels_mat1, stim_RSM,
                                   stim_is_nan, median_cond2=median_cond2)
        same_Ms_mat[v, :] = same_Ms_v
    return same_Ms_mat

@njit(fastmath=True, nopython=True, cache=True)
def run_vox_MI_RSA(voxels_mat, stim_RSM, stim_is_nan):
    n_trials = voxels_mat.shape[0]
    n_voxels = voxels_mat.shape[1]
    same_Ms = np.empty(n_voxels)
    for v in range(n_voxels):
        voxel_l = voxels_mat[:, v]
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
                         median_cond2=False):
    voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM = (
        load_img_rsm(region, fp, sn, semantic, layer))
    t_st = time()
    stim_is_nan = np.isnan(stim_RSM)
    same_Ms = run_vox_MI_RSA(voxels_mat, voxels_mat_bool, stim_RSM, stim_is_nan)
    # same_Ms = run_map_MI_RSA_mat(voxels_mat, voxels_mat, stim_RSM)
    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    same_Ms[same_Ms == GLOBAL_NAN_VALUE] = np.nan

    return same_Ms, valid_voxel_idxs


# if True:
@njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA(voxel_l, voxels_mat, stim_RSM,
                   stim_is_nan, #stim_M, stim_SD,
                   median_cond2=False):
    # break into bins
    n_trials = voxel_l.shape[0]
    assert n_trials == 114
    n_voxels = voxels_mat.shape[1]
    # voxel_zeroes = 2
    # print(median_cond2)
    if median_cond2:
        # pass
        center_is_zero = np.argwhere(voxel_l)
        cnt_zero = center_is_zero.shape[0]
        # print(voxel_l.shape)
        # print(voxel_l)
        # quit()
        center_is_one = np.argwhere(~voxel_l)
        cnt_one = center_is_one.shape[0]
        voxel_Ms = np.empty((2, n_voxels))
        for v in range(n_voxels):
            t_at_is_zero = np.full(n_trials, 0, dtype=np.float64)
            for i in range(cnt_zero):
                t = center_is_zero[i][0]
                t_at_is_zero[i] = voxels_mat[t, v]

            voxel_Ms[0, v] = np.median(t_at_is_zero[:cnt_zero])
            t_at_is_one = np.full(n_trials, 0, dtype=np.float64)
            for i in range(cnt_one):
                t = center_is_one[i][0]
                t_at_is_one[i] = voxels_mat[t, v]
            voxel_Ms[1, v] = np.median(t_at_is_one[:cnt_one])

    t2cond0 = np.empty(n_trials, dtype=np.int8)
    for t in range(n_trials):
        t2cond0[t] = np.int8(voxel_l[t])


    same_Ms = np.empty(n_voxels)
    # cnter = np.zeros((2, 2), dtype=np.int8)

    trial2cond_cond = np.empty(n_trials, dtype=np.int8)
    # t2cond1 = np.empty(n_trials, dtype=np.int8)
    for v in range(n_voxels):
        # continue
        for t in range(n_trials):
            # continue
            # cond0 = np.int8(voxel_l[t] == 1)
            if median_cond2:
                cond0 = t2cond0[t]
                cond1 = voxels_mat[t, v] < voxel_Ms[cond0, v]
                cond1 = np.int8(cond1)
                trial2cond_cond[t] = cond0 * 2 + cond1
            else:
                # same_Ms[v] = voxels_mat[t, v] * 5
                trial2cond_cond[t] = t2cond0[t] * 2 + np.int8(voxels_mat[t, v])
        # continue
        same_total = 0
        n_same = 0
        # continue
        for t0 in range(n_trials):
            for t1 in range(t0):
                if stim_is_nan[t0, t1]:
                    break
                if trial2cond_cond[t0] == trial2cond_cond[t1]:
                    n_same += 1
                    same_total += stim_RSM[t0, t1]
                    # print(stim_RSM[t0, t1])

        if n_same == 0:
            same_M = GLOBAL_NAN_VALUE
        else:
            same_M = same_total / n_same
        same_Ms[v] = same_M
    return same_Ms

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

def load_img_rsm(region, fp, sn, semantic, layer, easy_override=False):
    # fp_nii = fr'cache/get_img_region_nii/{region}_{fp}_{sn}.nii'
    # Path(fp_nii).parent.mkdir(parents=True, exist_ok=True)
    # if os.path.exists(fp_nii) and not easy_override:
    #     img = image.load_img(fp_nii).get_fdata()
    # else:
    img = get_img_region(region, fp, sn)
        # image.new_img_like(get_atlas()['maps'], img).to_filename(fp_nii)



    # t_st = time()
    # img = get_img_region(region, fp, sn)
    # print('Time: ', time() - t_st)
    # image.new_img_like(get_atlas()['maps'], img).to_filename(fp_nii)
    # t_st = time()
    # img = image.load_img(fp_nii).get_fdata()
    # print('Time: ', time() - t_st)
    # quit()

    # img = utils.pickle_wrap(get_img_region, None,
    #                         kwargs={'region': region, 'fp': fp, 'sn': sn},
    #                         verbose=-1, easy_override=False)

    valid_voxel_idxs = np.argwhere(~np.isnan(img[:, :, :, 0]))

    voxels_mat = img[valid_voxel_idxs[:, 0], valid_voxel_idxs[:, 1],
                     valid_voxel_idxs[:, 2], :]
    voxels_mat = voxels_mat.T

    # if not median_cond2:
    voxels_mat_bool = voxels_mat > np.mean(voxels_mat, axis=0)


    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    return voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM

def generate_RSA_size_map(sn, fp, region, semantic, layer,
                          median_cond2=False):
    voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM = (
        load_img_rsm(region, fp, sn, semantic, layer))
    t_st = time()
    same_Ms = run_map_MI_RSA_mat(voxels_mat, voxels_mat_bool, stim_RSM,
                                 median_cond2=median_cond2)
    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    same_Ms[same_Ms == GLOBAL_NAN_VALUE] = np.nan

    return same_Ms, valid_voxel_idxs




def get_dist2MI(sn, fp, semantic, layer, region, subtract_vox, max_vox,
                subtract_other=True, median_cond2=False):
    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_M = np.nanmean(stim_RSM) * 10_000
    print(f'Doing: {sn}, {fp}')
    kw = {'sn': sn, 'fp': fp, 'region': region, 'semantic': semantic,
          'layer': layer}
    if median_cond2:
        kw['median_cond2'] = True
    same_Ms, valid_voxel_idxs = utils.pickle_wrap(generate_RSA_size_map, None,
                                                  kwargs=kw, verbose=-1,
                                                  easy_override=False)
    # same_Ms *= 10_000
    # same_Ms -= stim_M
    # plt.hist(same_Ms.flatten(), bins=100, range=(-100, 100))
    # plt.show()
    # quit()

    if subtract_vox:
        # check for beng extra close to zero: Doing: 138, vis7_fMRI | 4.121147867408581e-13
        kw = {'sn': sn, 'fp': fp, 'region': region, 'semantic': semantic,
              'layer': layer}
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

    vox_dist2MI, cnter = get_voxel_dist2MI(same_Ms, euc_mtx, max_vox=max_vox)
    # print(cnter.shape)
    # plt.imshow(cnter, aspect='auto')
    # plt.show()
    # quit()
    vox_dist2MI[vox_dist2MI == GLOBAL_NAN_VALUE] = np.nan
    dist2MI = np.nanmean(vox_dist2MI, axis=0)
    print(f'Time needed for dist2MI: {time() - t_st:.2f} s')

    # dist2MI, cnter = calc_dist2MI(same_Ms, euc_mtx, max_vox=max_vox)
    # dist2MI[dist2MI == GLOBAL_NAN_VALUE] = np.nan
    # # dist2MI -= np.nanmean(dist2MI)
    #
    # r, p = stats.spearmanr(dist2MI, dist2MI2, nan_policy='omit')
    # print(F'{r=}, {p=}')
    # quit()

    print(f'Time needed for dist2MI: {time() - t_st:.2f} s')
    return vox_dist2MI, dist2MI, valid_voxel_idxs, cnter

def run_RSA_map_all_sn(region, semantic, layer, max_vox=55, subtract_vox=True,
                       median_cond2=False):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # sns = sns[4::5]
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
                                                             'median_cond2': median_cond2},
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
    same_Ms, valid_voxel_idxs = utils.pickle_wrap(generate_RSA_size_map, None,
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
    same_Ms, valid_voxel_idxs = utils.pickle_wrap(generate_RSA_size_map, None,
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
    # run_RSA_map_all_sn('OC_IT_R', True, None)
    # run_RSA_map_all_sn('Occipital_L', True, None)

