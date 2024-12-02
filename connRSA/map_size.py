import os.path

import numpy as np
from numba import njit, prange
from time import time

from Utils.atlas_funcs import get_atlas
from connRSA.fft_funcs import get_ROIs_from_region
from connRSA.jit_funcs import NAN_VAL, do_int_downsample
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

GLOBAL_NAN_VALUE = -999_999

    # for the less than/greater than, we just check whether in trial 1, a > b
    #    is similar to, say, trial 2 a > b

    # Note that high RSA correlation when a predictor is 1s and 0s
    #   is similar to just finding a set of 1s where the average similar is high

# @njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA_mat(voxels_mat0, voxels_mat_bool, stim_RSM, median_cond2=False,
                       just_second_vox=False, cross_only=False, xor=False):
    n_voxels0 = voxels_mat0.shape[1]
    same_Ms_mat = np.empty((n_voxels0, n_voxels0))
    stim_is_nan = np.isnan(stim_RSM)
    voxels_mat1 = voxels_mat0 if median_cond2 else voxels_mat_bool

    MI_both_mat = np.empty((n_voxels0, n_voxels0))
    MI_vox0_mat = np.empty((n_voxels0, n_voxels0))
    MI_vox1_mat = np.empty((n_voxels0, n_voxels0))
    MI_not0only1_mat = np.empty((n_voxels0, n_voxels0))
    MI_not1only0_mat = np.empty((n_voxels0, n_voxels0))
    MI_neither_mat = np.empty((n_voxels0, n_voxels0))

    ex_both_mat = np.empty((n_voxels0, n_voxels0))
    ex_vox0_mat = np.empty((n_voxels0, n_voxels0))
    ex_vox1_mat = np.empty((n_voxels0, n_voxels0))
    ex_not0only1_mat = np.empty((n_voxels0, n_voxels0))
    ex_not1only0_mat = np.empty((n_voxels0, n_voxels0))
    ex_neither_mat = np.empty((n_voxels0, n_voxels0))
    ex_RSM_RSM_interaction_equal_mat = np.empty((n_voxels0, n_voxels0))
    ex_RSM_RSM_interaction_unequal_mat = np.empty((n_voxels0, n_voxels0))

    for v in tqdm(range(0, n_voxels0)):
        voxel_l = voxels_mat_bool[:, v]
        if np.all(voxel_l) or np.all(~voxel_l):
            MI_both_mat[v, :] = np.nan
            MI_vox0_mat[v, :] = np.nan
            MI_vox1_mat[v, :] = np.nan
            MI_not0only1_mat[v, :] = np.nan
            MI_not1only0_mat[v, :] = np.nan
            MI_neither_mat[v, :] = np.nan
            ex_both_mat[v, :] = np.nan
            ex_vox0_mat[v, :] = np.nan
            ex_vox1_mat[v, :] = np.nan
            ex_not0only1_mat[v, :] = np.nan
            ex_not1only0_mat[v, :] = np.nan
            ex_neither_mat[v, :] = np.nan
            ex_RSM_RSM_interaction_equal_mat[v, :] = np.nan
            ex_RSM_RSM_interaction_unequal_mat[v, :] = np.nan
            continue
        try:
            (MI_both, MI_vox0, MI_vox1, MI_not0only1, MI_not1only0,
             ex_both, ex_vox0, ex_vox1, ex_not0only1, ex_not1only0,
             MI_neither, ex_neither,
             ex_RSM_RSM_interaction_equal, ex_RSM_RSM_interaction_unequal
             ) = (
                run_map_MI_RSA(voxel_l, voxels_mat1, stim_RSM,
                               stim_is_nan, median_cond2=median_cond2,
                               just_second_vox=just_second_vox,
                               cross_only=cross_only, xor=xor))
        except AssertionError:
            MI_both_mat[v, :] = np.nan
            MI_vox0_mat[v, :] = np.nan
            MI_vox1_mat[v, :] = np.nan
            MI_not0only1_mat[v, :] = np.nan
            MI_not1only0_mat[v, :] = np.nan
            MI_neither_mat[v, :] = np.nan
            ex_both_mat[v, :] = np.nan
            ex_vox0_mat[v, :] = np.nan
            ex_vox1_mat[v, :] = np.nan
            ex_not0only1_mat[v, :] = np.nan
            ex_not1only0_mat[v, :] = np.nan
            ex_neither_mat[v, :] = np.nan
            ex_RSM_RSM_interaction_equal_mat[v, :] = np.nan
            ex_RSM_RSM_interaction_unequal_mat[v, :] = np.nan
            continue
        MI_both_mat[v, :] = MI_both
        MI_vox0_mat[v, :] = MI_vox0
        MI_vox1_mat[v, :] = MI_vox1
        MI_not0only1_mat[v, :] = MI_not0only1
        MI_not1only0_mat[v, :] = MI_not1only0
        MI_neither_mat[v, :] = MI_neither
        ex_both_mat[v, :] = ex_both
        ex_vox0_mat[v, :] = ex_vox0
        ex_vox1_mat[v, :] = ex_vox1
        ex_not0only1_mat[v, :] = ex_not0only1
        ex_not1only0_mat[v, :] = ex_not1only0
        ex_neither_mat[v, :] = ex_neither
        ex_RSM_RSM_interaction_equal_mat[v, :] = ex_RSM_RSM_interaction_equal
        ex_RSM_RSM_interaction_unequal_mat[v, :] = ex_RSM_RSM_interaction_unequal

    if xor:
        ex_benefit = None
    else:
        ex_benefit = (ex_both_mat + ex_neither_mat) / (ex_not0only1_mat + ex_not1only0_mat)

    return (MI_both_mat, MI_vox0_mat, MI_vox1_mat, MI_not0only1_mat, MI_not1only0_mat,
            #ex_both_mat, ex_vox0_mat, ex_vox1_mat, ex_not0only1_mat, ex_not1only0_mat,
            MI_neither_mat,
            ex_benefit,
            ex_RSM_RSM_interaction_equal_mat, ex_RSM_RSM_interaction_unequal_mat
            #ex_neither_mat
            )

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


@njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA(voxel_l, voxels_mat, stim_RSM,
                   stim_is_nan, median_cond2=False,
                   just_second_vox=False, cross_only=False,
                   xor=False):
    # break into bins
    n_trials = voxel_l.shape[0]
    assert n_trials == 114
    n_voxels = voxels_mat.shape[1]

    if median_cond2:
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
    MI_neither = np.empty(n_voxels, dtype=np.float32)
    ex_both = np.empty(n_voxels, dtype=np.int16)
    ex_neither = np.empty(n_voxels, dtype=np.int16)
    # if xor:
    #     MI_vox0 = 0
    #     MI_vox1 = 0
    #     MI_not0only1 = 0
    #     MI_not1only0 = 0
    #     ex_vox0 = 0#
    #     ex_vox1 = 0
    #     ex_not0only1 = 0
    #     ex_not1only0 = 0
    # else:
    MI_vox0 = np.empty(n_voxels, dtype=np.float32)
    MI_vox1 = np.empty(n_voxels, dtype=np.float32)
    MI_not0only1 = np.empty(n_voxels, dtype=np.float32)
    MI_not1only0 = np.empty(n_voxels, dtype=np.float32)
    ex_vox0 = np.empty(n_voxels, dtype=np.int16)
    ex_vox1 = np.empty(n_voxels, dtype=np.int16)
    ex_not0only1 = np.empty(n_voxels, dtype=np.int16)
    ex_not1only0 = np.empty(n_voxels, dtype=np.int16)

    # ex_both_equal = np.empty(n_voxels, dtype=np.int16)
    # ex_both_unequal = np.empty(n_voxels, dtype=np.int16)
    # ex_neither_equal = np.empty(n_voxels, dtype=np.int16)
    # ex_neither_unequal = np.empty(n_voxels, dtype=np.int16)
    # ex_not0only1_equal = np.empty(n_voxels, dtype=np.int16)
    # ex_not0only1_unequal = np.empty(n_voxels, dtype=np.int16)
    # ex_not1only0_equal = np.empty(n_voxels, dtype=np.int16)
    # ex_not1only0_unequal = np.empty(n_voxels, dtype=np.int16)

    ex_RSM_RSM_interaction_equal = np.empty(n_voxels, dtype=np.float32)
    ex_RSM_RSM_interaction_unequal = np.empty(n_voxels, dtype=np.float32)

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
        MI_voxel_not0only1 = 0
        MI_voxel_not1only0 = 0
        MI_neither_voxel = 0

        n_same = 0
        n_same0 = 0
        n_same1 = 0
        n_not0only1 = 0
        n_not1only0 = 0
        n_neither = 0

        n_same_equal = 0
        n_same_unequal = 0
        n_neither_equal = 0
        n_neither_unequal = 0
        n_not0only1_equal = 0
        n_not0only1_unequal = 0
        n_not1only0_equal = 0
        n_not1only0_unequal = 0



        for t0 in range(n_trials):
            if cross_only:
                if t2vox0[t0] == t2vox1[t0]:
                    continue
            for t1 in range(t0):
                if stim_is_nan[t0, t1]:
                    break
                # if cross_only:
                #     if t2vox0[t1] == t2vox1[t1]:
                #         continue

                if xor:
                    if t2vox0[t0] == t2vox1[t0]:
                        if t2vox0[t1] == t2vox1[t1]:
                            n_same += 1
                            MI_both_voxel += stim_RSM[t0, t1]
                        else:
                            n_neither += 1
                            MI_neither_voxel += stim_RSM[t0, t1]
                    else:
                        if t2vox0[t1] != t2vox1[t1]:
                            n_same += 1
                            MI_both_voxel += stim_RSM[t0, t1]
                        else:
                            n_neither += 1
                            MI_neither_voxel += stim_RSM[t0, t1]
                    continue

                if t2vox0[t0] == t2vox0[t1]:
                    n_same0 += 1
                    MI_voxel0 += stim_RSM[t0, t1]
                    if t2vox1[t0] == t2vox1[t1]:
                        n_same1 += 1
                        n_same += 1
                        if t2vox0[t0] == t2vox1[t0]:
                            n_same_equal += 1
                        else:
                            n_same_unequal += 1
                        MI_voxel1 += stim_RSM[t0, t1]
                        MI_both_voxel += stim_RSM[t0, t1]
                    else:
                        n_not1only0 += 1
                        if t2vox0[t0] == t2vox1[t0]:
                            n_not1only0_equal += 1
                        else:
                            n_not1only0_unequal += 1
                        MI_voxel_not1only0 += stim_RSM[t0, t1]
                else:
                    if t2vox1[t0] == t2vox1[t1]:
                        n_same1 += 1
                        MI_voxel1 += stim_RSM[t0, t1]
                        n_not0only1 += 1
                        if t2vox0[t0] == t2vox1[t0]:
                            n_not0only1_equal += 1
                        else:
                            n_not0only1_unequal += 1
                        MI_voxel_not0only1 += stim_RSM[t0, t1]
                    else:
                        n_neither += 1
                        if t2vox0[t0] == t2vox1[t0]:
                            n_neither_equal += 1
                        else:
                            n_neither_unequal += 1
                        MI_neither_voxel += stim_RSM[t0, t1]
        # print(f'{n_same_equal=}')
        # print(f'{n_same_unequal=}')
        # print(f'{n_neither_equal=}')
        # print(f'{n_neither_unequal=}')
        # print(f'{n_not0only1_equal=}')
        # print(f'{n_not0only1_unequal=}')
        # print(f'{n_not1only0_equal=}')
        # print(f'{n_not1only0_unequal=}')
        # print()


        ex_both[v] = n_same
        ex_neither[v] = n_neither

        if n_same == 0:
            MI_both[v] = GLOBAL_NAN_VALUE
        else:
            MI_both[v] = MI_both_voxel / n_same
        if n_neither == 0:
            MI_neither[v] = GLOBAL_NAN_VALUE
        else:
            MI_neither[v] = MI_neither_voxel / n_neither
        if xor:
            continue

        ex_vox0[v] = n_same0
        ex_vox1[v] = n_same1
        ex_not0only1[v] = n_not0only1
        ex_not1only0[v] = n_not1only0

        if n_same0 == 0:
            MI_vox0[v] = GLOBAL_NAN_VALUE
        else:
            MI_vox0[v] = MI_voxel0 / n_same0
        if n_same1 == 0:
            MI_vox1[v] = GLOBAL_NAN_VALUE
        else:
            MI_vox1[v] = MI_voxel1 / n_same1
        if n_not0only1 == 0:
            MI_not0only1[v] = GLOBAL_NAN_VALUE
        else:
            MI_not0only1[v] = MI_voxel_not0only1 / n_not0only1
        if n_not1only0 == 0:
            MI_not1only0[v] = GLOBAL_NAN_VALUE
        else:
            MI_not1only0[v] = MI_voxel_not1only0 / n_not1only0

        # ex_RSM_RSM_interaction_equal[v] = n_same_equal + n_neither_equal

        # ex_both_equal[v] = n_same_equal
        # ex_both_unequal[v] = n_same_unequal
        # ex_neither_equal[v] = n_neither_equal
        # ex_neither_unequal[v] = n_neither_unequal
        # ex_not0only1_equal[v] = n_not0only1_equal
        # ex_not0only1_unequal[v] = n_not0only1_unequal
        # ex_not1only0_equal[v] = n_not1only0_equal
        # ex_not1only0_unequal[v] = n_not1only0_unequal

        if n_not1only0_equal + n_not0only1_equal > 0:
            ex_RSM_RSM_interaction_equal[v] = ((n_same_equal + n_neither_equal) /
                                           (n_not1only0_equal + n_not0only1_equal))
        else:
            ex_RSM_RSM_interaction_equal[v] = GLOBAL_NAN_VALUE
        if n_not1only0_unequal + n_not0only1_unequal > 0:
            ex_RSM_RSM_interaction_unequal[v] = ((n_same_unequal + n_neither_unequal) /
                                                 (n_not1only0_unequal + n_not0only1_unequal))
        else:
            ex_RSM_RSM_interaction_unequal[v] = GLOBAL_NAN_VALUE


    # return
    return (MI_both, MI_vox0, MI_vox1, MI_not0only1, MI_not1only0,
            ex_both, ex_vox0, ex_vox1, ex_not0only1, ex_not1only0,
            MI_neither, ex_neither,
            ex_RSM_RSM_interaction_equal, ex_RSM_RSM_interaction_unequal
            )

# @cache
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
                 stdize_by_run=True, downsample_rate=2, stdize_vol=False):

    img = get_img_region(region, fp, sn)

    if downsample_rate:
        nan_val = GLOBAL_NAN_VALUE
        mask = ~np.isnan(img)
        img[np.isnan(img)] = nan_val

        img = do_int_downsample(img, downsample_rate, mask,
                                nan_val=nan_val, fourD_mask=True)
        # print(np.sum(img == nan_val))
        img[img == nan_val] = np.nan
        num_not_nan = np.sum(~np.isnan(img))
        if stdize_vol:
            # img = stats.zscore(img, axis=(0, 1, 2), nan_policy='omit')
            img = utils.stdize(img, stdize_by_run=False, axis=(0, 1, 2),
                               nans=True)
            # print(np.nanmean(img, axis=(0, 1, 2)).shape)
            # quit()
        # print(img.shape)

        # quit()
        # num_nan = np.sum(np.isnan(img))
        # print(f'{num_nan=}, {num_not_nan=}')
        # quit()
        assert num_not_nan > 0, f'All NaN: {num_not_nan=}'


    valid_voxel_idxs = np.argwhere(~np.isnan(img[:, :, :, 0]))

    voxels_mat = img[valid_voxel_idxs[:, 0], valid_voxel_idxs[:, 1],
                     valid_voxel_idxs[:, 2], :]
    voxels_mat = voxels_mat.T
    # print(f'{voxels_mat.shape=}')
    # quit()


    if stdize_by_run:
        run0 = stats.zscore(voxels_mat[:38, :], axis=0, nan_policy='omit')
        run1 = stats.zscore(voxels_mat[38:76, :], axis=0, nan_policy='omit')
        run2 = stats.zscore(voxels_mat[76:114, :], axis=0, nan_policy='omit')
        voxels_mat = np.concatenate((run0, run1, run2), axis=0)

        # voxels_mat = utils.stdize(voxels_mat, stdize_by_run=True, axis=0)

    voxels_mat_bool = voxels_mat > np.nanmedian(voxels_mat, axis=0)


    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    return voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM

# load_img_rsm('ITL_L', 'obj7_fMRI', 102, True, None)

def generate_RSA_size_map(sn, fp, region, semantic, layer,
                          median_cond2=False, stdize_by_run=True,
                          just_second_vox=False, downsample_rate=None,
                          stdize_vol=True, cross_only=False,
                          xor=False, just_RSM_RSM=False):
    voxels_mat, voxels_mat_bool, valid_voxel_idxs, stim_RSM = (
        load_img_rsm(region, fp, sn, semantic, layer, stdize_by_run=stdize_by_run,
                     downsample_rate=downsample_rate, stdize_vol=stdize_vol))

    has_nans = []
    for v in range(voxels_mat_bool.shape[1]):
        cnt_nan = np.sum(np.isnan(voxels_mat[:, v]))
        has_nans.append(cnt_nan > 0)
    has_nans = np.array(has_nans)

    t_st = time()
    (MI_both_mat, MI_vox0_mat, MI_vox1_mat, MI_not0only1_mat, MI_not1only0_mat,
     #ex_both_mat, ex_vox0_mat, ex_vox1_mat, ex_not0only1_mat, ex_not1only0_mat,
     MI_neither_mat, ex_benefit,
     ex_RSM_RSM_interaction_equal_mat, ex_RSM_RSM_interaction_unequal_mat
     #ex_neither_mat
     ) = (
        run_map_MI_RSA_mat(voxels_mat, voxels_mat_bool, stim_RSM,
                           median_cond2=median_cond2, just_second_vox=just_second_vox,
                           cross_only=cross_only, xor=xor))
    # same_Ms = np.float32(same_Ms)
    # MI_both_mat = np.float32(MI_both_mat)
    # MI_vox0_mat = np.float32(MI_vox0_mat)
    # MI_vox1_mat = np.float32(MI_vox1_mat)
    # ex_both_mat = np.float32(ex_both_mat)
    # ex_vox0_mat = np.float32(ex_vox0_mat)
    # ex_vox1_mat = np.float32(ex_vox1_mat)

    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    MI_both_mat[MI_both_mat == GLOBAL_NAN_VALUE] = np.nan
    if xor:
        MI_vox0_mat = None
        MI_vox1_mat = None
        MI_not0only1_mat = None
        MI_not1only0_mat = None
    else:
        MI_vox0_mat[MI_vox0_mat == GLOBAL_NAN_VALUE] = np.nan
        MI_vox1_mat[MI_vox1_mat == GLOBAL_NAN_VALUE] = np.nan
        MI_not0only1_mat[MI_not0only1_mat == GLOBAL_NAN_VALUE] = np.nan
        MI_not1only0_mat[MI_not1only0_mat == GLOBAL_NAN_VALUE] = np.nan
        MI_neither_mat[MI_neither_mat == GLOBAL_NAN_VALUE] = np.nan

    if just_RSM_RSM:
        MI_both_mat = None
        MI_vox0_mat = None
        MI_vox1_mat = None
        MI_not0only1_mat = None
        MI_not1only0_mat = None
        MI_neither_mat = None

    return (MI_both_mat, MI_vox0_mat, MI_vox1_mat, MI_not0only1_mat, MI_not1only0_mat,
            # ex_both_mat, ex_vox0_mat, ex_vox1_mat, ex_not0only1_mat, ex_not1only0_mat,
            MI_neither_mat, #ex_neither_mat,
            ex_benefit, ex_RSM_RSM_interaction_equal_mat, ex_RSM_RSM_interaction_unequal_mat,
            valid_voxel_idxs, has_nans)




def get_dist2MI(sn, fp, semantic, layer, region, subtract_vox, max_vox,
                subtract_other=False, median_cond2=False, stdize_by_run=True,
                downsample_rate=None, stdize_vol=False, cross_only=False,
                do_itr=False, dist_downplay=False, xor=False,
                RSM_RSM=False, RSM_RSM_override=True, #vox0_RSA=False
                ):
    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_M = np.nanmean(stim_RSM) * 10_000
    print(f'Doing: {sn}, {fp}')
    kw = {'sn': sn, 'fp': fp, 'region': region, 'semantic': semantic,
          'layer': layer, 'stdize_by_run': stdize_by_run,
          'median_cond2': median_cond2, 'downsample_rate': downsample_rate,
          'stdize_vol': stdize_vol, 'cross_only': cross_only, 'xor': xor,
          'just_RSM_RSM': RSM_RSM}

    (MI_both_mat, MI_vox0_mat, MI_vox1_mat, MI_not0only1_mat, MI_not1only0_mat,
     # ex_both_mat, ex_vox0_mat, ex_vox1_mat, ex_not0only1_mat, ex_not1only0_mat,
     MI_neither_mat,
     ex_benefit,
     #ex_neither_mat,
     ex_RSM_RSM_interaction_equal_mat, ex_RSM_RSM_interaction_unequal_mat,
     valid_voxel_idxs, has_nans) = utils.pickle_wrap(generate_RSA_size_map, None,
                                                     kwargs=kw, verbose=-1,
                                                     easy_override=False)

    ex_benefit[np.diag_indices_from(ex_benefit)] = np.nan
    ex_RSM_RSM_interaction_equal_mat[np.diag_indices_from(ex_RSM_RSM_interaction_equal_mat)] = np.nan
    ex_RSM_RSM_interaction_unequal_mat[np.diag_indices_from(ex_RSM_RSM_interaction_unequal_mat)] = np.nan
    if RSM_RSM_override:
        ex_RSM_RSM_interaction_equal_mat[ex_RSM_RSM_interaction_equal_mat == GLOBAL_NAN_VALUE] = np.nan
        ex_RSM_RSM_interaction_unequal_mat[ex_RSM_RSM_interaction_unequal_mat == GLOBAL_NAN_VALUE] = np.nan

        # ex_benefit = ex_RSM_RSM_interaction_equal_mat
        ex_benefit = (ex_RSM_RSM_interaction_equal_mat + ex_RSM_RSM_interaction_unequal_mat) / 2

    if not RSM_RSM:
        MI_both_mat *= 10_000
    if RSM_RSM:
        if np.sum(np.isinf(ex_benefit)) > 0:
            print(f'RSM x RSM has infinites: {np.sum(np.isinf(ex_benefit))}')
            ex_benefit[np.isinf(ex_benefit)] = np.nan
        M_benefit = np.nanmean(np.log(ex_benefit))
        print(f'RSM x RSM bias: {M_benefit:.4f}')
    elif xor:
        MI_neither_mat *= 10_000
        dif = MI_both_mat - MI_neither_mat
        print(f'Stim: {np.nanmean(stim_M):.2f}, main: {np.nanmean(MI_both_mat):.2f}, '
              f'neither: {np.nanmean(MI_neither_mat):.2f}, '
              f'xor = {np.nanmean(dif):.2f}, '
              f'{stim_M=:.2f}')
    elif vox0_RSA:
        MI_vox0_mat *= 10_000
        MI_vox0_mat -= stim_M
        print(f'{np.nanmean(MI_vox0_mat)=:.2f} | '
              f'{stim_M=:.2f}')
    else:
        MI_both_mat -= stim_M

        MI_vox0_mat *= 10_000
        MI_vox0_mat -= stim_M
        MI_vox1_mat *= 10_000
        MI_vox1_mat -= stim_M
        MI_not0only1_mat *= 10_000
        MI_not0only1_mat -= stim_M
        MI_not1only0_mat *= 10_000
        MI_not1only0_mat -= stim_M

        dif = MI_both_mat - MI_vox0_mat - MI_vox1_mat
        # Does voxel 0 kick into overdrive when voxel 1 doesn't cover it
        # dif2 = MI_not1only0_mat - MI_vox0_mat

        itr = MI_both_mat + MI_neither_mat - MI_not0only1_mat - MI_not1only0_mat


        print(f'{np.nanmean(MI_both_mat):.2f} = {np.nanmean(MI_vox0_mat):.2f} + '
              f'{np.nanmean(MI_vox1_mat):.2f} | Difference = {np.nanmean(dif):.2f}, '
              f'Interaction = {np.nanmean(itr):.2f} | '
              f'{stim_M=:.2f}')


        if do_itr:
            MI_both_mat = itr

    euc_mtx = get_idx2euc_custom(valid_voxel_idxs)
    # TODO: This won't work because the distance isnt precise


    if RSM_RSM:
        vox_dist2MI, cnter = get_voxel_dist2MI(np.log(ex_benefit), euc_mtx, max_dist=max_vox)
    elif xor:
        # MI_both_mat = dif
        vox_dist2MI, cnter = get_voxel_dist2MI(dif, euc_mtx, max_dist=max_vox)
    elif do_itr:
        vox_dist2MI, cnter = get_voxel_dist2MI(MI_both_mat, euc_mtx, max_dist=max_vox)
    elif vox0_RSA:
        vox_dist2MI, cnter = get_voxel_dist2MI(MI_vox0_mat, euc_mtx, max_dist=max_vox)
    else:
        vox_dist2MI_both_mat, cnter = get_voxel_dist2MI(MI_both_mat, euc_mtx, max_dist=max_vox)
        vox_dist2MI_both_mat[vox_dist2MI_both_mat == GLOBAL_NAN_VALUE] = np.nan
        vox_dist2MI_vox0, cnter = get_voxel_dist2MI(MI_vox0_mat, euc_mtx, max_dist=max_vox)
        vox_dist2MI_vox0[vox_dist2MI_vox0 == GLOBAL_NAN_VALUE] = np.nan
        vox_dist2MI_vox1, cnter = get_voxel_dist2MI(MI_vox1_mat, euc_mtx, max_dist=max_vox)
        vox_dist2MI_vox1[vox_dist2MI_vox1 == GLOBAL_NAN_VALUE] = np.nan

        if dist_downplay:
            dist2corr = [1, 0.7676304760421793, 0.40210419420651833, 0.23158682900552213, 0.19385777196049508, 0.17110289844885315, 0.14556471539576724, 0.10900815045204933, 0.09763350991271262, 0.06542207850538125, 0.047928747382955605, 0.03853565803287913]
            dist2corr = [1 - x for x in dist2corr]
            if len(dist2corr) < max_vox:
                dist2corr = dist2corr + [0] * (max_vox - len(dist2corr))
            dist2corr = np.array(dist2corr)
            vox_dist2MI = vox_dist2MI_both_mat - vox_dist2MI_vox0 - vox_dist2MI_vox1 * dist2corr
        else:
            vox_dist2MI = vox_dist2MI_both_mat - vox_dist2MI_vox0 - vox_dist2MI_vox1


    dist2MI = np.nanmean(vox_dist2MI, axis=0)

    return vox_dist2MI, dist2MI, valid_voxel_idxs, cnter

def run_RSA_map_all_sn(region, semantic, layer, max_vox=30, subtract_vox=True,
                       median_cond2=False, stdize_by_run=True, downsample_rate=2,
                       subtract_other=False, stdize_vol=True, cross_only=False,
                       xor=False, RSM_RSM=False, RSM_RSM_override=False):
    if 'OC_IT' in region:
        max_vox = 50
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    sns = sns[3::4]
    # sns = sns[1::2]
    # sns = sns[0:]
    dist2MI_all = []
    for i, sn in enumerate(sns):
        sn_dist2MI = []

        for j, fp in enumerate(fps):
            # kw = {'sn': sn, 'fp': fp, 'semantic': semantic, 'layer': layer,
            #       'region': region, 'subtract_vox': subtract_vox, 'max_vox': max_vox,
            #       'median_cond2': median_cond2, 'stdize_by_run': True,
            #       'downsample_rate': downsample_rate}

            kw = {'sn': sn, 'fp': fp, 'semantic': semantic, 'layer': layer,
                  'region': region, 'subtract_vox': subtract_vox, 'max_vox': max_vox,
                  'median_cond2': median_cond2, 'stdize_by_run': True,
                  'subtract_other': subtract_other, 'downsample_rate': downsample_rate,
                  'stdize_vol': stdize_vol, 'cross_only': cross_only,
                  'do_itr': False, 'dist_downplay': False, 'xor': xor,
                  'RSM_RSM': RSM_RSM, 'RSM_RSM_override': RSM_RSM_override}

            print(kw)
            # quit()
            vox_dist2MI, dist2MI, _, _ = utils.pickle_wrap(get_dist2MI, None,
                                                           kwargs=kw, verbose=-1,
                                                           easy_override=False)
            kw['do_itr'] = True
            # kw = kw | {'do_itr': True}

            print(kw)
            # quit()
            # vox_dist2MI, dist2MI, _, _ = utils.pickle_wrap(get_dist2MI, None,
            #                                                kwargs=kw, verbose=-1,
            #                                                easy_override=False)
            #
            # print('Done')

            # vox_dist2MI, dist2MI = get_dist2MI(sn, fp, semantic, layer, region,
            #                                    subtract_vox, max_vox)

            sn_dist2MI.append(dist2MI)
        continue
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
def get_voxel_dist2MI(same_Ms, euc_mtx, max_dist=100):
    # TODO: Increase max vox for OC_IT
    n_rows, n_cols = same_Ms.shape
    result = np.zeros((n_rows, max_dist))
    cnter = np.zeros((n_rows, max_dist), dtype=np.int32)

    for i in range(n_rows):
        for j in range(n_cols):
            dist = euc_mtx[i, j]
            # print(dist)
            if dist >= max_dist: continue
            result[i, dist] += same_Ms[i, j]
            cnter[i, dist] += 1

        for dist in range(max_dist):
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

def get_mean_corr(dist_x=1, dist_y=0, dist_z=0, sn=102):
    img = get_img_region('OC_IT_R', 'bl7_fMRI', sn)
    # print(img.shape)
    img = utils.stdize(img, stdize_by_run=False, axis=(0, 1, 2), nans=True)

    corrs = []
    while len(corrs) < 100:
        coord = np.random.randint(0, img.shape[0], 3)
        if np.isnan(img[coord[0], coord[1], coord[2], 0]):
            continue
        coord2 = (coord[0] + dist_x, coord[1] + dist_y, coord[2] + dist_z)
        if coord2[0] >= img.shape[0] or coord2[1] >= img.shape[1] or coord2[2] >= img.shape[2]:
            continue
        if np.isnan(img[coord2[0], coord2[1], coord2[2], 0]):
            continue
        try:
            r, p = stats.pearsonr(img[coord[0], coord[1], coord[2], :],
                                  img[coord2[0], coord2[1], coord2[2], :],)
        except ValueError:
            continue
        # print(f'{r=:.3f}')
        corrs.append(r)
    M_corr = np.mean(corrs)
    SD_corr = np.std(corrs)
    print(f'{dist_x}, {dist_y}, {dist_z} | {M_corr=:.3f}, {SD_corr=:.3f}')
    # quit()
    return M_corr

@cache
def get_dist_to_subtract():
    d = {1: 0.78, 2: 0.4, 3: 0.23, 4: 0.17, 5: 0.16, 6: 0.12,
         7: 0.10, 8: 0.09, 9: 0.08, 10: 0.1, 11: 0.075, 12: 0.05
         }
    return d

def map_distance_space():
    M_corrs = []

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    # fps = ['bl7_fMRI', 'obj7_fMRI']#, 'con7_fMRI', 'vis7_fMRI']
    dist2corr = []
    for dist in range(1, 20):
        M_corr_all = []
        for sn in sns[:10]:
            M_corr = get_mean_corr(dist, 0, 0, sn)
            M_corr_all.append(M_corr)
        M_corr_all = np.array(M_corr_all)
        M_corr = np.mean(M_corr_all)
        print(f'--- Distance: {dist} | {M_corr=:.3f}')
        dist2corr.append(M_corr)
        print(dist2corr)
    quit()

if __name__ == '__main__':
    # get_mean_corr()
    # map_distance_space()

    # TODO: Try the cortical with median_cond2=True

    # TODO: Maybe try median_cond2=False?

    # run_RSA_map_all_sn('cortical_L', False, 0, downsample_rate=3,
    #                    median_cond2=True, cross_only=False, xor=False,
    #                    RSM_RSM=True)

    run_RSA_map_all_sn('cortical_L', False, 0, downsample_rate=2,
                       median_cond2=False, cross_only=False, xor=False,
                       RSM_RSM=True, RSM_RSM_override=True)

    run_RSA_map_all_sn('cortical_R', False, 0, downsample_rate=2,
                       median_cond2=False, cross_only=False, xor=False,
                       RSM_RSM=True, RSM_RSM_override=True)

    # run_RSA_map_all_sn('cortical_L', False, 0, downsample_rate=2,
    #                    median_cond2=True, cross_only=False, xor=False,
    #                    RSM_RSM=True, RSM_RSM_override=True)
    #
    # run_RSA_map_all_sn('OC_IT_L', False, 0, downsample_rate=1,
    #                    median_cond2=False, cross_only=False, xor=False,
    #                    RSM_RSM=True, RSM_RSM_override=True)

    # run_RSA_map_all_sn('cortical_R', False, 0, downsample_rate=2,
    #                    median_cond2=True, cross_only=False, xor=False,
    #                    RSM_RSM=True, RSM_RSM_override=True)
    #
    # run_RSA_map_all_sn('OC_IT_R', False, 0, downsample_rate=1,
    #                    median_cond2=False, cross_only=False, xor=False,
    #                    RSM_RSM=True, RSM_RSM_override=True)


    # run_RSA_map_all_sn('PFC_L', False, 0, downsample_rate=2,
    #                    median_cond2=False, cross_only=False, xor=False,
    #                    RSM_RSM=True, RSM_RSM_override=True)

    # run_RSA_map_all_sn('cortical_L', True, None, downsample_rate=3,
    #                    median_cond2=True, cross_only=False, xor=True)


    # run_RSA_map_all_sn('OC_IT_L', False, 0)
    # run_RSA_map_all_sn('OC_IT_R', False, 0)

    # run_RSA_map_all_sn('OC_IT_L', False, 0, cross_only=True)
    # run_RSA_map_all_sn('OC_IT_R', False, 0, cross_only=True)

    # run_RSA_map_all_sn('OC_IT_L', True, None)
    # run_RSA_map_all_sn('OC_IT_R', True, None)

    # run_RSA_map_all_sn('OC_IT_L', True, None, cross_only=True)
    # run_RSA_map_all_sn('OC_IT_R', True, None, cross_only=True)

    # run_RSA_map_all_sn('OC_IT_L', True, None, median_cond2=True)
    # run_RSA_map_all_sn('OC_IT_L', False, 0, median_cond2=True)

    # TODO: Still need to try median_cond2=False, because with it true then the comparison to the
    #   voxel is worse... or I could compute the voxel one based on the median's implied by
    #   the first voxel

