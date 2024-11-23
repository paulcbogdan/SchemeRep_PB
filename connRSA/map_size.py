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

GLOBAL_NAN_VALUE = -999_999_999

def run_map_xor_RSA(voxel_l, voxels_mat, stim_RSM, stim_M, stim_SD):
    # TODO: subtract trial-wise mean

    n_trials = voxel_l.shape[0]
    n_voxels = voxels_mat.shape[1]
    xor_mat = np.zeros_like(voxels_mat)
    for t in range(n_trials):
        for v in range(n_voxels):
            xor_mat[t, v] = voxel_l[t] ^ voxels_mat[t, v]


    for v in range(n_voxels):
        E_xy = 0
        E_x = 0
        E_y = 0
        for t0 in range(n_trials):
            for t1 in range(t0):
                pass


    # for the less than/greater than, we just check whether in trial 1, a > b
    #    is similar to, say, trial 2 a > b

    # Note that high RSA correlation when a predictor is 1s and 0s
    #   is similar to just finding a set of 1s where the average similar is high

# @njit(fastmath=True, nopython=True, cache=True, parallel=True)
def run_map_MI_RSA_mat(voxels_mat0, voxels_mat1, stim_RSM, median_cond2=False):
    n_voxels0 = voxels_mat0.shape[1]
    same_Ms_mat = np.empty((n_voxels0, n_voxels0))

    stim_is_nan = np.isnan(stim_RSM)
    for v in tqdm(range(n_voxels0)):
        voxel_l = voxels_mat0[:, v]
        same_Ms_v = run_map_MI_RSA(voxel_l, voxels_mat1, stim_RSM,
                                   stim_is_nan,
                                   median_cond2=median_cond2)
        same_Ms_mat[v, :] = same_Ms_v
    return same_Ms_mat

# @njit(fastmath=True, nopython=True, cache=True, parallel=True)
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

    if median_cond2:
        # pass
        center_is_zero = np.argwhere(voxel_l)
        cnt_zero = center_is_zero.shape[0]
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

def generate_RSA_size_map(sn, fp, region, semantic, layer,
                          median_cond2=False):
    img = utils.pickle_wrap(get_img_region, None,
                            kwargs={'region': region, 'fp': fp, 'sn': sn},
                            verbose=-1, easy_override=False)

    valid_voxel_idxs = np.argwhere(~np.isnan(img[:, :, :, 0]))


    voxels_mat = img[valid_voxel_idxs[:, 0], valid_voxel_idxs[:, 1],
                     valid_voxel_idxs[:, 2], :]
    voxels_mat = voxels_mat.T
    # print(voxels_mat.shape)
    # print(np.mean(voxels_mat, axis=0).shape)
    # quit()
    if not median_cond2:
        voxels_mat = voxels_mat > np.mean(voxels_mat, axis=0)

    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)

    t_st = time()
    same_Ms = run_map_MI_RSA_mat(voxels_mat, voxels_mat, stim_RSM)
    print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
    same_Ms[same_Ms == GLOBAL_NAN_VALUE] = np.nan

    return same_Ms, valid_voxel_idxs



def run_RSA_map_all_sn(region, semantic, layer):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # sns = sns[3::5]
    dist2MI_all = []
    for i, sn in enumerate(sns):
        sn_dist2MI = []
        for j, fp in enumerate(fps):
            stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                          layer=layer)
            stim_M = np.nanmean(stim_RSM) * 10_000
            # print(f'{stim_M=:.5f}')
            print(f'Doig: {sn}, {fp}')
            t_st = time()
            same_Ms, valid_voxel_idxs = utils.pickle_wrap(generate_RSA_size_map, None,
                                                          kwargs={'sn': sn, 'fp': fp,
                                                                  'region': region,
                                                                  'semantic': semantic,
                                                                  'layer': layer},
                                                          verbose=-1, easy_override=False)
            # print(same_Ms.shape)
            # print(len(valid_voxel_idxs))
            # quit()

            print(f'Time needed for same_Ms: {time() - t_st:.2f} s')
            has_nan = np.any(np.isnan(same_Ms))
            same_Ms *= 10_000
            if has_nan:
                M_M = np.nanmean(same_Ms)
                same_Ms[np.isnan(same_Ms)] = M_M
                has_nan_str = ' (has nan)'
            else:
                M_M = np.mean(same_Ms)
                has_nan_str = ''
            dif = M_M - stim_M
            same_Ms -= stim_M
            print(f'\t{sn}, {fp}, {stim_M=:.5f} | {dif=:.5f}{has_nan_str}')
            # euc_mtx = get_idx2euc(region)
            t_st = time()
            euc_mtx = get_idx2euc_custom(valid_voxel_idxs)
            # for dist in range(30):
            #     print(f'{dist}: {np.sum(euc_mtx == dist)}')
            # # quit()
            print(f'Time needed for euc_mtx: {time() - t_st:.2f} s')
            t_st = time()
            # print(np.nanmean(same_Ms))
            # print(np.nansum(same_Ms))
            dist2MI, cnter = calc_dist2MI(same_Ms, euc_mtx)
            # print(np.nansum(dist2MI))
            dist2MI[dist2MI == GLOBAL_NAN_VALUE] = np.nan

            # dist2MI_weighted_avg = np.nansum(dist2MI * cnter) / np.nansum(cnter)
            # print(cnter)
            # print(dist2MI_weighted_avg)
            # quit()

            # dist2MI -= np.nanmean(dist2MI)
            # print(np.nanmean(dist2MI))
            print(f'Time needed for dist2MI: {time() - t_st:.2f} s')
            # print(dist2MI)
            # quit()
            sn_dist2MI.append(dist2MI)
        sn_dist2MI = np.nanmean(np.array(sn_dist2MI), axis=0)
        dist2MI_all.append(sn_dist2MI)

        M = np.nanmean(dist2MI_all, axis=0)
        SE = np.nanstd(dist2MI_all, axis=0) / np.sqrt(len(dist2MI_all))
        t = M / SE
        for dist in range(30):
            print(f'{dist}: {M[dist]=:.2f}, {SE[dist]=:.2f}, {t[dist]=:.2f}')

    # TODO: Use surface meshes to estimate distance


@njit(fastmath=True, nopython=True, cache=True)
def calc_dist2MI(same_Ms, euc_mtx, max_vox=100):
    result = np.zeros(max_vox)
    rows, cols = same_Ms.shape

    cnter = np.zeros(max_vox, dtype=np.int32)
    idxs = np.zeros((max_vox, 3_000_000, 2), dtype=np.int32)
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
    # plot_euc_hist()
    # quit()

    run_RSA_map_all_sn('ITL_L', True, None)
    # generate_RSA_size_map(102, 'bl7_fMRI', 'Occipital', False, 0)


    # voxel_l = np.random.normal(0, 1, 114)
    # voxel_l = voxel_l > np.mean(voxel_l)
    # voxels_mat = np.random.normal(0, 1, (114, 13000))
    # # print(voxels_mat.shape)
    # # quit()
    # voxels_mat = voxels_mat > np.mean(voxels_mat, axis=0)
    # voxels_mat = np.int8(voxels_mat)
    #
    # stim_RSM = np.random.normal(0, 1, (114, 114))
    # stim_is_nan = np.random.rand(114, 114) < -2
    #
    # t = time()
    # test = run_map_MI_RSA(voxel_l, voxels_mat, stim_RSM, stim_is_nan)
    # print(time() - t)
    #
    # t = time()
    # test = run_map_MI_RSA(voxel_l, voxels_mat, stim_RSM, stim_is_nan)
    # print(time() - t)
    # print(test)




