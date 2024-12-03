from xarray.core.nputils import nanquantile

from Utils.atlas_funcs import get_atlas
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from connRSA.map_plot import idxs2img
from connRSA.map_size import get_img_region, get_idx2euc_custom, get_voxel_dist2MI
from numba import njit
import numpy as np

from org_sns import get_sns
import utils
import scipy.spatial as spatial
import scipy.stats as stats
import matplotlib.pyplot as plt
from time import time
from nilearn import image, plotting
from tqdm import tqdm

GLOBAL_NAN_VALUE = -999_999
HIGH_GLOBAL = -999_998
LOW_GLOBAL = -1_000_000

@njit(fastmath=True, nopython=True, cache=True)
def parse_into_cubes(img, ROI_size=2, threshold=6):
    cube_size = ROI_size ** 3
    x, y, z, n = img.shape
    max_entries = x * y * z // cube_size

    out = np.full((max_entries, cube_size, n), GLOBAL_NAN_VALUE, dtype=np.float32)
    cnt_good_spots = 0
    valid_voxel_idxs = np.zeros((max_entries, 3), dtype=np.float32)
    size_over_two = ROI_size / 2
    ROI_sizes = np.zeros(max_entries, dtype=np.int8)

    for i in range(0, x, ROI_size):
        for j in range(0, y, ROI_size):
            for k in range(0, z, ROI_size):
                # print(22)
                # if cnt_good_spots >
                is_good = True
                good_in_all_voxels = np.ones(cube_size, dtype=np.bool_)
                # cube_goods = np.zeros(n, dtype=np.int8)
                for m in range(n):
                    cube = img[i:i + ROI_size, j:j + ROI_size, k:k + ROI_size, m].flatten()
                    if len(cube) < cube_size:
                        is_good = False
                        break

                    # goods = 0
                    for x in range(cube.shape[0]):
                        if cube[x] < HIGH_GLOBAL:
                            good_in_all_voxels[x] = False

                if np.sum(good_in_all_voxels) < threshold:
                    is_good = False
                # print(is_good)

                if is_good:
                    num_good_voxels = np.sum(good_in_all_voxels)
                    # print(num_good_voxels)
                    # print(1)
                    for m in range(n):
                        cube = img[i:i + ROI_size, j:j + ROI_size, k:k + ROI_size, m].flatten()
                        # print(m, cube)
                        good_voxel_num = 0
                        for x in range(cube.shape[0]):
                            if good_in_all_voxels[x]:
                                out[cnt_good_spots, good_voxel_num, m] = cube[x]
                                good_voxel_num += 1

                            # if cube[x] > HIGH_GLOBAL:
                            #     out[cnt_good_spots, x_good, m] = cube[x]
                            #     x_good += 1
                    # print(out[cnt_good_spots, :, m])
                    # print(cnt_good_spots)

                    ROI_sizes[cnt_good_spots] = num_good_voxels
                    # cnt_good_spots += 1
                    valid_voxel_idxs[cnt_good_spots, :] = np.array([i + size_over_two,
                                                                    j + size_over_two,
                                                                    k + size_over_two])
                    cnt_good_spots += 1

    out = out[:cnt_good_spots, :, :]
    valid_voxel_idxs = valid_voxel_idxs[:cnt_good_spots, :]
    ROI_sizes = ROI_sizes[:cnt_good_spots]
    return out, valid_voxel_idxs, ROI_sizes



@njit(fastmath=True, nopython=True, cache=True)
def make_RSMs(ROI_trial_vec, ROI_sizes,
              spearman=False, euc=False
              ):
    num_ROIS = ROI_trial_vec.shape[0]
    num_trials = ROI_trial_vec.shape[1]

    if spearman:
        ROI_trial_vec_ranked = np.full(ROI_trial_vec.shape, GLOBAL_NAN_VALUE, dtype=np.float32)
        for i in range(num_ROIS):
            ROI_size = ROI_sizes[i]
            for j in range(num_trials):
                ROI_trial_vec_ranked[i, j, :ROI_size] = np.argsort(
                    np.argsort(ROI_trial_vec[i, j, :ROI_size]))
        ROI_trial_vec = ROI_trial_vec_ranked

    out = np.full((num_ROIS, num_trials, num_trials),
                  GLOBAL_NAN_VALUE, dtype=np.float32)
    for i in range(num_ROIS):
        ROI_size = ROI_sizes[i]
        for j in range(num_trials):
            for k in range(j):
                # if j == k:
                #     out[i, j, k] = 1
                #     continue
                # elif j < k:
                #     continue

                if euc:
                    # num_points = 0
                    total_dif = 0
                    for x in range(ROI_size):
                        if ROI_trial_vec[i, j, x] < HIGH_GLOBAL:
                            raise ValueError
                        if ROI_trial_vec[i, k, x] < HIGH_GLOBAL:
                            raise ValueError
                        dif = np.abs(ROI_trial_vec[i, j, x] - ROI_trial_vec[i, k, x])
                        total_dif += dif ** 2
                        # num_points += 1
                    # if num_points == 0: continue
                    dist = np.sqrt(total_dif)
                    out[i, j, k] = dist
                    out[i, k, j] = dist
                else:

                    prod_sum = 0
                    j_sum = 0
                    jj_sum = 0
                    k_sum = 0
                    kk_sum = 0
                    num_points = 0
                    # print(ROI_size)

                    for x in range(ROI_size):
                        # if is_na[i, j, l] or is_na[i, k, l]:
                        #     continue
                        if ROI_trial_vec[i, j, x] < HIGH_GLOBAL:
                            raise ValueError
                        if ROI_trial_vec[i, k, x] < HIGH_GLOBAL:
                            raise ValueError
                        prod_sum += (ROI_trial_vec[i, j, x] *
                                     ROI_trial_vec[i, k, x])
                        jj_sum += ROI_trial_vec[i, j, x] ** 2
                        kk_sum += ROI_trial_vec[i, k, x] ** 2
                        j_sum += ROI_trial_vec[i, j, x]
                        k_sum += ROI_trial_vec[i, k, x]
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

@njit(fastmath=True, nopython=True, cache=True)
def do_synergy_RSM_x_RSM(ROI_trial_vec, ROI_sizes, do_only, RSM_stim,
                         do_synergy=True):
    num_ROIs = ROI_trial_vec.shape[0]
    num_trials = ROI_trial_vec.shape[1]

    out = np.full((num_ROIs, num_ROIs),
                  GLOBAL_NAN_VALUE, dtype=np.float32)

    idx_to_run = np.empty(num_trials)
    for t in range(num_trials):
        idx_to_run[t] = t // 38

    ROI_is_valid = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.bool_)
    # if do_synergy:
    ROI_prod_sum = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)
    ROI_j_sum = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)
    ROI_jj_sum = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)
    ROI_k_sum = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)
    ROI_kk_sum = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)
    ROI_num_points = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.int32)
    # else:
    ROI_r_fMRI = np.zeros((num_ROIs, num_trials, num_trials), dtype=np.float32)

    for i in range(num_ROIs):
        ROI_size = ROI_sizes[i]
        for j in range(num_trials):
            for k in range(j):
                if idx_to_run[j] == idx_to_run[k]:
                    continue
                # if j == k:
                #     out[i, j, k] = 1
                #     continue
                # elif j < k:
                #     continue
                prod_sum = 0
                j_sum = 0
                jj_sum = 0
                k_sum = 0
                kk_sum = 0
                num_points = 0
                for x in range(ROI_size):
                    prod_sum += (ROI_trial_vec[i, j, x] *
                                 ROI_trial_vec[i, k, x])
                    jj_sum += ROI_trial_vec[i, j, x] ** 2
                    kk_sum += ROI_trial_vec[i, k, x] ** 2
                    j_sum += ROI_trial_vec[i, j, x]
                    k_sum += ROI_trial_vec[i, k, x]
                    num_points += 1
                if num_points == 0: continue
                if np.abs(prod_sum) < 1e-12: # all zeroes in one
                    continue

                ROI_is_valid[i, j, k] = True
                if do_synergy:
                    ROI_prod_sum[i, j, k] = prod_sum
                    ROI_j_sum[i, j, k] = j_sum
                    ROI_jj_sum[i, j, k] = jj_sum
                    ROI_k_sum[i, j, k] = k_sum
                    ROI_kk_sum[i, j, k] = kk_sum
                    ROI_num_points[i, j, k] = num_points
                else:
                    E_JK = prod_sum / num_points
                    E_J = j_sum / num_points
                    E_K = k_sum / num_points
                    numerator = E_JK - (E_J * E_K)
                    E_JJ = jj_sum / num_points
                    E_KK = kk_sum / num_points
                    denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
                    r_fMRI0 = numerator / denominator
                    ROI_r_fMRI[i, j, k] = r_fMRI0

    for n0 in range(num_ROIs):
        # ROI_size0 = ROI_sizes[n0]
        for n1 in range(n0):
            # if n1 > 1: continue
            if not do_only[n0, n1]:
                continue
            # print(n0, n1)
            # ROI_size1 = ROI_sizes[n1]

            RSM_RSM_prod_sum = 0
            RSM_RSM_j_sum = 0
            RSM_RSM_jj_sum = 0
            RSM_RSM_k_sum = 0
            RSM_RSM_kk_sum = 0
            RSM_RSM_num_points = 0

            for j in range(num_trials):
                for k in range(j):
                    if idx_to_run[j] == idx_to_run[k]:
                        continue
                    # prod_sum0 = 0
                    # j_sum0 = 0
                    # jj_sum0 = 0
                    # k_sum0 = 0
                    # kk_sum0 = 0
                    # num_points0 = 0
                    # for x in range(ROI_size0):
                    #     prod_sum0 += (ROI_trial_vec[n0, j, x] *
                    #                  ROI_trial_vec[n0, k, x])
                    #     jj_sum0 += ROI_trial_vec[n0, j, x] ** 2
                    #     kk_sum0 += ROI_trial_vec[n0, k, x] ** 2
                    #     j_sum0 += ROI_trial_vec[n0, j, x]
                    #     k_sum0 += ROI_trial_vec[n0, k, x]
                    #     num_points0 += 1
                    # if num_points0 == 0: continue
                    # if np.abs(prod_sum0) < 1e-12:  # all zeroes in one
                    #     continue
                    #
                    # prod_sum1 = 0
                    # j_sum1 = 0
                    # jj_sum1 = 0
                    # k_sum1 = 0
                    # kk_sum1 = 0
                    # num_points1 = 0
                    # for x in range(ROI_size1):
                    #     prod_sum1 += (ROI_trial_vec[n1, j, x] *
                    #                  ROI_trial_vec[n1, k, x])
                    #     jj_sum1 += ROI_trial_vec[n1, j, x] ** 2
                    #     kk_sum1 += ROI_trial_vec[n1, k, x] ** 2
                    #     j_sum1 += ROI_trial_vec[n1, j, x]
                    #     k_sum1 += ROI_trial_vec[n1, k, x]
                    #     num_points1 += 1
                    # if num_points1 == 0:
                    #     continue
                    # if np.abs(prod_sum1) < 1e-12:  # all zeroes in one
                    #     continue
                    if not ROI_is_valid[n0, j, k] or not ROI_is_valid[n1, j, k]:
                        continue



                    if do_synergy:
                        prod_sum0 = ROI_prod_sum[n0, j, k]
                        j_sum0 = ROI_j_sum[n0, j, k]
                        jj_sum0 = ROI_jj_sum[n0, j, k]
                        k_sum0 = ROI_k_sum[n0, j, k]
                        kk_sum0 = ROI_kk_sum[n0, j, k]
                        num_points0 = ROI_num_points[n0, j, k]

                        prod_sum1 = ROI_prod_sum[n1, j, k]
                        j_sum1 = ROI_j_sum[n1, j, k]
                        jj_sum1 = ROI_jj_sum[n1, j, k]
                        k_sum1 = ROI_k_sum[n1, j, k]
                        kk_sum1 = ROI_kk_sum[n1, j, k]
                        num_points1 = ROI_num_points[n1, j, k]

                        # print(num_points0, num_points1)

                        prod_sum = prod_sum0 + prod_sum1
                        jj_sum = jj_sum0 + jj_sum1
                        kk_sum = kk_sum0 + kk_sum1
                        j_sum = j_sum0 + j_sum1
                        k_sum = k_sum0 + k_sum1
                        num_points = num_points0 + num_points1

                        E_JK = prod_sum / num_points
                        E_J = j_sum / num_points
                        E_K = k_sum / num_points
                        numerator = E_JK - (E_J * E_K)
                        E_JJ = jj_sum / num_points
                        E_KK = kk_sum / num_points
                        denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
                        # print(E_JJ)
                        # print(E_KK)

                        r_fMRI = numerator / denominator
                    else:
                        r_fMRI0 = ROI_r_fMRI[n0, j, k]
                        r_fMRI1 = ROI_r_fMRI[n1, j, k]

                        # E_JK = prod_sum0 / num_points0
                        # E_J = j_sum0 / num_points0
                        # E_K = k_sum0 / num_points0
                        # numerator = E_JK - (E_J * E_K)
                        # E_JJ = jj_sum0 / num_points0
                        # E_KK = kk_sum0 / num_points0
                        # denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
                        # r_fMRI0 = numerator / denominator
                        #
                        # E_JK = prod_sum1 / num_points1
                        # E_J = j_sum1 / num_points1
                        # E_K = k_sum1 / num_points1
                        # numerator = E_JK - (E_J * E_K)
                        # E_JJ = jj_sum1 / num_points1
                        # E_KK = kk_sum1 / num_points1
                        # denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
                        # r_fMRI1 = numerator / denominator

                        r_fMRI = r_fMRI0 + r_fMRI1

                    r_stim = RSM_stim[j, k]

                    RSM_RSM_prod_sum += r_fMRI * r_stim
                    RSM_RSM_jj_sum += r_fMRI ** 2
                    RSM_RSM_kk_sum += r_stim ** 2
                    RSM_RSM_j_sum += r_fMRI
                    RSM_RSM_k_sum += r_stim
                    RSM_RSM_num_points += 1

            if RSM_RSM_num_points == 0: continue
            if np.abs(RSM_RSM_prod_sum) < 1e-12:  # all zeroes in one
                # print('SKIP 1')
                continue
            # print()
            # print()
            # print(n0, n1)
            # print(RSM_RSM_prod_sum)
            # print(RSM_RSM_num_points)
            # print(RSM_RSM_j_sum)
            # print(RSM_RSM_k_sum)
            # print(RSM_RSM_jj_sum)
            # print(RSM_RSM_kk_sum)

            E_JK = RSM_RSM_prod_sum / RSM_RSM_num_points
            E_J = RSM_RSM_j_sum / RSM_RSM_num_points
            E_K = RSM_RSM_k_sum / RSM_RSM_num_points
            numerator = E_JK - (E_J * E_K)
            # if numerator < 1e-12: # ??? idk why this can trigger
            #     print('SKIP 2')
            #     print(E_JK, E_J, E_K)
            #     continue
            # print(numerator)
            E_JJ = RSM_RSM_jj_sum / RSM_RSM_num_points
            E_KK = RSM_RSM_kk_sum / RSM_RSM_num_points
            denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
            out[n0, n1] = numerator / denominator
            out[n1, n0] = numerator / denominator

    return out



@njit(fastmath=True, nopython=True, cache=True)
def numba_fast_RSM_x_RSM(RSMs, do_only):
    n_ROI = RSMs.shape[0]
    n_trials = RSMs.shape[1]
    idx_to_run = np.empty(n_trials)
    for t in range(n_trials):
        idx_to_run[t] = t // 38

    out = np.full((n_ROI, n_ROI), GLOBAL_NAN_VALUE, dtype=np.float32)
    for j in range(n_ROI):
        for k in range(j):
            if not do_only[j, k]:
                continue
            prod_sum = 0
            j_sum = 0
            jj_sum = 0
            k_sum = 0
            kk_sum = 0
            num_points = 0
            for t0 in range(n_trials):
                for t1 in range(n_trials):
                    if idx_to_run[t0] == idx_to_run[t1]:
                        continue
                    prod_sum += (RSMs[j, t0, t1] *
                                 RSMs[k, t0, t1])
                    jj_sum += RSMs[j, t0, t1] ** 2
                    kk_sum += RSMs[k, t0, t1] ** 2
                    j_sum += RSMs[j, t0, t1]
                    k_sum += RSMs[k, t0, t1]
                    num_points += 1

            if num_points == 0: continue
            if np.abs(prod_sum) < 1e-12:  # all zeroes in one
                continue

            E_JK = prod_sum / num_points
            E_J = j_sum / num_points
            E_K = k_sum / num_points
            numerator = E_JK - (E_J * E_K)
            E_JJ = jj_sum / num_points
            E_KK = kk_sum / num_points
            denominator = np.sqrt((E_JJ - E_J ** 2) * (E_KK - E_K ** 2))
            out[j, k] = numerator / denominator
            out[k, j] = numerator / denominator
    return out


def prep_RSM_synergy_RSM_stim(region, fp, sn, ROI_size, max_dist=15, smallest_cube=6,
                              size_limit=False, semantic=True, layer=None,
                              stdize_by_run=False):

    img_mat, ROI_sizes_, ROI_sizes, within_range, euc_mtx, valid_voxel_idxs = (
        org_size2_data(region, fp, sn, ROI_size, max_dist, smallest_cube, size_limit,
                       stdize_by_run=stdize_by_run))

    print(img_mat.shape)
    # print(img_mat[:, :, :38].shape)
    # quit()

    RSM_stim = get_sn_fp_stim_RSM(sn, fp, semantic=semantic, layer=layer)
    # if stdize_by_run:
    #     run0 = stats.zscore(img_mat[:, :38, :], axis=1, nan_policy='omit')
    #     run1 = stats.zscore(img_mat[:, 38:76, :], axis=1, nan_policy='omit')
    #     run2 = stats.zscore(img_mat[:, 76:114, :], axis=1, nan_policy='omit')
    #     img_mat = np.concatenate((run0, run1, run2), axis=1)
    # print(img_mat.shape)
    # img_mat = img_mat[:100]
    # print(img_mat)
    # quit()
    # print(f'{img_mat.shape=}')
    # print(img_mat)
    # quit()
    t_st = time()
    # plt.imshow()
    # print(img_mat.shape)
    # print(img_mat)
    # quit()
    synergies = do_synergy_RSM_x_RSM(img_mat, ROI_sizes, within_range, RSM_stim, do_synergy=True)
    no_synergies = do_synergy_RSM_x_RSM(img_mat, ROI_sizes, within_range, RSM_stim,
                                        do_synergy=False)

    synergies[synergies < HIGH_GLOBAL] = np.nan
    t_end = time()
    print(f'Synergy RSMs x RSM speed: {t_end - t_st=:.3f} s')
    # plt.imshow(synergies)
    # plt.colorbar()
    # plt.show()
    #
    no_synergies[no_synergies < HIGH_GLOBAL] = np.nan
    # plt.imshow(no_synergies)
    # plt.colorbar()
    # plt.show()
    # quit()

    synergy_ef = synergies - no_synergies

    return synergy_ef, euc_mtx, valid_voxel_idxs, ROI_sizes

    # quit()


def org_size2_data(region, fp, sn, ROI_size, max_dist=15, smallest_cube=6,
                   size_limit=False, stdize_by_run=False):

    img = get_img_region(region, fp, sn)
    # num_nans = np.sum(np.isnan(img))
    # print(f'Num nans: {num_nans}')
    # num_non_nans = np.prod(img.shape) - num_nans
    # print(f'Proportion non-nans: {num_non_nans / np.prod(img.shape):.1%}')
    # print(f'{ROI_size=}')
    # print(f'{smallest_cube=}')
    # print(img.shape)
    # plt.imshow(img[:, :, 50, 0])
    # plt.show()
    # quit()
    img[np.isnan(img)] = GLOBAL_NAN_VALUE
    img_mat, valid_voxel_idxs, ROI_sizes = parse_into_cubes(img, ROI_size=ROI_size,
                                                            threshold=smallest_cube)
    img_mat[img_mat < HIGH_GLOBAL] = np.nan
    # print(img_mat.shape)
    # print(ROI_sizes.shape)
    # quit()
    if stdize_by_run:
        run0 = stats.zscore(img_mat[:, :, :38], axis=-1, nan_policy='omit')
        run1 = stats.zscore(img_mat[:, :, 38:76], axis=-1, nan_policy='omit')
        run2 = stats.zscore(img_mat[:, :, 76:114], axis=-1, nan_policy='omit')
        img_mat = np.concatenate((run0, run1, run2), axis=-1)
    img_mat[np.isnan(img_mat)] = GLOBAL_NAN_VALUE

    img_mat = np.transpose(img_mat, (0, 2, 1))
    if size_limit:
        ROI_sizes_ = np.copy(ROI_sizes)
        ROI_sizes_[:] = smallest_cube
    else:
        ROI_sizes_ = ROI_sizes

    # print(valid_voxel_idxs)
    euc_mtx = get_idx2euc_custom(valid_voxel_idxs)
    # plt.imshow(euc_mtx)
    # plt.colorbar()
    # plt.show()
    # quit()
    within_range = euc_mtx < max_dist
    num_within = np.sum(within_range)
    num_all = np.prod(within_range.shape)
    print(f'Proportion within range ({max_dist} voxels): {num_within / num_all:.1%}')
    return img_mat, ROI_sizes_, ROI_sizes, within_range, euc_mtx, valid_voxel_idxs


def prep_RSM_x_RSM(region, fp, sn, ROI_size, max_dist=15, smallest_cube=6,
                   size_limit=False, spearman=False, stdize_by_run=False,
                   euc=False):

    if euc:
        assert size_limit
        assert not spearman

    img_mat, ROI_sizes_, ROI_sizes, within_range, euc_mtx, valid_voxel_idxs = (
        org_size2_data(region, fp, sn, ROI_size, max_dist, smallest_cube, size_limit,
                       stdize_by_run))
    # print(ROI_sizes_)
    RSMs = make_RSMs(img_mat, ROI_sizes_, spearman=spearman, euc=euc)

    t_st = time()
    RSM_x_RSM_map = numba_fast_RSM_x_RSM(RSMs, within_range)
    t_end = time()
    print(f'RSM x RSM speed: {t_end - t_st=:.3f} s')
    RSM_x_RSM_map[RSM_x_RSM_map < HIGH_GLOBAL] = np.nan
    return RSM_x_RSM_map, euc_mtx, valid_voxel_idxs, ROI_sizes


def get_dist2MI_v2_RSA(region, fp, sn, ROI_size, max_dist=15, smallest_cube=6,
                       size_limit=True, semantic=True, layer=None,
                       stdize_by_run=True):
    # RSM_x_RSM_map, euc_mtx, valid_voxel_idxs, ROI_sizes = (
    #     prep_RSM_x_RSM(region, fp, sn, ROI_size, max_dist, smallest_cube))
    synergy_ef, euc_mtx, valid_voxel_idxs, ROI_sizes = (
        utils.pickle_wrap(prep_RSM_synergy_RSM_stim,
                          kwargs={'region': region, 'fp': fp, 'sn': sn, 'ROI_size': ROI_size,
                                  'max_dist': max_dist, 'smallest_cube': smallest_cube,
                                  'size_limit': size_limit, 'semantic': semantic,
                                  'layer': layer, 'stdize_by_run': stdize_by_run},
                          verbose=-1, easy_override=False))
    # plt.imshow(synergy_ef, aspect='auto', interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(euc_mtx, aspect='auto', interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # euc_mtx = euc_mtx[:100, :100]

    vox_dist2MI, cnter = get_voxel_dist2MI(synergy_ef, euc_mtx,
                                           max_dist=max_dist)
    vox_dist2MI[vox_dist2MI < HIGH_GLOBAL] = np.nan

    # plt.imshow(vox_dist2MI, aspect='auto', interpolation='none')
    # plt.colorbar()
    # plt.show()
    # quit()
    # plt.imshow(synergy_ef, aspect='auto', interpolation='none')
    # plt.show()
    # quit()
    return vox_dist2MI, None, valid_voxel_idxs, cnter

def get_dist2MI_v2(region, fp, sn, ROI_size, max_dist=15, smallest_cube=6,
                   size_limit=True, spearman=False, stdize_by_run=False,
                   euc=False):
    # RSM_x_RSM_map, euc_mtx, valid_voxel_idxs, ROI_sizes = (
    #     prep_RSM_x_RSM(region, fp, sn, ROI_size, max_dist, smallest_cube))
    RSM_x_RSM_map, euc_mtx, valid_voxel_idxs, ROI_sizes = (
        utils.pickle_wrap(prep_RSM_x_RSM, kwargs={'region': region, 'fp': fp,
                                                  'sn': sn, 'ROI_size': ROI_size,
                                                  'max_dist': max_dist,
                                                  'smallest_cube': smallest_cube,
                                                  'size_limit': size_limit,
                                                  'spearman': spearman,
                                                  'stdize_by_run': stdize_by_run,
                                                  'euc': euc},
                          verbose=-1, easy_override=False))
    # plt.imshow(RSM_x_RSM_map, aspect='auto', interpolation='none')
    # plt.colorbar()
    # plt.show()
    # quit()

    vox_dist2MI, cnter = get_voxel_dist2MI(RSM_x_RSM_map, euc_mtx,
                                           max_dist=max_dist)
    vox_dist2MI[vox_dist2MI < HIGH_GLOBAL] = np.nan
    return vox_dist2MI, None, valid_voxel_idxs, cnter


def run_RSA_map_all_sn(region, ROI_size=3, max_dist=20,
                       min_vox=1, max_vox=20, smallest_cube=20,
                       size_limit=True, RSA=False, spearman=False,
                       euc=False
                       ):
# def run_RSA_map_all_sn(region, ROI_size=3, max_dist=20,
#                        min_vox=1, max_vox=10, smallest_cube=5,
#                        size_limit=True, RSA=False):
    # if 'OC_IT' in region:
    #     max_vox = 50

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI'] #
    bad_tups = [('224', 'obj7_fMRI'), ('234', 'obj7_fMRI'), ('132', 'bl7_fMRI'),
                ('138', 'vis7_fMRI'), ('132', 'obj7_fMRI')]
    # fps = ['obj7_fMRI', 'con7_fMRI',  'vis7_fMRI']
    # fps = ['con7_fMRI']
    # sns = sns[::-1]
    # sns = sns[5::6]
    # sns = ['132']
    img_data_l = []
    for i, sn in tqdm(enumerate(sns)):
        if sn in ['136']: break
        # if sn in ['106']: break
        sn_img_data = []
        for j, fp in enumerate(fps):
            if (sn, fp) in bad_tups:
                print(f'Bad: {(sn, fp)})')
                continue
            print(f'{sn=}, {fp=}')

            kw = {'sn': sn, 'fp': fp, 'region': region, 'ROI_size': ROI_size,
                  'max_dist': max_dist, 'smallest_cube': smallest_cube,
                  'size_limit': size_limit, 'stdize_by_run': True,
                  'euc': euc}

            if RSA:
                vox_dist2MI, _, vox_idxs, cnter = (
                    utils.pickle_wrap(get_dist2MI_v2_RSA, None, kwargs=kw, verbose=-1,
                                      easy_override=False))
            else:
                kw['spearman'] = spearman
                vox_dist2MI, _, vox_idxs, cnter = (
                    utils.pickle_wrap(get_dist2MI_v2, None, kwargs=kw, verbose=-1,
                                      easy_override=False))

            # plt.imshow(vox_dist2MI, aspect='auto', interpolation='none')
            # plt.colorbar()
            # plt.show()
            # quit()


            vox_dist2MI -= np.nanmean(vox_dist2MI, axis=0)
            # print(vox_dist2MI)
            # quit()
            # vox_dist2MI = stats.zscore(vox_dist2MI, axis=0)
            vox_idxs = np.array(vox_idxs // ROI_size, dtype=np.int8)
            img_data = idxs2img(vox_dist2MI, vox_idxs,
                                max_vox=max_vox, min_vox=min_vox,
                                downsample_rate=ROI_size)
            # print(img_data)
            # plt.imshow(img_data[:, :, 32], aspect='auto', interpolation='none')
            # plt.show()
            # quit()
            # print(img_data)
            # quit()
            sn_img_data.append(img_data)
        img_data = np.nanmean(sn_img_data, axis=0)
        img_data_l.append(img_data)

    num_subj = np.sum(~np.isnan(img_data_l), axis=0)
    img_data = (np.nanmean(img_data_l, axis=0) /
                np.nanstd(img_data_l, axis=0) *
                np.sqrt(num_subj))
    # img_data = np.nanmean(img_data_l, axis=0)
    # img_data -= np.nanmean(img_data)


    # print(img_data)
    # plt.imshow(img_data[:, :, 32], aspect='auto', interpolation='none')
    # plt.show()
    # quit()


    img_data[np.isinf(img_data)] = np.nan
    vmin = np.nanquantile(img_data, .01)
    print(f'{vmin=:.3f}')
    vmax = np.nanquantile(img_data, .99)
    print(f'{vmax=:.3f}')
    # quit()

    vabs = np.max(np.abs([vmin, vmax]))


    img_data[np.isnan(img_data)] = 0

    # img_data[img_data < vmin] = vmin
    # img_data[img_data > vmax] = vmax
    # plt.hist(img_data.flatten())
    # plt.show()
    # quit()

    # img_data *= 10

    # plt.imshow(img_data[20, :, :], aspect='auto', interpolation='none',)
    # plt.show()
    # quit()

    img = image.new_img_like(get_atlas()['maps'], img_data)
    # img = image.smooth_img(img, fwhm=1)

    if vmax > 2:
        print('TEST')
        # pass
        # img = image.smooth_img(img, fwhm=2)
        # img = image.threshold_img(img, threshold=2, copy=False, cluster_threshold=100)
    else:
        # pass
        img = image.threshold_img(img, threshold=0.01, copy=False, cluster_threshold=20)

    fig, axs = plotting.plot_img_on_surf(img,
                                         # vmin=vmin, vmax=vmax,
                                         vmin=-vabs, vmax=vabs,
                                         # vmin=-0.1, vmax=0.2,
                                         # vmin=0.05, vmax=0.13,
                                         # inflate=True,
                                         # views=['posterior', 'ventral'],
                                         surf_mesh='fsaverage5',
                                         avg_method='median',
                                         hemispheres=['right' if '_R' in region else 'left'],
                                         # cmap='cold_hot_r',
                                         cmap='turbo_r',
                                         cbar_tick_format='%.1f',
                                         # threshold=-1,
                                         threshold=0.001,
                                         # threshold=1 if vmax > 2 else 0.0,
                                         )
    plt.show()
    return

    view = plotting.view_img(img, threshold=0, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             vmin=vmin, vmax=vmax,
                             # vmin=-5, vmax=5,
                             cmap='cold_hot_r'
                             )
    view.open_in_browser()


    # quit()



if __name__ == '__main__':
    # spearman makes no difference

    # run_RSA_map_all_sn('OC_IT_L', ROI_size=2, smallest_cube=6, RSA=True, size_limit=True)
    # run_RSA_map_all_sn('cortical_L', ROI_size=2, smallest_cube=6, RSA=True, size_limit=True)

    # TODO:

    # Promising:
    # run_RSA_map_all_sn('cortical_L', ROI_size=2, smallest_cube=6, size_limit=True,
    #                    )

    # 3/20

    # DID OVERNIGHT 12/1/2024
    #
    # run_RSA_map_all_sn('cortical_L', ROI_size=2, smallest_cube=6, size_limit=True, euc=False)
    # run_RSA_map_all_sn('cortical_L', ROI_size=3, smallest_cube=20, size_limit=True, euc=False)
    # run_RSA_map_all_sn('cortical_L', ROI_size=3, smallest_cube=20, size_limit=True, euc=True)

    # run_RSA_map_all_sn('cortical_R', ROI_size=2, smallest_cube=6, size_limit=False)

    # run_RSA_map_all_sn('PFC_L', ROI_size=2, smallest_cube=6, size_limit=True, euc=False)
    run_RSA_map_all_sn('OC_T_R', ROI_size=2, smallest_cube=6, size_limit=True, euc=False)

    # run_RSA_map_all_sn('PFC_L', ROI_size=3, smallest_cube=20, size_limit=True, euc=False)
    # run_RSA_map_all_sn('OC_T_R', ROI_size=3, smallest_cube=20, size_limit=True, euc=False)
