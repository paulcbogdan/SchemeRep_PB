from Utils.atlas_funcs import get_atlas
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
              # spearman=True
              ):
    num_ROIS = ROI_trial_vec.shape[0]
    num_trials = ROI_trial_vec.shape[1]

    # if spearman:
    #     ROI_trial_vec_ranked = np.full(ROI_trial_vec.shape,
    #                                    GLOBAL_NAN_VALUE, dtype=int)
    #     for i in range(num_ROIS):
    #         ROI_size = ROI_sizes[i]
    #         for j in range(num_trials):
    #             ROI_trial_vec_ranked[i, j, :ROI_size] = np.argsort(
    #                 np.argsort(ROI_trial_vec[i, j, :ROI_size]))
    #     ROI_trial_vec = ROI_trial_vec_ranked

    out = np.full((num_ROIS, num_trials, num_trials),
                  GLOBAL_NAN_VALUE, dtype=int)
    for i in range(num_ROIS):
        ROI_size = ROI_sizes[i]
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
                    # if is_na[i, j, l] or is_na[i, k, l]:
                    #     continue
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

@njit(fastmath=True, nopython=True, cache=True)
def numba_fast_RSM_x_RSM(RSMs, do_only):
    n_ROI = RSMs.shape[0]
    n_trials = RSMs.shape[1]
    idx_to_run = np.empty(n_trials)
    for t in range(n_trials):
        idx_to_run[t] = t // 38
    # print(idx_to_run)
    # quit()

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


def get_dist2MI_v2(region, fp, sn, ROI_size, max_dist=15):
    # vox = true voxel size, not downsampled
    img = get_img_region(region, fp, sn)
    img[np.isnan(img)] = GLOBAL_NAN_VALUE
    img_mat, valid_voxel_idxs, ROI_sizes = parse_into_cubes(img, ROI_size=ROI_size)
    img_mat = np.transpose(img_mat, (0, 2, 1))
    # img_mat = img_mat[:255]

    RSMs = make_RSMs(img_mat, ROI_sizes)
    t_st = time()
    euc_mtx = get_idx2euc_custom(valid_voxel_idxs)
    within_range = euc_mtx < max_dist
    t_end = time()
    print(f'Time to define range: {t_end - t_st=:.3f} s')
    num_within = np.sum(within_range)
    num_all = np.prod(within_range.shape)
    print(f'Proportion within range ({max_dist} voxels): {num_within / num_all:.1%}')

    t_st = time()
    RSM_x_RSM_map = numba_fast_RSM_x_RSM(RSMs, within_range)
    t_end = time()
    print(f'RSM x RSM speed: {t_end - t_st=:.3f} s')
    RSM_x_RSM_map[RSM_x_RSM_map < HIGH_GLOBAL] = np.nan

    vox_dist2MI, cnter = get_voxel_dist2MI(RSM_x_RSM_map, euc_mtx,
                                           max_dist=max_dist)
    # vox_dist2MI[np.isnan(vox_dist2MI)] = GLOBAL_NAN_VALUE
    vox_dist2MI[vox_dist2MI < HIGH_GLOBAL] = np.nan
    return vox_dist2MI, None, valid_voxel_idxs, cnter


def run_RSA_map_all_sn(region, ROI_size=2, max_dist=30,
                       min_vox=1, max_vox=10):
    # if 'OC_IT' in region:
    #     max_vox = 50
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # sns = sns[7::8]
    img_data_l = []
    for i, sn in enumerate(sns):
        # if sn in ['112']: break
        print(f'{sn=}')
        sn_img_data = []
        for j, fp in enumerate(fps):
            kw = {'sn': sn, 'fp': fp, 'region': region, 'ROI_size': ROI_size,
                  'max_dist': max_dist}
            vox_dist2MI, _, vox_idxs, cnter = (
                utils.pickle_wrap(get_dist2MI_v2, None, kwargs=kw, verbose=-1,
                                  easy_override=False))
            vox_dist2MI -= np.nanmean(vox_dist2MI, axis=0)
            vox_idxs = np.array(vox_idxs // ROI_size, dtype=np.int8)
            img_data = idxs2img(vox_dist2MI, vox_idxs,
                                max_vox=max_vox, min_vox=min_vox,
                                downsample_rate=ROI_size)
            sn_img_data.append(img_data)
        img_data = np.nanmean(sn_img_data, axis=0)
        img_data_l.append(img_data)

    num_subj = np.sum(~np.isnan(img_data_l), axis=0)
    img_data = (np.nanmean(img_data_l, axis=0) /
                np.nanstd(img_data_l, axis=0) *
                np.sqrt(num_subj))
    img_data[np.isinf(img_data)] = np.nan
    vmin = np.nanquantile(img_data, .01)
    # print(f'{vmin=:.3f}')
    vmax = np.nanquantile(img_data, .99)
    # print(f'{vmax=:.3f}')
    img_data[img_data < vmin] = vmin
    img_data[img_data > vmax] = vmax
    # plt.hist(img_data.flatten())
    # plt.show()
    # quit()

    img = image.new_img_like(get_atlas()['maps'], img_data)
    fig, axs = plotting.plot_img_on_surf(img,
                                         vmin=-10, vmax=10,
                                         inflate=False,
                                         surf_mesh='fsaverage5',
                                         avg_method='median',
                                         hemispheres=['right' if '_R' in region else 'left'],
                                         cmap='cold_hot_r'
                                         # threshold=3,
                                         )
    plt.show()

    view = plotting.view_img(img, threshold=0, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             vmin=vmin, vmax=vmax,
                             cmap='cold_hot_r'
                             )
    view.open_in_browser()
    quit()



if __name__ == '__main__':
    run_RSA_map_all_sn('cortical_L', ROI_size=2)

