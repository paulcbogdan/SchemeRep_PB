from numba import jit, njit, prange, config

import numpy as np

config.CACHE_DIR = r'H:\PycharmProjects_H\SchemeRep\cache\numba_test'

NAN_VAL = 10000001

CACHE_NUMBA = True
@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_models_searchlight(fMRI_RDMs, stim_RDM):
    num_centers = fMRI_RDMs.shape[0]
    rs = np.empty(num_centers)
    for i in range(num_centers):
        r = np.corrcoef(fMRI_RDMs[i], stim_RDM)[0, 1]
        rs[i] = r
    return rs

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_models_searchlight_spear(fMRI_RDMs, stim_RDM):
    stim_RDM = np.argsort(stim_RDM)
    num_centers = fMRI_RDMs.shape[0]
    rs = np.empty(num_centers)
    for i in range(num_centers):
        r = np.corrcoef(np.argsort(np.argsort(fMRI_RDMs[i])),
                        stim_RDM)[0, 1]
        rs[i] = r
    return rs

@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_regresslight(fMRI_RDMs1, fMRI_RDMs2, stim_RDM):
    num_centers = fMRI_RDMs1.shape[0]
    ones = np.ones(fMRI_RDMs1.shape[1])
    betas = np.empty((num_centers, 3))
    for i in range(num_centers):
        X = np.vstack((ones, fMRI_RDMs1[i], fMRI_RDMs2[i])).T
        solution, residuals, rank, s = np.linalg.lstsq(X, stim_RDM)
        betas[i, :] = solution
    return betas

@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_regresslight_std(fMRI_RDMs1, fMRI_RDMs2, stim_RDM):
    stim_RDM_ = (stim_RDM - np.mean(stim_RDM)) / np.std(stim_RDM)
    num_centers = fMRI_RDMs1.shape[0]
    ones = np.ones(fMRI_RDMs1.shape[1])
    betas = np.empty((num_centers, 3))
    for i in range(num_centers):
        X = np.vstack((ones,
                       (fMRI_RDMs1[i] - np.mean(fMRI_RDMs1[i])) / np.std(fMRI_RDMs1[i]),
                       (fMRI_RDMs2[i] - np.mean(fMRI_RDMs2[i])) / np.std(fMRI_RDMs2[i])
                       )).T
        solution, residuals, rank, s = np.linalg.lstsq(X, stim_RDM_)
        betas[i, :] = solution
    return betas

@njit(parallel=True, fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_multilight_std(fMRI_RDMs1, regressors, stim_RDM):
    stim_RDM_ = (stim_RDM - np.mean(stim_RDM)) / np.std(stim_RDM)
    num_centers = fMRI_RDMs1.shape[0]
    ones = np.ones(fMRI_RDMs1.shape[1])
    betas = np.empty((num_centers, 2 + regressors.shape[1]))
    for i in prange(num_centers):
        X = np.empty((fMRI_RDMs1.shape[1], 2 + regressors.shape[1]))
        X[:, 0] = ones
        X[:, 1] = (fMRI_RDMs1[i] - np.mean(fMRI_RDMs1[i])) / np.std(fMRI_RDMs1[i])
        for j in range(regressors.shape[1]):
            X[:, 2 + j] = ((regressors[i, j] - np.mean(regressors[i, j])) /
                           np.std(regressors[i, j]))
        solution, residuals, rank, s = np.linalg.lstsq(X, stim_RDM_)
        betas[i, :] = solution
    return betas


@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def jit_searchlight_RDMs(data_2d, neighbors):
    num_trials = data_2d.shape[0]
    num_centers = neighbors.shape[0]
    neighbor_len = neighbors.shape[1]
    RDM_size = (num_trials - 1) * num_trials // 2
    out = np.full((num_centers, RDM_size), NAN_VAL, dtype=np.float32)

    for i in range(num_centers):
        for k in range(neighbor_len):
            if neighbors[i, k] == NAN_VAL:
                cutoff = k
                break
        else:
            cutoff = neighbor_len
        neighbors_i = neighbors[i, :cutoff]
        sphere_data = np.empty((num_trials, cutoff))
        for j in range(num_trials): # TODO: change data 2d to have nans
            sphere_data[j] = data_2d[j, neighbors_i]

        sphere_RDM = np.corrcoef(sphere_data)
        RDM_flat = np.empty(RDM_size)
        cnt = 0
        for j in range(114):
            for k in range(j):
                RDM_flat[cnt] = sphere_RDM[j, k]
                cnt += 1
        out[i] = RDM_flat
    return out

@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def jit_searchlight_RDMs_euc(data_2d, neighbors):
    num_trials = data_2d.shape[0]
    num_centers = neighbors.shape[0]
    neighbor_len = neighbors.shape[1]
    RDM_size = (num_trials - 1) * num_trials // 2
    out = np.full((num_centers, RDM_size), NAN_VAL, dtype=np.float32)

    for i in range(num_centers):
        for k in range(neighbor_len):
            if neighbors[i, k] == NAN_VAL:
                cutoff = k
                break
        else:
            cutoff = neighbor_len
        neighbors_i = neighbors[i, :cutoff]
        sphere_data = np.empty((num_trials, cutoff))
        for j in range(num_trials): # TODO: change data 2d to have nans
            sphere_data[j] = data_2d[j, neighbors_i]

        RDM_flat = np.empty(RDM_size)
        cnt = 0
        for j in range(num_trials):
            for k in range(j):
                RDM_flat[cnt] = np.linalg.norm(sphere_data[j] -
                                               sphere_data[k])
                cnt += 1

        # sphere_RDM = np.corrcoef(sphere_data)
        # RDM_flat = np.empty(RDM_size)
        # cnt = 0
        # for j in range(114):
        #     for k in range(j):
        #         RDM_flat[cnt] = sphere_RDM[j, k]
        #         cnt += 1
        out[i] = RDM_flat
    return out

@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def jit_searchlight_orged_RDMs(data_ready):
    num_trials = data_ready.shape[0]
    num_centers = data_ready.shape[1]
    RDM_size = (num_trials - 1) * num_trials // 2
    out = np.full((num_centers, RDM_size), NAN_VAL, dtype=np.float32)
    for i in range(num_centers):
        sphere_data = data_ready[:, i, :]
        sphere_RDM = np.corrcoef(sphere_data)
        RDM_flat = np.empty(RDM_size)
        cnt = 0
        for j in range(114):
            for k in range(j):
                RDM_flat[cnt] = sphere_RDM[j, k]
                cnt += 1
        out[i] = RDM_flat
    return out

@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def prep_data_Ms(data_2d, neighbors):
    num_trials = data_2d.shape[0]
    num_centers = neighbors.shape[0]
    out = np.empty((num_trials, num_centers), dtype=np.float32)
    for i in range(num_centers):
        for k in range(neighbors.shape[1]):
            if neighbors[i, k] == NAN_VAL:
                cutoff = k
                break
        else:
            cutoff = neighbors.shape[1]
        neighbors_i = neighbors[i, :cutoff]
        for j in range(num_trials):
            out[j, i] = np.mean(data_2d[j, neighbors_i])
    return out


@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def jit_volume_searchlight(mask, radius=2, threshold=0.5, resample=1,
                           cube=True):
    X_len = mask.shape[0]
    Y_len = mask.shape[1]
    Z_len = mask.shape[2]
    voxel2idx = np.full(mask.shape, NAN_VAL, dtype=np.int32)
    cnt = 0
    cnt_all = 0
    centers = np.full((X_len * Y_len * Z_len, 3), NAN_VAL, dtype=np.int32)
    for x in range(0, X_len):
        for y in range(0, Y_len):
            for z in range(0, Z_len):
                if mask[x, y, z]:
                    centers[cnt, :] = [x, y, z]
                    cnt += 1
                voxel2idx[x, y, z] = cnt_all
                cnt_all += 1

    centers = centers[:cnt]

    # Define sphere as points relative to a center (e.g., [-1, 0, 2])
    radius_sq = radius * radius
    rel_points = np.full(((radius * 2) ** 3, 3), NAN_VAL, dtype=np.int32)
    # print(f'{rel_points.shape=}')
    # print(rel_points.shape)
    # print(radius)
    cnt = 0
    for x in range(-radius, radius + (0 if cube else 1)):
        for y in range(-radius, radius + (0 if cube else 1)):
            for z in range(-radius, radius + (0 if cube else 1)):
                if (x*x + y*y + z*z <= radius_sq) or cube:
                    rel_points[cnt, :] = [x, y, z]
                    cnt += 1
    rel_points = rel_points[:cnt]
    max_cnt = rel_points.shape[0]
    # print(rel_points.shape)

    if resample > 1:
        idxs = np.arange(0, centers.shape[0])
        idxs = np.random.choice(idxs, idxs.shape[0] // resample, replace=False)
        centers_new = np.empty((idxs.shape[0], 3), dtype=np.int32)
        for i, idx in enumerate(idxs):
            centers_new[i] = centers[idx]

    # Below messes up
    neighbors = np.full((centers.shape[0], max_cnt), NAN_VAL, dtype=np.int32)
    good_neighbors = np.full((centers.shape[0], max_cnt), NAN_VAL, dtype=np.int32)
    good_centers = np.full(centers.shape, NAN_VAL, dtype=np.int32)
    thresh_int = int(threshold * max_cnt)
    voxel2idx_centers = np.full(mask.shape, NAN_VAL, dtype=np.int32)

    cnt_good_center = 0
    for cnt_center, center in enumerate(centers):
        cnt = 0
        for rel_point in rel_points:
            if (center[0] + rel_point[0] < 0 or
                    center[1] + rel_point[1] < 0 or
                    center[2] + rel_point[2] < 0):
                continue
            if (center[0] + rel_point[0] >= X_len or
                    center[1] + rel_point[1] >= Y_len or
                    center[2] + rel_point[2] >= Z_len):
                continue
            if mask[center[0] + rel_point[0],
                    center[1] + rel_point[1],
                    center[2] + rel_point[2]]:
                neighbors[cnt_center, cnt] = (
                    voxel2idx)[center[0] + rel_point[0],
                               center[1] + rel_point[1],
                               center[2] + rel_point[2]]
                cnt += 1

        if cnt >= thresh_int:
            good_centers[cnt_good_center, :] = centers[cnt_center, :]
            good_neighbors[cnt_good_center, :] = neighbors[cnt_center, :]
            voxel2idx_centers[centers[cnt_center, 0],
                              centers[cnt_center, 1],
                              centers[cnt_center, 2]] = cnt_good_center
            cnt_good_center += 1


    good_centers = good_centers[:cnt_good_center]
    good_neighbors = good_neighbors[:cnt_good_center, :]
    return good_centers, good_neighbors, voxel2idx_centers


@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def do_int_downsample(img, downsample, mask, nan_val=NAN_VAL,
                      edge_drop=False, fourD_mask=False):
    img_smaller = np.zeros((img.shape[0] // downsample,
                            img.shape[1] // downsample,
                            img.shape[2] // downsample,
                            img.shape[3]))
    X_len_ = img_smaller.shape[0]
    Y_len_ = img_smaller.shape[1]
    Z_len_ = img_smaller.shape[2]
    n_samples = img.shape[3]
    size = downsample ** 3
    for x in range(X_len_):
        x_orig = x * downsample
        for y in range(Y_len_):
            y_orig = y * downsample
            for z in range(Z_len_):
                z_orig = z * downsample
                # if not fourD_mask:
                #     if not mask[x_orig, y_orig, z_orig]:
                #         img_smaller[x, y, z] = nan_val
                #         continue
                for n in range(n_samples):
                    vec = np.zeros(size)
                    cnt2 = 0
                    has_nan = False
                    for dii in range(x_orig, x_orig + downsample):
                        for djj in range(y_orig, y_orig + downsample):
                            for dkk in range(z_orig, z_orig + downsample):
                                if fourD_mask:
                                    if mask[dii, djj, dkk, n]:
                                        vec[cnt2] = img[dii, djj, dkk, n]
                                        cnt2 += 1
                                    else:
                                        has_nan = True
                                else:
                                    if mask[dii, djj, dkk]:
                                        vec[cnt2] = img[dii, djj, dkk, n]
                                        cnt2 += 1
                                    else:
                                        has_nan = True
                    if edge_drop:
                        if has_nan:
                            img_smaller[x, y, z, n] = nan_val
                        else:
                            img_smaller[x, y, z, n] = np.sum(vec)
                    else:
                        if cnt2 == 0:
                            img_smaller[x, y, z, n] = nan_val
                        else:
                            # TODO: maybe double check that this vec is proper
                            #  and not shrinking edge voxels
                            img_smaller[x, y, z, n] = np.sum(vec)# / cnt2

    return img_smaller

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def do_int_upsample(img, upsample, mask):
    img_bigger = np.zeros((img.shape[0] * upsample,
                           img.shape[1] * upsample,
                           img.shape[2] * upsample,
                           ))
    X_len_ = img_bigger.shape[0]
    Y_len_ = img_bigger.shape[1]
    Z_len_ = img_bigger.shape[2]
    for x in range(X_len_):
        x_orig = x // upsample
        for y in range(Y_len_):
            y_orig = y // upsample
            for z in range(Z_len_):
                z_orig = z // upsample
                img_bigger[x, y, z] = img[x_orig, y_orig, z_orig]
    return img_bigger

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def get_closest_dists(centers1, centers2, mult1):
    idx1to2 = np.full(len(centers1), NAN_VAL, dtype=np.int32)
    for i, center1 in enumerate(centers1):
        dists = np.empty(len(centers2))
        center1 *= mult1
        for j, center2 in enumerate(centers2):
            # 10x speedup relative to np.linalg.norm(center2 - center1)
            dists[j] = ((center2[0] - center1[0]) ** 2 +
                        (center2[1] - center1[1]) ** 2 +
                        (center2[2] - center1[2]) ** 2)
        idx1to2[i] = np.argmin(dists)
    return idx1to2

@jit(parallel=True, fastmath=True, nopython=True, cache=CACHE_NUMBA)
def get_closest_dists_all(centers1, centers2, l):
    num_centers = centers1.shape[0]
    # num_mods =
    ar = np.empty((num_centers, l.shape[0] ** 3, ), dtype=np.int32)
    mod_combos = np.empty((l.shape[0] ** 3, 3), dtype=np.int32)
    cnt = 0
    for x_mod in l:
        for y_mod in l:
            for z_mod in l:
                mod_combos[cnt, :] = [x_mod, y_mod, z_mod]
                cnt += 1

    for i in prange(num_centers):
        for j, mod_combo in enumerate(mod_combos):
            x_ = centers1[i, 0] + mod_combo[0]
            y_ = centers1[i, 1] + mod_combo[1]
            z_ = centers1[i, 2] + mod_combo[2]
            min_dist = np.inf
            for k, center2 in enumerate(centers2):
                dist = ((center2[0] - x_) ** 2 +
                        (center2[1] - y_) ** 2 +
                        (center2[2] - z_) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    ar[i, j] = k
    return ar

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def get_closest_dists_vox2c(centers1, l, vox2center, search_range,
                            no_y_dim):
    num_centers = centers1.shape[0]
    # num_mods =
    if no_y_dim:
        y_l = np.zeros(1, dtype=np.int32)
        ar = np.empty((num_centers, l.shape[0] ** 2, ), dtype=np.int32)
        mod_combos = np.empty((l.shape[0] ** 2, 3), dtype=np.int32)
    else:
        y_l = l
        ar = np.empty((num_centers, l.shape[0] ** 3, ), dtype=np.int32)
        mod_combos = np.empty((l.shape[0] ** 3, 3), dtype=np.int32)
    cnt = 0
    for x_mod in l:
        for y_mod in y_l:
            for z_mod in l:
                mod_combos[cnt, :] = [x_mod, y_mod, z_mod]
                cnt += 1

    good_finds = np.full(num_centers, 0, dtype=np.int32)
    # good_finds = np.full(num_centers, 1, dtype=np.int32)

    for i in range(num_centers):
        for j, mod_combo in enumerate(mod_combos):
            x_ = centers1[i, 0] + mod_combo[0]
            y_ = centers1[i, 1] + mod_combo[1]
            z_ = centers1[i, 2] + mod_combo[2]

            bullseye = vox2center[x_, y_, z_]
            if bullseye != NAN_VAL:
                ar[i, j] = bullseye
                good_finds[i] += 1
                # print('BULLSEYE')
                continue
            # loop around the bullseye
            found = False
            for k in range(1, search_range):
                for x in range(-k, k+1):
                    for y in range(-k, k+1):
                        for z in range(-k, k+1):
                            if x == -k or x == k or y == -k or y == k or z == -k or z == k:
                                x_ = centers1[i, 0] + mod_combo[0] + x
                                y_ = centers1[i, 1] + mod_combo[1] + y
                                z_ = centers1[i, 2] + mod_combo[2] + z
                                if x_ < 0 or y_ < 0 or z_ < 0:
                                    continue
                                if (x_ >= vox2center.shape[0] or
                                        y_ >= vox2center.shape[1] or
                                        z_ >= vox2center.shape[2]):
                                    continue
                                if vox2center[x_, y_, z_] != NAN_VAL:
                                    ar[i, j] = vox2center[x_, y_, z_]
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                good_finds[i] += 1
                # print('FOUND WITH SEARCH')
                continue
            else:
                # good_finds[i] = 0
                ar[i, j] = NAN_VAL
                # print('?????????')
            # extremely extremely rare to not be found by now...
            # else:

            # if not found:
            #     ar[i, j] = NAN_VAL
            #     print('STILL NOT FOUND????')
            # else:
            #     print('FOUND WITH SEARCH')

            # print(vox2center[x_, y_, z_])

            # print(vox2center[x_-1:x_+2, y_-1:y_+2, z_-1:z_+2])
            #
            # min_dist = np.inf
            # for k, center2 in enumerate(centers2):
            #     dist = ((center2[0] - x_) ** 2 +
            #             (center2[1] - y_) ** 2 +
            #             (center2[2] - z_) ** 2)
            #     if dist < min_dist:
            #         min_dist = dist
            #         ar[i, j] = k
            # print(f'{ar[i, j]}')
            # print('-')
    return ar, good_finds



@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def convert_back_to_img(eval_results, mask, centers):
    img = np.full(mask.shape, NAN_VAL, dtype=np.float32)
    for i in range(centers.shape[0]):
        center = centers[i]
        img[center[0], center[1], center[2]] = eval_results[i]
    return img

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def idx_and_mean(fMRI_RDMs, ar):
    regressors = np.empty((ar.shape[0], 1, fMRI_RDMs.shape[1]),
                          dtype=np.float32)
    for i in range(ar.shape[0]):
        for j in range(fMRI_RDMs.shape[1]):
            s = 0
            for k in range(ar.shape[1]):
                s += fMRI_RDMs[ar[i, k], j]
            regressors[i, 0, j] = s / ar.shape[1]
    return regressors


# if __name__ == '__main__':
#     a = [[2, 1, 2], [100, 10, 100], [-1000, -100, -10]]
#     # [3, 1, 2]
#
#     b = [[100, 10, 100], [-1000, -100, -10], [1, 2, 1]]
#     b_vals = np.array([1, 2, 3])
#     idx1to2 = get_closest_dists(np.array(a),  np.array(b),  1)
#     print(idx1to2)
#     print(b_vals[idx1to2])
#
