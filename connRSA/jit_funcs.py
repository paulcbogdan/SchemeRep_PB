from time import time
from functools import cache
from pathlib import Path

from numba import jit, njit, prange, set_num_threads, config
import numba as nb
from scipy.interpolate import RegularGridInterpolator

import utils
from atlas_utils import get_atlas
from connRSA.searchlight import do_downsample
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np
import matplotlib.pyplot as plt

from stim import get_stim_RDM
from nilearn import plotting, image
import scipy.stats as stats


config.CACHE_DIR = r'E:\PycharmProjects_E\SchemeRep\cache\numba_test'

NAN_VAL = 10001

CACHE_NUMBA = True
# sig = nb.float64[:](nb.float64[:, :], nb.float64[:])
@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def evaluate_models_searchlight(fMRI_RDMs, stim_RDM):
    num_centers = fMRI_RDMs.shape[0]
    rs = np.empty(num_centers)
    for i in range(num_centers):
        r = np.corrcoef(fMRI_RDMs[i], stim_RDM)[0, 1]
        rs[i] = r
    return rs
# 
# sig = nb.float64[:](nb.float64[:, :], nb.float64[:])
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



# sig = nb.float64[:, :](nb.float64[:, :], nb.int32[:, :])
@njit(parallel=True, fastmath=True, nopython=True, cache=CACHE_NUMBA)
def jit_searchlight_RDMs(data_2d, neighbors):
    num_trials = data_2d.shape[0]
    num_centers = neighbors.shape[0]
    neighbor_len = neighbors.shape[1]
    RDM_size = (num_trials - 1) * num_trials // 2
    out = np.full((num_centers, RDM_size), np.nan, dtype=np.float64)

    for i in prange(num_centers):
        for k in range(neighbor_len):
            if neighbors[i, k] == NAN_VAL:
                cutoff = k# + 1
                break
        else:
            cutoff = neighbor_len
        neighbors_i = neighbors[i, :cutoff]
        sphere_data = np.empty((num_trials, cutoff))
        for j in range(num_trials): # TODO: change data 2d to have nans
            # print(f'{j=}')
            # print(f'{neighbors_i=}')
            # if np.max(neighbors_i) > 1e7:
            #     print(f'{neighbors_i=}')
            #     quit()
            # if np.min(neighbors_i) < -2:
            #     print(f'{neighbors_i=}')
            #     quit()
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
def jit_volume_searchlight(mask, radius=2, threshold=0.5):
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
    # max_neighbors = np.max(voxel2idx)
    # print(max_neighbors)

    centers = centers[:cnt]

    # Define sphere as points relative to a center (e.g., [-1, 0, 2])
    radius_sq = radius * radius
    rel_points = np.full(((radius * 2) ** 3, 3), NAN_VAL, dtype=np.int32)
    cnt = 0
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                # if x == 0 and y == 0 and z == 0:
                #     continue
                # if (x < 0 or y < 0 or z < 0 or
                #         x >= X_len or y >= Y_len or z >= Z_len):
                #     continue
                if x*x + y*y + z*z <= radius_sq:
                    rel_points[cnt, :] = [x, y, z]
                    cnt += 1
    rel_points = rel_points[:cnt]
    max_cnt = rel_points.shape[0]

    # if threshold == 1:
    #     neighbors = np.full((centers.shape[0], max_cnt), -1, dtype=np.int32)
    #     for cnt_center, center in enumerate(centers):
    #         cnt = 0
    #         for rel_point in rel_points:
    #             if mask[center[0] + rel_point[0],
    #                     center[1] + rel_point[1],
    #                     center[2] + rel_point[2]]:
    #                 neighbors[cnt_center, cnt] = (
    #                     voxel2idx)[center[0] + rel_point[0],
    #                                center[1] + rel_point[1],
    #                                center[2] + rel_point[2]]
    #                 cnt += 1
    #     return centers, neighbors

    # Below messes up
    neighbors = np.full((centers.shape[0], max_cnt), NAN_VAL, dtype=np.int32)
    good_neighbors = np.full((centers.shape[0], max_cnt), NAN_VAL, dtype=np.int32)
    good_centers = np.full(centers.shape, NAN_VAL, dtype=np.int32)
    thresh_int = int(threshold * max_cnt)

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
            # print(rel_point)
            if mask[center[0] + rel_point[0],
                    center[1] + rel_point[1],
                    center[2] + rel_point[2]]:
                neighbors[cnt_center, cnt] = (
                    voxel2idx)[center[0] + rel_point[0],
                               center[1] + rel_point[1],
                               center[2] + rel_point[2]]
                # if neighbors[cnt_center, cnt] > 1e7:
                #     print(neighbors[cnt_center, cnt])
                cnt += 1

        if cnt >= thresh_int:
            good_centers[cnt_good_center, :] = centers[cnt_center, :]
            good_neighbors[cnt_good_center, :] = neighbors[cnt_center, :]
            cnt_good_center += 1
    good_centers = good_centers[:cnt_good_center]
    good_neighbors = good_neighbors[:cnt_good_center, :]
    return good_centers, good_neighbors
