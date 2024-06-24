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

    centers = centers[:cnt]

    # Define sphere as points relative to a center (e.g., [-1, 0, 2])
    radius_sq = radius * radius
    rel_points = np.full(((radius * 2) ** 3, 3), NAN_VAL, dtype=np.int32)
    cnt = 0
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                if x*x + y*y + z*z <= radius_sq:
                    rel_points[cnt, :] = [x, y, z]
                    cnt += 1
    rel_points = rel_points[:cnt]
    max_cnt = rel_points.shape[0]

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
            cnt_good_center += 1
    good_centers = good_centers[:cnt_good_center]
    good_neighbors = good_neighbors[:cnt_good_center, :]
    return good_centers, good_neighbors


@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def do_int_downsample(img, downsample, mask):
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
                for n in range(n_samples):
                    vec = np.zeros(size)
                    cnt2 = 0
                    for dii in range(x_orig, x_orig + downsample):
                        for djj in range(y_orig, y_orig + downsample):
                            for dkk in range(z_orig, z_orig + downsample):
                                if mask[dii, djj, dkk]:
                                    vec[cnt2] = img[dii, djj, dkk, n]
                                    cnt2 += 1
                    if cnt2 == 0:
                        img_smaller[x, y, z, n] = NAN_VAL
                    else:
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
    idx0to2 = np.full(len(centers1), NAN_VAL, dtype=np.int32)
    for i, center1 in enumerate(centers1):
        dists = np.empty(len(centers2))
        center1 *= mult1
        for j, center2 in enumerate(centers2):
            dists[j] = np.linalg.norm(center2 - center1)
        idx0to2[i] = np.argmin(dists)
    return idx0to2

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def convert_back_to_img(eval_results, mask, centers):
    img = np.full(mask.shape, NAN_VAL, dtype=np.float32)
    for i in range(centers.shape[0]):
        center = centers[i]
        img[center[0], center[1], center[2]] = eval_results[i]
    return img
