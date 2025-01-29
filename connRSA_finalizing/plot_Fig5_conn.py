from collections import defaultdict
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit, prange
from scipy import stats
from tqdm import tqdm

from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
# from connRSA.conn_regress import get_ERS_scores
from connRSA.single_trial_conn import prep_fps
from fMRI_proc import within_run_to_nan, get_IRAFs
from Study1A.load_Study1A_funcs import load_FC
from networks.old.networks import prep_networks

from Utils.plotting_funcs import plot_connectivity
from organize_bhv import get_trial_info, sort_df_sn
from stim import get_semantic_vectors, get_DNN_vecs
from utils import stdize
from Utils.pickle_wrap_funcs import pickle_wrap
from functools import cache
import seaborn as sns

from numba import njit

# suppress RuntimeWarning: All-NaN slice
from warnings import filterwarnings
from nilearn import image, plotting
from nilearn import plotting

filterwarnings("ignore", category=RuntimeWarning,
               message="All-NaN slice encountered")


import os
# os.chdir(r'/')


@cache
def get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run, second_order,
                flat=True, within_nan=True, obj_sort=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    try:
        with open(fp_focus1, 'rb') as f:
            RSM = np.load(f, )
    except FileNotFoundError:
        # print('Not found!')
        RSM = np.full((114, 114), np.nan)
        print(f'Not found! {sn}/{fp}/{ROI}')
    except ValueError as e:
        print(f'{fp_focus1=}')
        print(f'allow_pickl=False ({sn}, {ROI}, {fp}): {e=}')
        quit()

    if within_nan:
        RSM = within_run_to_nan(RSM)
    # print(RSM.shape)
    # plt.imshow(RSM)
    # plt.show()
    # quit()

    df_sn = get_trial_info(sn, verbose=-1)
    df_sn, sess = sort_df_sn(df_sn, fp)

    if obj_sort:
        obj_idx = df_sn['obj'].argsort()
        RSM = RSM[np.ix_(obj_idx, obj_idx)]

    if flat:
        trils = np.tril_indices(RSM.shape[0], k=-1)
        return RSM[trils]
    else:
        return RSM

def get_IC_mat(sn, ROIs, fp, trial_similarity, stdize_by_run, second_order,
               within_nan=True):
    RSMs = []
    for ROI in ROIs:
        RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                          second_order, within_nan=within_nan)
        RSMs.append(RSM)
    RSMs = np.array(RSMs)
    nan_cols = np.all(np.isnan(RSMs), axis=0)
    RSMs = RSMs[:, ~nan_cols]
    corr = np.ma.corrcoef(RSMs)
    corr[np.diag_indices_from(corr)] = np.nan
    return corr

def load_fp2RSMs(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, within_nan=True, obj_sort=True):
    fp2RSMs = {}
    for fp in fps:
        RSMs = []
        for ROI in ROIs:
            RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                              second_order, within_nan=within_nan,
                              obj_sort=obj_sort)
            RSMs.append(RSM)
        RSMs = np.array(RSMs)
        fp2RSMs[fp] = RSMs
    return fp2RSMs

def get_cross_IC_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, within_nan=True, obj_sort=True,
                     same_RSM_corr=True):
    kw = {'ROIs': ROIs, 'trial_similarity': trial_similarity,
          'stdize_by_run': stdize_by_run, 'second_order': second_order,
          'within_nan': within_nan, 'sn': sn, 'fps': fps,
          'obj_sort': obj_sort}
    fp2RSMs = pickle_wrap(load_fp2RSMs, None, kwargs=kw,
                          easy_override=True, verbose=-1)

    t_st = time()
    l = []
    for fp0, RSMs0 in fp2RSMs.items():
        for fp1, RSMs1 in fp2RSMs.items():
            if fp0 == fp1:
                continue
            pair = [RSMs0, RSMs1]
            l.append(pair)
    l = np.array(l)
    # print(f'{same_RSM_corr=}')
    if same_RSM_corr:
        # Below code would make it so every RSM pair is calculated on the exact same
        # print(f'{l.shape=}')
        bad_cols = np.all(np.isnan(l), axis=(2)) # RSM cell is NaN for all ROI
        bad_cols = np.any(bad_cols, axis=(0, 1)) # any ROI has NaNs. Not great.
    else:
        bad_cols = np.all(np.isnan(l), axis=(0, 1, 2)) # RSM cell is NaN for all
        # Barely does anything if obj_sort = True. Just drops like 200
        #   drops like 2k if obj_sort = False
    print(f'{same_RSM_corr}, {np.sum(bad_cols)=}')

    l = l[:, :, :, ~bad_cols]
    l = stdize(l, axis=-1, nans=obj_sort)

    nan_ar = np.any(np.isnan(l), axis=(1))  # RSM cell is NaN for all ROI
    nan_ar = np.all(nan_ar, axis=1)

    corrs = numba_fp_x_fp_RSMs(l, nans=obj_sort, nan_ar=nan_ar)
    # print(corrs)
    t_end = time()
    print(f'Numba corr calc: {t_end - t_st:.3f}')
    return corrs

@cache
def get_feat_RSMs(sn, fp, semantic=False):
    if semantic == 'random':
        out = np.random.normal(size=(10_000, 114, 114))
        return out
    df_sn = get_trial_info(sn)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').
            replace('4', '').replace('7', '').replace('8', ''))
    df_sn['sess'] = sess
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=0, PCA=True)

    num_feats = d_vecs[next(iter(d_vecs.keys()))]
    # print(len(d_vecs))
    # quit()
    out = np.full((len(num_feats), 114, 114), np.nan)
    for k in range(len(num_feats)):
        for i, obj0 in enumerate(df_sn['obj']):
            for j, obj1 in enumerate(df_sn['obj']):
                if i >= j:
                    continue
                dif = np.abs(d_vecs[obj0][k] - d_vecs[obj1][k])
                out[k, i, j] = dif
                out[k, j, i] = dif
    return out



def get_RSA_feat_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, RDM_method, semantic,
                     just_get_IRAFs=False,
                     odd_even=True):
    dir_root = fr'cache/conn_RSA/ars/RSA'
    # num_pairs = (len(fps) * (len(fps) - 1)) // 2

    sn_corrs = []
    for j, fp0 in enumerate(fps):
        ROI_RSMs = []
        skip_idx_ROIs = []
        # stim_RSMs = []
        for i, ROI in enumerate(ROIs):
            # feat_RSMs = get_feat_RSMs(sn, fp0, semantic=semantic)
            dir0_focus = (f'{dir_root}/{fp0}_{trial_similarity}_'
                          f'{second_order}_{RDM_method}_{stdize_by_run}')
            fp0_focus = f'{dir0_focus}/{sn}_{ROI}_BOLD.npy'
            try:
                with open(fp0_focus, 'rb') as f:
                    ROI_RSM = np.load(f)
                skip_idx_ROIs.append(False)
                # if RDM_method == 'within_nan':
                #     RSM0_focus = within_run_to_nan(RSM0_focus)
            except FileNotFoundError:
                ROI_RSM = np.zeros((114, 114))
                skip_idx_ROIs.append(True)
                print(f'Missing: {ROI}, {fp0=}')
                # ar[i, j, :] = np.full(114, np.nan)
                # continue
            ROI_RSMs.append(ROI_RSM)

        ROI_RSMs = np.array(ROI_RSMs)
        stim_RSMs = get_feat_RSMs(sn, fp0, semantic=semantic)
        skip_idx_ROIs = np.array(skip_idx_ROIs)
        if odd_even:
            ROI_RSMs_odd = ROI_RSMs[:, ::2, ::2]
            ROI_RSMs_even = ROI_RSMs[:, 1::2, 1::2]
            stim_RSMs_odd = stim_RSMs[:, ::2, ::2]
            stim_RSMs_even = stim_RSMs[:, 1::2, 1::2]
            from scipy import spatial
            t_st = time()
            ROI_feat_rs_odd = get_ROI_RSMs_x_stim_RSMs(ROI_RSMs_odd, stim_RSMs_odd,
                                                       skip_idx_ROIs)
            ROI_feat_rs_even = get_ROI_RSMs_x_stim_RSMs(ROI_RSMs_even, stim_RSMs_even,
                                                        skip_idx_ROIs)
            t_end = time()
            print(f'Numba feat x stim calc: {t_end - t_st:.3f} s')

            corr = spatial.distance.cdist(ROI_feat_rs_odd, ROI_feat_rs_even, 'correlation')
            # print(corr)
            # quit()
            # flip corr over diagonal
            # corr = np.zeros((5, 5))
            # corr[2, 4] = 1
            # corr_flip = np.flip(corr, axis=1)
            # plt.imshow(corr)
            # plt.show()
            # plt.imshow(corr_flip)
            # plt.show()

            corr_t = np.transpose(corr)
            corr = (corr + corr_t) / 2
            # plt.imshow(corr90)
            # plt.show()
            #
            # print(corr.shape)
            # quit()

            # corr = np.corrcoef(ROI_feat_rs)
            sn_corrs.append(corr)
        else:
            t_st = time()
            ROI_feat_rs = get_ROI_RSMs_x_stim_RSMs(ROI_RSMs, stim_RSMs, skip_idx_ROIs)
            t_end = time()
            print(f'Numba feat x stim calc: {t_end - t_st:.3f} s')
            corr = np.corrcoef(ROI_feat_rs)
            sn_corrs.append(corr)
    sn_corrs = np.array(sn_corrs)
    # plt.imshow(sn_corrs[0])
    # plt.show()
    # quit()
    return sn_corrs

# num_trials = 57
# idx_to_run = np.empty(num_trials)
# for t in range(num_trials):
#     idx_to_run[t] = t // (num_trials // 3)
#
# idx = 0
# for t0 in range(num_trials):
#     for t1 in range(t0):
#         if idx_to_run[t0] == idx_to_run[t1]:
#             continue
#         # print(ROI_RSMs[ROI_i, t0, t1])
#         idx += 1
# print(idx)
# quit()
@njit(fastmath=True, nopython=True, cache=True)
def get_std_rsm_flat(ROI_RSMs, skip_idx):
    # print(ROI_RSMs.shape)
    # quit()
    num_trials = ROI_RSMs.shape[2]
    idx_to_run = np.empty(num_trials)
    for t in range(num_trials):
        idx_to_run[t] = t // (num_trials // 3)
    num_ROIs = ROI_RSMs.shape[0]
    # print(ROI_RSMs)
    # quit()
    if num_trials == 114:
        ROI_RSMs_flat = np.empty((num_ROIs, 4332), dtype=np.float32)
    else:
        ROI_RSMs_flat = np.empty((num_ROIs, 1083), dtype=np.float32)
    for ROI_i in range(num_ROIs):
        if skip_idx[ROI_i]:
            if num_trials == 114:
                ROI_RSMs_flat[ROI_i, :] = np.zeros(4332)
            else:
                ROI_RSMs_flat[ROI_i, :] = np.zeros(1083)
            continue
        # zscore
        if num_trials == 114:
            v = np.empty(4332)
        else:
            v = np.empty(1083)
        idx = 0
        for t0 in range(num_trials):
            for t1 in range(t0):
                if idx_to_run[t0] == idx_to_run[t1]:
                    continue
                v[idx] = ROI_RSMs[ROI_i, t0, t1]
                # print(ROI_RSMs[ROI_i, t0, t1])
                idx += 1
        # TODO: Chck if any nan

        M = np.mean(v)
        # print(M)
        # quit()
        sd = np.std(v)
        # print(sd)
        # print(v)
        # quit()
        ROI_RSMs_flat[ROI_i, :] = (v - M) / sd
    return ROI_RSMs_flat

@njit(fastmath=True, nopython=True, cache=True)
def get_ROI_RSMs_x_stim_RSMs(ROI_RSMs, stim_RSMs, skip_idx_ROIs):
    num_ROIs = ROI_RSMs.shape[0]
    num_stim = stim_RSMs.shape[0]
    out = np.empty((num_ROIs, num_stim))

    stim_RSMs_flat = get_std_rsm_flat(stim_RSMs, np.zeros(num_stim, dtype=np.bool_))
    ROI_RSMs_flat = get_std_rsm_flat(ROI_RSMs, skip_idx_ROIs)

    for ROI_i in range(num_ROIs):
        for stim_j in range(num_stim):
            ROI_v = ROI_RSMs_flat[ROI_i, :]
            stim_v = stim_RSMs_flat[stim_j, :]
            out[ROI_i, stim_j] = np.mean(ROI_v * stim_v)
    return out

def get_cross_IRAF_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                       second_order, RDM_method, semantic,
                       just_get_IRAFs=False):
    dir_root = fr'cache/conn_RSA/ars/RSA'
    # ROI2fps2iERS = {}
    num_pairs = (len(fps) * (len(fps) - 1)) // 2
    ROI2fps2iERS = np.full((num_pairs, len(ROIs), 114), np.nan)

    M_ERS_difs = []

    fp2stim_RSM = {}

    for fp0 in fps:
        dir0_focus = (f'{dir_root}/{fp0}_{trial_similarity}_'
                      f'{second_order}_{RDM_method}_{stdize_by_run}')
        fp_stim = f'{dir0_focus}/{sn}_stim_{semantic}.npy'
        with open(fp_stim, 'rb') as f:
            RSM_stim = np.load(f)
        if RDM_method == 'within_nan':
            RSM_stim = within_run_to_nan(RSM_stim)
        fp2stim_RSM[fp0] = RSM_stim

    df_sn = get_trial_info(sn)

    ROI2fp2IRAFs = defaultdict(dict)

    # ar = np.empty((246, 4, 114))
    ar = np.full((246, len(fps), 114), np.nan)

    for i, ROI in enumerate(ROIs):
        # ROI2fps2iERS[ROI] = {}
        # l_ERS_dif_M = []
        cnt = 0
        for j, fp0 in enumerate(fps):
            dir0_focus = (f'{dir_root}/{fp0}_{trial_similarity}_'
                          f'{second_order}_{RDM_method}_{stdize_by_run}')
            fp0_focus = f'{dir0_focus}/{sn}_{ROI}_BOLD.npy'
            try:
                with open(fp0_focus, 'rb') as f:
                    RSM0_focus = np.load(f)
                if RDM_method == 'within_nan':
                    RSM0_focus = within_run_to_nan(RSM0_focus)
            except FileNotFoundError:
                # print(f'Missing: {ROI}, {fp0=}')
                ar[i, j, :] = np.full(114, np.nan)
                continue

            RSM_stim = fp2stim_RSM[fp0]
                # get_IRAFs sorts by df_sn['obj']
            IRAFs = get_IRAFs(RSM0_focus, RSM_stim, df_sn, within_to_nan=True,
                              by_run=False, second_order='spear')
            ROI2fp2IRAFs[ROI][fp0] = IRAFs
            ar[i, j, :] = IRAFs

    if get_IRAFs:
        ar = np.transpose(ar, (1, 0, 2))
        return ar

    ar = stdize(ar, axis=-1)
    t_st = time()
    mats = numba_IRAF_x_IRAF(ar)
    t_end = time()
    print(f'Numba IRAF x IRAF calc: {t_end - t_st:.3f} s')
    return mats

# @jit(nopython=True, parallel=True, fastmath=True)
def numba_IRAF_x_IRAF(ar):
    # ar.shape = (246, 4, 114)
    num_ROIs = ar.shape[0]
    num_fps = ar.shape[1]
    # num_trials = ar.shape[2]
    num_prods = num_fps * (num_fps - 1) # // 2

    if num_fps == 4:
        lookup = {(1, 0): 0, (2, 0): 1, (2, 1): 2,
                  (3, 0): 3, (3, 1): 4, (3, 2): 5,

                  (0, 1): 6, (0, 2): 7, (1, 2): 8,
                  (0, 3): 9, (1, 3): 10, (2, 3): 11
                  }
    elif num_fps == 3:
        lookup = {(1, 0): 0, (2, 0): 1, (2, 1): 2,
                  (0, 1): 3, (0, 2): 4, (1, 2): 5,}
    else:
         raise ValueError

    assert num_fps == 4 or num_fps == 3
    # ar_out = np.full((num_prods, num_ROIs, num_ROIs), np.nan)
    ar_out = np.empty((num_prods, num_ROIs, num_ROIs)
                      ) # 0.2s faster than np.full(..., nan). i live for danger.

    for ROI_i in prange(num_ROIs):
        for ROI_j in range(num_ROIs):
            for fp0 in range(num_fps):
                IRAFs0 = ar[ROI_i, fp0, :]
                for fp1 in range(fp0):
                    IRAFs1 = ar[ROI_j, fp1, :]
                    r = np.mean(IRAFs0 * IRAFs1)
                    ar_out[lookup[(fp0, fp1)], ROI_i, ROI_j,] = r
                    ar_out[lookup[(fp1, fp0)], ROI_j, ROI_i,] = r
                    # ar_out[lookup[(fp0, fp1)], ROI_j, ROI_i, ] = r
    return ar_out


def get_cross_ERS_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                      cross=False, nan_block=True, do_same=False,
                      get_var=False,
                      ):
    dir_root = fr'cache/conn_RSA/ars/ERS'
    # ROI2fps2iERS = {}
    num_pairs = (len(fps) * (len(fps) - 1)) // 2
    fp2ROI2iNPS = np.full((num_pairs, len(ROIs), 114), np.nan)


    tup2cnt = {}
    fp12block2objs = {}

    M_ERS_difs = []

    for i, ROI in enumerate(ROIs):
        # ROI2fps2iERS[ROI] = {}
        # l_ERS_dif_M = []
        cnt = 0
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                stdize_by_run_ = False
                dir_focus = (fr'{dir_root}/{fp0}_{fp1}_{trial_similarity}_'
                             fr'{stdize_by_run_}')
                fp = f'{dir_focus}/{sn}_{ROI}_BOLD.npy'

                if nan_block:
                    if fp1 not in fp12block2objs:
                        df_sn = get_trial_info(sn, verbose=-1)
                        df_sn.sort_values(by='obj', inplace=True)
                        sess1 = (fp1.split('_')[0].replace('2', '').
                                 replace('3', '').replace('4', '').
                                 replace('7', '').replace('8', ''))
                        df_sn[f'{sess1}_trial'] = (
                            stats.rankdata(df_sn[f'{sess1}_trial']))
                        df_sn[f'{sess1}_trial'] = (df_sn[f'{sess1}_trial'].
                                                   astype(int))
                        obj2block_l = []
                        block2objs = defaultdict(list)
                        for j in range(len(df_sn)):
                            row = df_sn.iloc[j]
                            block = (row[f'{sess1}_trial'] - 1) // 38
                            obj2block_l.append(block)
                            block2objs[block].append(j)
                        block2objs[0] = set(block2objs[0])
                        block2objs[1] = set(block2objs[1])
                        block2objs[2] = set(block2objs[2])
                        fp12block2objs[fp1] = block2objs
                try:
                    with open(fp, 'rb') as f:
                        ERS = np.load(f)

                    if nan_block:
                        for j in range(len(obj2block_l)):
                            for k in range(114):
                                if (k in block2objs[obj2block_l[j]]):
                                    continue

                                ERS[k, j] = np.nan

                    ERS_dif, _ = get_ERS_scores(ERS, get_same=do_same)

                except EOFError:
                    ERS_dif = np.full(114, np.nan)

                except FileNotFoundError:
                    ERS_dif = np.full(114, np.nan)

                fp2ROI2iNPS[cnt, i, :] = ERS_dif
                tup = tuple(sorted((fp0, fp1)))
                tup2cnt[tup] = cnt
                cnt += 1
        if np.any(np.isnan(fp2ROI2iNPS[:, i, :])):
            M_ERS_difs.append(np.nanmean(fp2ROI2iNPS[:, i, :]))
        else:
            M_ERS_difs.append(np.mean(fp2ROI2iNPS[:, i, :]))

    if cross:
        valid_pairs = set()
        for fp0 in fps:
            for fp1 in fps:
                if fp0 == fp1: continue
                tup01 = (fp0, fp1)
                tup01 = tuple(sorted(tup01))
                for fp2 in fps:
                    for fp3 in fps:
                        if fp2 == fp3: continue
                        tup23 = (fp2, fp3)
                        tup23 = tuple(sorted(tup23))
                        if set(tup01).intersection(tup23):
                            continue
                        valid_pairs.add((tup01, tup23))
        valid_pairs = sorted(list(valid_pairs))
        # print(valid_pairs)
        # quit()

        pair2fp2ROI2iNPS = np.full((len(valid_pairs), 2, len(ROIs), 114), np.nan)
        for i, (tup01, tup23) in enumerate(valid_pairs):
            cnt = tup2cnt[tup01]
            cnt2 = tup2cnt[tup23]
            pair2fp2ROI2iNPS[i, 0, :, :] = fp2ROI2iNPS[cnt, :, :]
            pair2fp2ROI2iNPS[i, 1, :, :] = fp2ROI2iNPS[cnt2, :, :]
        # print(fp2ROI2iNPS)
        # quit()
        if stdize_by_run:
            # print(pair2fp2ROI2iNPS)
            run0 = pair2fp2ROI2iNPS[..., :38]
            run0 = stats.zscore(run0, axis=-1, nan_policy='omit')
            run1 = pair2fp2ROI2iNPS[..., 38:76]
            run1 = stats.zscore(run1, axis=-1, nan_policy='omit')
            run2 = pair2fp2ROI2iNPS[..., 76:]
            run2 = stats.zscore(run2, axis=-1, nan_policy='omit')
            pair2fp2ROI2iNPS = np.concatenate([run0, run1, run2], axis=-1)
            # print(pair2fp2ROI2iNPS)
            # quit()
        else:
            pair2fp2ROI2iNPS = stdize(pair2fp2ROI2iNPS, axis=-1)
        corrs = numba_fpfp_x_fpfp_NPS(pair2fp2ROI2iNPS)
        # print(corrs.shape)
        # diag = np.diag(np.nanmean(corrs, axis=0))
        # print(f'{diag=}')
        # for i, corr in enumerate(corrs):
        #     plt.title(f'{i=}')
        #     plt.imshow(corr)
        #     plt.show()
        # plt.imshow(np.nanmean(corrs, axis=0))
        # plt.show()
        # quit()
    else:

        if get_var == 'all':
            return fp2ROI2iNPS
        elif get_var != False:
            idxs = get_idxs(get_var)
            return fp2ROI2iNPS[:, idxs, :]
            # v = np.nanvar(fp2ROI2iNPS[:, idxs, :], axis=1)
            # return v
            # return fp2ROI2iNPS
        fp2ROI2iNPS = stdize(fp2ROI2iNPS, axis=-1)

        # print(fp2ROI2iNPS.shape)

        # print(f'{fp2ROI2iNPS.shape=}')
        # quit()
        corrs = numba_fp_x_fp_ERS(fp2ROI2iNPS)

    return corrs, M_ERS_difs

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fpfp_x_fpfp_NPS(l):
    num_fpfp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fpfp_pairs * 2, num_ROIs, num_ROIs))
    for i in prange(num_fpfp_pairs):
        for j0 in range(num_ROIs):
            ERS0 = l[i, 0, j0, :]
            for j1 in range(num_ROIs):
                out[i, j0, j1] = np.mean(ERS0 * l[i, 1, j1, :])

            ERS0 = l[i, 1, j0, :]
            for j1 in range(num_ROIs):
                out[i + 6, j0, j1] = np.mean(ERS0 * l[i, 0, j1, :])
    return out

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fp_x_fp_ERS(l):
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[1]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in prange(num_fp_pairs):
        for j0 in range(num_ROIs):
            ERS0 = l[i, j0, :]
            for j1 in range(num_ROIs):
                out[i, j0, j1] = np.mean(ERS0 * l[i, j1, :])
    return out

@jit(nopython=False, parallel=True, fastmath=True)
def numba_fp_x_fp_RSMs(l, nans=False, nan_ar=True): # 4 seconds first then 3 seconds vs. 9 w/ numpy below
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in prange(num_fp_pairs): # prange gives an 8x speedup??
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        for j0 in range(num_ROIs):
            for j1 in range(num_ROIs):
                if nans:
                    either_nan = nan_ar[i, :]
                    out[i, j0, j1] = np.mean(RSM0[j0, ~either_nan] *
                                             RSM1[j1, ~either_nan])
                else:
                    out[i, j0, j1] = np.mean(RSM0[j0, :] * RSM1[j1, :])
    return out

def numpy_fp_x_fp_RSMs(l): # if numba doesn't use prange, this is same speed
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in range(num_fp_pairs):
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        corr = np.mean(RSM0[None, :, :] * RSM1[:, None, :], axis=-1)
        out[i] = corr
    return out

# calculate IC between tasks and normalize by an ROIs IC with itself
#   this is needed to rule out that high/low IC is due to data reliablity
#   e.g., Occipital 1 may have higher IC with occipital 2 than IPL 1 has with
#   IPL 2, but this could be due to occipital 1 & 2 having more reliable data

def get_idxs(ROI):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[ROI])[ROI])
    ROIs_match = [i for region in regions
                  for i, ROI in enumerate(get_BNA_ROIs()) if region in ROI]
    return ROIs_match


def plot_network_M(corrs, corrs_FC, sns_FC, fn_out):


    networks = ['Occipital', 'ITL', 'Parietal', 'PFC']
    network2name = {'Occipital': 'Occipital', 'ITL': 'Temporal',
                    'Parietal': 'Parietal', 'PFC': 'PFC'}
    names = [network2name[net] for net in networks]
    net2idxs = {}
    df_as_l = defaultdict(list)

    colors = ['dodgerblue', 'darkorange', 'crimson', 'limegreen']
    # plt.gcf().add_axes([0.1,0.1, 0.35,0.8])

    for net in networks:
        idxs = get_idxs(net)
        net2idxs[net] = idxs
        net_corr = corrs[:, idxs][:, :, idxs]
        net_sn_vals = np.nanmean(net_corr, axis=(1, 2))
        net_M = np.nanmean(net_sn_vals)
        net_SD = np.nanstd(net_sn_vals, ddof=1)
        net_SE = net_SD / np.sqrt(np.sum(~np.isnan(net_sn_vals)))
        df_as_l['net'].extend([network2name[net]]*len(sns_FC))
        df_as_l['sn'].extend(sns_FC)
        df_as_l['val'].extend(net_sn_vals)

    df = pd.DataFrame(df_as_l)

    # cross huge outlier
    # df = df[df['sn'] != '205']

    plt.figure(figsize=(3.5, 4))
    plt.rcParams.update({'font.size': 14})
    bp = sns.stripplot(x='net', y='val', #hue='net', #palette=colors,
                       alpha=0.5, linewidth=.7,
                       data=df, order=names,
                       palette=colors)
    plt.ylabel('Mean correlation', fontsize=14)
    plt.xlabel('')
    # plt.xticks([0, 1, 2, 3], networks, color=colors)
    plt.xticks([0, 1, 2, 3], ['', '', '', ''])
    plt.xlim(-0.5, 3.65)
    plt.tick_params(bottom=False)
    # plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().spines[['top', 'left', 'bottom']].set_visible(False)
    plt.gca().yaxis.set_label_position('right')
    plt.gca().set_ylabel('Mean correlation', rotation=270, labelpad=20)
    plt.gca().yaxis.tick_right()

    colors = ['dodgerblue', 'darkorange', 'crimson', 'green']
    for i, name in enumerate(names):
        # plt.gca().get_xticklabels()[i].set_color(colors[i])
        df_net = df[df['net'] == name]
        # lowest_net = df_net['val'].min()
        highest_net = df_net['val'].max()
        # plt.text(i, lowest_net - 0.025, f'{name}', ha='center',
        #          color=colors[i])
        plt.text(i, highest_net + 0.0175, f'{name}', ha='center',
                 fontsize=14, color=colors[i])

    df_OC = df[df['net'] == 'Occipital']['val'].values
    df_ITL = df[df['net'] == 'Temporal']['val'].values
    t, p = stats.ttest_rel(df_OC, df_ITL)
    # print(df_ITL)
    idx = np.argmax(df_ITL)
    # print(idx)
    # print(df_ITL[idx])
    # quit()
    N = np.sum(~np.isnan(df_OC))
    d = t / np.sqrt(len(df_OC))
    M_OC = np.nanmean(df_OC)
    M_ITL = np.nanmean(df_ITL)
    print(f'{fn_out}: {d=:.3f}, {M_OC=:.3f}, {M_ITL=:.3f}, '
          f't[{N - 1}] = {t:.2f}, {p=:.3f}')


    # plt.text(0)
    # plt.title(title)
    if True:
        pass
    elif 'regressed' in fn_out:
        pass
    else:
        if 'NPS' in fn_out:
            plt.ylim(0.0, 0.345)
            plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3])
        elif 'RSM' in fn_out:
            plt.ylim(0, None)
        else:
            plt.ylim(0, None)
            # plt.ylim(0.02, 0.62)

    plt.yticks(fontsize=14)

    # plt.axis([-0.5, 3.5, 0, 0.25])
    # plt.axis('equal')
    # plt.axis([-0.5, 3.5, 0, 0.25], frameon=False)
    # plt.show()

    # ax2 = plt.gcf().add_axes([0.2, 0.2, 0.1, 0.8])
    # ax2 = plt.gcf().add_axes([0.1, 0.1, 0.01, 0.8])
    # ax2.set_ylabel('Mean correlation')
    # ax2.set_ylim(0.05, 0.35)

    # ax2.spines[['top', 'right', 'bottom']].set_visible(False)
    # ax2.set_ylim(0, 0.25)
    plt.tight_layout()
    fp_fig = fr'connRSA/FC_figs/M_{fn_out}'
    plt.savefig(fp_fig, dpi=300)
    plt.show()

def run_IC_analysis(ERS=True, regress_FC=True, get_M=False,
                    rsa_feat=False):
    # semantic = False
    # regress_FC = False
    # rsa_feat = False

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)

    ERS_nan_block = False
    cross = False
    IRAF = False
    # rsa_feat = True

    semantic = True
    drop_con = False
    same_RSM_corr = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    atlas = get_atlas()#combine_regions=True, combine_bilateral=True)
    ROIs = atlas['ROIs']

    if cross and ERS:
        assert not drop_con

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']
    sns = sns[::-1]

    # if cross:
    #     outlier = ['205']
    #     sns = [sn for sn in sns if sn not in outlier]

    four_tasks = '7'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]
    ERS_scores_all = []

    for i, sn in tqdm(enumerate(sns), desc=f'Looping IC: {cross=}'):


        kwargs['sn'] = sn
        if rsa_feat:
            kwargs['fps'] = fps
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = semantic
            sn_corrs = pickle_wrap(get_RSA_feat_mat,
                                   kwargs=kwargs, verbose=-1,
                                   easy_override=False,
                                   dt_max=dt_max)
            sn_corrs = 1 - sn_corrs
        elif IRAF:
            kwargs['fps'] = fps
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = semantic
            sn_corrs = pickle_wrap(get_cross_IRAF_mat,
                                   kwargs=kwargs, verbose=-1,
                                   easy_override=False,
                                   dt_max=dt_max)
        elif ERS:
            kwargs['fps'] = fps
            kwargs['cross'] = cross
            kwargs['nan_block'] = ERS_nan_block

            sn_corrs, ERS_scores = pickle_wrap(get_cross_ERS_mat,
                                               kwargs=kwargs, verbose=-1,
                                               easy_override=False,
                                               dt_max=dt_max)
            ERS_scores_all.append(ERS_scores)
            # print(sn_corrs)
            # quit()
        elif cross:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            kwargs['fps'] = fps
            kwargs['same_RSM_corr'] = same_RSM_corr
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)


            # print(f'{i}, {sn}')
            # if i == 36:
            #     # big outlier for temporal
            #     sn_corrs = np.full(sn_corrs.shape, np.nan)

        else:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)

        corr = np.nanmean(sn_corrs, axis=0)

        corrs.append(corr)


    corrs = np.array(corrs)
    diag = corrs[:, *np.diag_indices(corrs.shape[1])]
    # diag_M = np.nanmean(diag, axis=0)
    # diag_M[diag_M < 0.001] = np.nan

    # print(diag_M)
    # quit()
    # corrs /= diag_M[..., None]
    # print(corrs.shape)
    # quit()
    # corrs -= diag[..., None]
    # print(diag.shape)
    # quit()
    # print(len(diag[0]))
    # print(len(ROIs))
    # quit()
    if diag[0][0] < .99:
        for i in range(len(diag[0])):
            # if len(diag) == 27:
            # t, p = stats.ttest_1samp(
            #     np.nanmean(diag[:, atlas['tick_lows'][i]:atlas['tick_lows'][i+1]],
            #                axis=-1), 0, nan_policy='omit')
            # M = np.nanmean(diag[:, atlas['tick_lows'][i]:atlas['tick_lows'][i+1]])
            t, p = stats.ttest_1samp(
                np.nanmean(diag[:, i:i+1], axis=-1), 0, nan_policy='omit')
            M = np.nanmean(diag[:, i:i+1])

            ROI_i = ROIs[i]
            print(f'{ROI_i}: {t=:.3f}, {M=:.5f}')
    else:
        corrs[:, *np.diag_indices(corrs.shape[1])] = np.nan
    # # plt.plot(np.nanmean(diag, axis=0))
    # # plt.show()
    # quit()
    ERS_str = 'NPS' if ERS else 'RSM'

    # matrix_to_csv(corrs, f'connRSA_finalizing/{ERS_str}_conn_for_swd.csv')

    if get_M and not regress_FC:
        return corrs

    # M_ventral_FC, M_occ_FC,
    corrs_FC, sns_FC = (
        pickle_wrap(plot_FC_mat, None, easy_override=False, dt_max=dt_max))

    print(corrs_FC.shape)
    matrix_to_csv(corrs_FC, f'connRSA_finalizing/FC_conn_for_swd.csv')
    quit()

    title = 'RSM x RSM connectivity' if not ERS else 'NPS x NPS connectivity'
    if regress_FC:
        title += ' (FC regressed)'
    fn_out = r'NPS' if ERS else 'RSM'
    fn_out += '_regressed' if regress_FC else ''
    fn_out += '.png'

    atlas = get_atlas(lifu_labels=False)

    if len(ROIs) > 54:
        plot_network_M(corrs, corrs_FC, sns_FC, fn_out)

    if regress_FC:
        for sn_i in range(corrs.shape[0]):
            corr = corrs[sn_i]
            corr_flat = corr.flatten()
            corr_FC = corrs_FC[sn_i]
            corr_FC_flat = corr_FC.flatten()
            nans = np.isnan(corr_flat) | np.isnan(corr_FC_flat)
            corr_flat_ = corr_flat[~nans]
            corr_FC_flat_ = corr_FC_flat[~nans]
            slope, intercept, r, p, se = (
                stats.linregress(corr_FC_flat_, y=corr_flat_,
                                 alternative='two-sided'))

            corr_flat -= slope * corr_FC_flat
            corr = corr_flat.reshape(corr.shape)
            corrs[sn_i] = corr

    if get_M:
        return corrs

    M = np.nanmean(corrs, axis=0)
    SE = stats.sem(corrs, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(corrs), axis=0)

    for i in range(N.shape[-1]):
        val = 60 - N[i, 0]
        if i == 0 or val == 0:
            continue

    M[N <= 48] = np.nan

    if len(ROIs) == 27:
        plt.imshow(M, vmin=0)#, vmin=0, vmax=1)
        plt.yticks(range(27), ROIs)
        plt.xticks(range(27), ROIs, rotation=90)
        plt.colorbar()
        plt.show()
        quit()


    fn_out = r'NPS' if ERS else 'RSM'
    fn_out += '_regressed' if regress_FC else ''
    fn_out += '.png'
    fp_fig = fr'result_pics/connRSA/{fn_out}'


    M = M[:210, :210]
    if len(atlas['ticks']) == 24:
        atlas['ticks'] = atlas['ticks'][:-4]
        atlas['tick_labels'] = atlas['tick_labels'][:-4]
        atlas['tick_lows'] = atlas['tick_lows'][:-4]

    vmax = np.nanquantile(M, .98)
    vmin = np.nanquantile(M, .02)
    vmin = 0


    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title=title, vmin=vmin, vmax=vmax,
                      no_avg=True, cbar_label='Correlation (r)', fp=fp_fig,
                      adjust_HC_AMY=False, dontoverride=True)


def plot_FC_mat(drop_con=False, four_tasks='7'):
    fps = prep_fps(four_tasks)
    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]

    corrs = []
    sns = None
    prev_sns = None
    # fps = ['obj7_fMRI'] # may have caused error in figure being true
    for fp in fps:
        kwargs = {'fp': fp,
                  'key': 'inc',
                  'atlas_name': 'BNA',
                  'key_vals': (1, 2, 3),
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
            pickle_wrap(load_FC, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache',
                        RAM_cache=True)
        sns = [df_sn['sn'].iloc[0] for df_sn in df_sns]
        if prev_sns is None:
            prev_sns = sns
        else:
            assert tuple(prev_sns) == tuple(sns)
        corrs.append(sn_conn)

    corrs = np.nanmean(corrs, axis=0)
    matrix_to_csv(corrs, 'connRSA_finalizing/FC_conn_for_swd.csv')

    plt.rcParams.update({'font.size': 16,
                         'font.sans-serif': 'Arial'})
    M = np.nanmean(corrs, axis=0)
    M_flat = M.flatten()
    plt.hist(M_flat, bins=100)
    plt.title('SchemeRep FC histogram')
    plt.show()
    quit()

    atlas = get_atlas(lifu_labels=False)
    fn_out = r'basic_FC.png'
    fp_fig = fr'result_pics/connRSA/{fn_out}'
    plot_network_M(corrs, None, sns, fn_out)

    M = M[:210, :210]
    if len(atlas['ticks']) == 24:
        atlas['ticks'] = atlas['ticks'][:-4]
        atlas['tick_labels'] = atlas['tick_labels'][:-4]
        atlas['tick_lows'] = atlas['tick_lows'][:-4]

    vmax = np.nanquantile(M, .98)
    vmin = np.nanquantile(M, .02)

    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], vmin=vmin, vmax=vmax, dontoverride=True,
                      adjust_HC_AMY=False,
                      title='Functional connectivity', fp=fp_fig,
                      no_avg=True, cbar_label='Correlation (r)',)


    return corrs, sns

def plot_IC_vs_FC(ERS=True, regress_FC=False, RSA_feat=True):
    corr = run_IC_analysis(ERS=ERS, regress_FC=regress_FC, get_M=True,
                           rsa_feat=False)

    corrs_FC, sns_FC = (
        pickle_wrap(plot_FC_mat, None, easy_override=False))

    atlas = get_atlas()

    region2score_IC = defaultdict(list)
    region2score_FC = defaultdict(list)

    out = np.zeros(atlas['maps'].shape)
    for i, (ROI, region_i) in enumerate(zip(atlas['ROIs'],
                                            atlas['ROI_regions'])):
        for j, (ROI_j, region_j) in enumerate(zip(atlas['ROIs'],
                                                atlas['ROI_regions'])):
            if region_i == region_j:
                if i % 2 != j % 2: continue
                region2score_IC[region_i].append(
                    np.nanmedian(corr[:, i, j], axis=0))
                region2score_FC[region_i].append(
                    np.nanmedian(corrs_FC[:, i, j], axis=0))

    v_IC = []
    v_FC = []
    for region, l_IC in region2score_IC.items():
        IC_gm = np.nanmedian(l_IC)
        l_FC = region2score_FC[region]
        FC_gm = np.nanmedian(l_FC)
        print(f'{region}, IC={IC_gm:.4f}, FC={FC_gm:.4f}')
        v_IC.append(IC_gm / FC_gm)
        v_FC.append(FC_gm / FC_gm)
    plt.plot(v_IC, color='g')
    plt.plot(v_FC, color='r')
    plt.xticks(range(len(atlas['tick_labels'])), atlas['tick_labels'],
               rotation=90)
    plt.show()



def plot_IC_mat_on_brain(ERS=False, regress_FC=False, RSA_feat=True):
    # corr = run_IC_analysis(ERS=ERS, regress_FC=regress_FC, get_M=True,
    #                        rsa_feat=RSA_feat)
    corr = run_IC_analysis(ERS=ERS, regress_FC=regress_FC, get_M=True,
                           rsa_feat=False)

    corrs_FC, sns_FC = (
        pickle_wrap(plot_FC_mat, None, easy_override=False))

    # quit()
    atlas = get_atlas()

    out = np.zeros(atlas['maps'].shape)
    # print(set(atlas['ROI_regions']))
    # quit()
    for i, (ROI, region_i) in enumerate(zip(atlas['ROIs'],
                                            atlas['ROI_regions'])):
        # if region_i not in ['EVC', 'LOC', 'sOcG', 'ITG', 'FuG', 'PhG', 'ATL']:
        #     continue
        i_l = []
        for j, region_j in enumerate(atlas['ROI_regions']):
            if region_i == region_j:
                i_l.append(corr[:, i, j])
        ROI_i_score = np.nanmean(i_l)
        print(f'{ROI}: {ROI_i_score=:.3f}')
        out[atlas['maps'].get_fdata() == i + 1] = ROI_i_score
    # quit()
    img = image.new_img_like(atlas['maps'], out)
    # quit()

    vmin = np.nanquantile(out[out != 0], 0.01)
    vmax = np.nanquantile(out[out != 0], 0.99)
    print(f'{vmin=:.5f}, {vmax=:.5f}')
    fig, axs = plotting.plot_img_on_surf(img,
                                         # threshold=thresh,
                                         # cmap=cmap, title=title,
                                         # vmin=0 if only_positive else -vmax,
                                         # vmax=.005, vmin=-.005,
                                         vmax=vmax, vmin=vmin,
                                         # vmin=-10, vmax=10,
                                         inflate=False,
                                         surf_mesh='fsaverage5',
                                         avg_method='median',
                                         # hemispheres=['right' if '_R' in region else 'left'],
                                         # cmap='turbo_r',
                                         cmap='cold_hot_r',
                                         threshold=.00001,
                                         # threshold=2,
                                         )
    plt.show()

    view = plotting.view_img(img, threshold=0, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             vmin=vmin, vmax=vmax,
                             cmap='cold_hot_r'
                             )
    view.open_in_browser()
    quit()


def plot_IT():
    atlas = get_atlas(lifu_labels=False)
    corr = run_IC_analysis(ERS=False, regress_FC=True, rsa_feat=False,
                           get_M=True)
    corr = np.nanmean(corr, axis=0)

    regions = ['ITG', 'FuG', 'PhG']
    # regions = ['Cun', 'OcG']
    # regions = ['ITG']

    new2i = []
    new_coords = []
    for i, (ROI, region) in enumerate(
            zip(atlas['ROIs'], atlas['ROI_regions'])):
        if region not in regions:
            continue
        new2i.append(i)
        new_coords.append(atlas['coords'][i])
    new_corr = np.full((len(new2i), len(new2i)), np.nan)
    for new, i in enumerate(new2i):
        for new_j, j in enumerate(new2i):
            if i % 2 != j % 2:
                continue
            new_corr[new, new_j] = corr[i, j]
    plt.imshow(new_corr)
    plt.show()

    # new_corr[new_corr > .2] = .2
    new_corr[new_corr > np.nanquantile(new_corr, .9)] = (
        np.nanquantile(new_corr, .9))

    view = plotting.view_connectome(new_corr, new_coords,
                                    symmetric_cmap=False,
                                    edge_cmap='turbo'
                                    )
    view.open_in_browser()


def matrix_to_csv(mat, fp=r'connRSA_finalizing/DistRep_FC_for_SWD.csv'):
    mat = np.nanmean(mat, axis=0)
    atlas = get_atlas()
    ticks = atlas['ROIs']
    print(mat.shape)
    df = pd.DataFrame(mat, columns=ticks, index=ticks)
    df.to_csv(fp)


    # print(corr.shape)
    # quit()

if __name__ == '__main__':
    plt.rcParams.update({'font.sans-serif': 'Arial'})
    # plot_FC_mat(drop_con=False, four_tasks='7')
    # run_IC_analysis(ERS=True, regress_FC=False)
    run_IC_analysis(ERS=False, regress_FC=True)

    # run_IC_analysis(ERS=False, regress_FC=False, rsa_feat=True)
    # run_IC_analysis(ERS=False, regress_FC=True, rsa_feat=False)
    # plot_IT()

    # plot_IC_mat_on_brain()
    # plot_IC_vs_FC()