from time import time

import numpy as np
from matplotlib import pyplot as plt
from numba import jit, prange
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from old.networks import prep_networks
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_semantic_vectors
from utils import pickle_wrap, stdize
from functools import cache
from sklearn import decomposition

# suppress RuntimeWarning: All-NaN slice
from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning,
               message="All-NaN slice encountered")


import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

@cache
def get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run, second_order,
                flat=True, within_nan=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    try:
        with open(fp_focus1, 'rb') as f:
            RSM = np.load(f)
    except FileNotFoundError:
        # print('Not found!')
        RSM = np.full((114, 114), np.nan)

    if within_nan:
        RSM = within_run_to_nan(RSM)

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
                     second_order, within_nan=True):
    fp2RSMs = {}
    for fp in fps:
        RSMs = []
        for ROI in ROIs:
            RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                              second_order, within_nan=within_nan)
            RSMs.append(RSM)
        RSMs = np.array(RSMs)
        fp2RSMs[fp] = RSMs
    return fp2RSMs

def get_cross_IC_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, within_nan=True):
    kw = {'ROIs': ROIs, 'trial_similarity': trial_similarity,
          'stdize_by_run': stdize_by_run, 'second_order': second_order,
          'within_nan': within_nan, 'sn': sn, 'fps': fps}
    fp2RSMs = pickle_wrap(load_fp2RSMs, None, kwargs=kw,
                          easy_override=False, verbose=-1)

    t_st = time()
    l = []
    for fp0, RSMs0 in fp2RSMs.items():
        for fp1, RSMs1 in fp2RSMs.items():
            if fp0 == fp1:
                continue
            pair = [RSMs0, RSMs1]
            l.append(pair)
    l = np.array(l)

    bad_cols = np.all(np.isnan(l), axis=(0, 1, 2)) # nan across all ROIs and fps
    l = l[:, :, :, ~bad_cols]
    l = stdize(l, axis=-1)
    corrs = numba_fp_x_fp_RSMs(l)
    t_end = time()
    print(f'Numba corr calc: {t_end - t_st:.3f}')
    return corrs

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fp_x_fp_RSMs(l): # 4 seconds first then 3 seconds vs. 9 w/ numpy below
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in prange(num_fp_pairs): # prange gives an 8x speedup??
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        for j0 in range(num_ROIs):
            for j1 in range(num_ROIs):
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

def run_IC_analysis():
    # semantic = False
    cross = True
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    atlas = get_atlas()
    ROIs = atlas['ROIs']

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117', '118', '119', '120',
           '123', '124', '126', '127', '128', '129', '130', '134', '135',
           '136', '137', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214', '216', '217', '218',
           '219', '221', '222', '225', '227', '232', '233', '235']

    four_tasks = '8'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'second_order': second_order,
              'ROIs': ROIs,
              'within_nan': True}

    corrs = []

    ventral_idxs = get_idxs('Ventral')
    occ_idxs = get_idxs('Occipital')
    # print(test)
    # quit()
    # fps = [fp for fp in fps if 'con' not in fp]
    for i, sn in tqdm(enumerate(sns), desc=f'Looping IC: {cross=}'):
        # if i > 3:
        #     break
        # sn_corrs = []
        kwargs['sn'] = sn
        if cross:
            kwargs['fps'] = fps
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False)
            # corr = np.nanmean(corrs, axis=0)
        else:
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)
        # print(sn_corrs.shape)
        # quit()

        # corr = stats.trim_mean(sn_corrs, 0.1, axis=0)
        corr = np.nanmean(sn_corrs, axis=0)

        corr[:, 222:] = np.nan
        corr[222:, :] = np.nan

        if cross:

            diag = np.diag(corr)
            M_diag = np.nanmean(diag)
            print(f'{M_diag=:.3f}')
            diag_prod = np.sqrt(np.outer(diag, diag))
            corr -= diag_prod

        # corr = stats.zscore(corr, nan_policy='omit')
        corr -= np.nanmean(corr)
        corr /= np.nanstd(corr)

        corrs.append(corr)

    corrs = np.array(corrs)
    # print(corrs.shape)
    # quit()

    # corrs = stats.zscore(corrs, axis=(1, 2))
    # qu

    corrs[:, *np.diag_indices(corrs.shape[1])] = np.nan
    corrs_ventral = corrs[:, ventral_idxs][:, :, ventral_idxs]
    M_ventral = np.nanmean(corrs_ventral, axis=(1, 2))
    corrs_occ = corrs[:, occ_idxs][:, :, occ_idxs]
    M_occ = np.nanmean(corrs_occ, axis=(1, 2))
    # plt.scatter(M_ventral, M_occ)
    # for sn in sns:
    #     idx = sns.index(sn)
    #     plt.text(M_ventral[idx], M_occ[idx], sn)
    # plt.plot([-.14, 0], [-.14, 0])
    # plt.show()

    t, p = stats.ttest_rel(M_ventral, M_occ)
    N = M_ventral.shape[0]
    d = t / np.sqrt(N)
    print(f'Ventral vs. occ ({N=}): {t=:.3f}, {p=:.3f}, {d=:.3f}')

    M = np.nanmean(corrs, axis=0)



    SE = stats.sem(corrs, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(corrs), axis=0)
    t = M / SE

    atlas = get_atlas()
    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='Cross IC' if cross else 'Traditional IC',
                      no_avg=True, cbar_label='',)


if __name__ == '__main__':
    run_IC_analysis()
