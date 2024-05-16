from time import time

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_semantic_vectors
from utils import pickle_wrap
from functools import cache
from sklearn import decomposition

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

    # plt.imshow(RSM)
    # plt.show()

    if within_nan:
        RSM = within_run_to_nan(RSM)

    # plt.imshow(RSM)
    # plt.show()
    # quit()
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
    # corr = np.corrcoef(RSMs)
    corr = np.ma.corrcoef(RSMs)
    corr[np.diag_indices_from(corr)] = np.nan
    return corr

def get_cross_IC_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
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

    # corrs = []
    # for fp0, RSMs0 in fp2RSMs.items():
    #     RSMs0 = RSMs0[:, None, :]
    #     # print(RSMs0.shape)
    #     for fp1, RSMs1 in fp2RSMs.items():
    #         if fp0 == fp1:
    #             continue
    #         RSMs1 = RSMs1[None, :, :]
    #         # print(RSMs1.shape)
    #         # quit()
    #         corr = np.nanmean(RSMs0 * RSMs1, axis=-1)
    #         # plt.imshow(corr)
    #         # plt.show()
    #         # quit()
    #         corrs.append(corr)
    #
    #
    #
    t_st = time()
    l = []
    for fp0, RSMs0 in fp2RSMs.items():
        for fp1, RSMs1 in fp2RSMs.items():
            if fp0 == fp1:
                continue
            pair = [RSMs0, RSMs1]
            l.append(pair)
    l = np.array(l)
    # print(l.shape)
    # plt.imshow(l[0][0])
    # plt.show()

    bad_cols = np.all(np.isnan(l), axis=(0, 1, 2)) # nan across all ROIs and fps
    l = l[:, :, :, ~bad_cols]
    # print(l.shape)
    corrs = numba_fp_x_fp_RSMs(l)
    # quit()
    # print(bad_cols)
    # print(np.mean(bad_cols))
    # #
    # bad_cols = np.any(np.isnan(l), axis=(0, 1))
    # print(bad_cols)
    # print(np.mean(bad_cols))
    # quit()


    # corrs = np.array(corrs)
    t_end = time()
    print(f'Numba corr calc: {t_end - t_st:.3f}')
    # quit()
    return corrs

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fp_x_fp_RSMs(l):
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in range(num_fp_pairs):
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        for j0 in range(num_ROIs):
            for j1 in range(num_ROIs):
                out[i, j0, j1] = np.mean(RSM0[:, j0] * RSM1[:, j1])
    return out

# calculate IC between tasks and normalize by an ROIs IC with itself
#   this is needed to rule out that high/low IC is due to data reliablity
#   e.g., Occipital 1 may have higher IC with occipital 2 than IPL 1 has with
#   IPL 2, but this could be due to occipital 1 & 2 having more reliable data

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
    for i, sn in tqdm(enumerate(sns), desc=f'Looping IC: {cross=}'):
        # if i > 3:
        #     break
        # sn_corrs = []
        kwargs['sn'] = sn
        if cross:
            kwargs['fps'] = fps
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=True)
            # corr = np.nanmean(corrs, axis=0)
        else:
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)
        corrs.append(np.nanmean(sn_corrs, axis=0))
    corrs = np.array(corrs)

    M = np.nanmean(corrs, axis=0)
    print(M)
    SE = stats.sem(corrs, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(corrs), axis=0)
    t = M / SE

    atlas = get_atlas()
    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='eh', no_avg=True,
                      cbar_label='t-value',)


if __name__ == '__main__':
    run_IC_analysis()
