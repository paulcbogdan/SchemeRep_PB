import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
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
                flat=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    try:
        with open(fp_focus1, 'rb') as f:
            RSM = np.load(f)
    except FileNotFoundError:
        RSM = np.full((114, 114), np.nan)

    if flat:
        trils = np.tril_indices(RSM.shape[0], k=-1)
        return RSM[trils]
    else:
        return RSM

def get_IC_mat(sn, ROIs, fp, trial_similarity, stdize_by_run, second_order):
    RSMs = []
    for ROI in ROIs:
        RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                          second_order)
        RSMs.append(RSM)
    corr = np.corrcoef(RSMs)
    corr[np.diag_indices_from(corr)] = np.nan
    return corr



def run_IC_analysis():
    semantic = False
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
              'ROIs': ROIs}

    corrs = []
    for sn in tqdm(sns):
        sn_corrs = []
        for fp in fps:
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1)
            sn_corrs.append(corr)
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
