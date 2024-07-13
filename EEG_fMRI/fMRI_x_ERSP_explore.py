import os
import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from EEG_fMRI.EEG_fMRI_test import get_EEG_score_sn, get_fMRI_ar
from EEG_fMRI.FC_x_ERP import conv, get_sess_setup
from atlas_utils import get_atlas
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
import scipy.stats as stats

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.io import loadmat

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

def get_fMRI_ROI_vals(sn, sess='01', combine_regions=True,
                      clean=False, ):
    ar_fMRI, key2idxs = pickle_wrap(get_fMRI_ar,
                          kwargs={'sn': sn, 'sess': sess,
                                  'combine_regions': combine_regions,
                                  'clean': clean},
                          easy_override=True, verbose=0,)
    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]

    atlas = get_atlas(natview=True, combine_regions=combine_regions)
    ROI2idx = {ROI: [] for ROI in atlas['ROI_regions']}
    for i, region in enumerate(atlas['ROI_regions']):
        ROI2idx[region].append(i)
    region_l = list(ROI2idx.keys())

    out_ar = np.full((len(region_l), len(region_l),
                      ar_fMRI.shape[-1]), np.nan, dtype=float)
    for i in range(len(region_l)):
        for j in range(i):
            region1 = region_l[i]
            region2 = region_l[j]
            idxs1 = ROI2idx[region1]
            idxs2 = ROI2idx[region2]
            fluc = conn_fMRI[np.ix_(idxs1, idxs2)].mean(axis=(0, 1))
            out_ar[i, j] = fluc
            out_ar[j, i] = fluc
    return out_ar
    #
    #     idxs = ROI2idx[region]
    #     vals = ar_fMRI[:, idxs]
    #
def prep_EEG_fluc():
    pass

def explore_sn(sn='02', sess='01_task-rest', avg_before=False,
                     high_gamma=False, many_ROI=True,
                     super_slow=False, avg_ref=False,
                     abs_analysis=False, all_conn=False,
                     double_speed=True,
                     Fz_Pz_abs_dif=False):
    ar_fluc = pickle_wrap(get_fMRI_ROI_vals,
                            kwargs={'sn': sn, 'sess': sess,
                                    'combine_regions': True,
                                    'clean': False},
                            easy_override=False, verbose=-1)
    num_TRs = ar_fluc.shape[-1]

    picks = ['F1', 'Fz', 'F2', 'F3', 'F4',
             'FC1', 'FCz', 'FC2', 'FC3', 'FC4',
             'C1', 'Cz', 'C2', 'C3', 'C4',
             'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
             'P1', 'Pz', 'P2', 'P3', 'P4']

    EEG_fluc = pickle_wrap(get_EEG_score_sn,
                           kwargs={'sn': sn, 'sess': sess,
                                   'num_TRs': num_TRs,
                                   'picks': picks,
                                   'avg_before': avg_before,
                                   'high_gamma': high_gamma,
                                   'super_slow': super_slow,
                                   'avg_ref': avg_ref,
                                   'double_speed': double_speed,
                                   'excl_before': True,
                                   'Fz_Pz_abs_dif': Fz_Pz_abs_dif,},
                           easy_override=False, verbose=-1)
    out = np.full((ar_fluc.shape[0], ar_fluc.shape[0]), np.nan, dtype=float)
    if EEG_fluc is None:
        return out
    # print(EEG_fluc.shape)
    EEG_fluc = conv(EEG_fluc)
    theta_fluc = np.nanmean(EEG_fluc[:, :, 8:61], axis=(1, 2))

    for i in range(ar_fluc.shape[0]):
        for j in range(i):
            fluc = ar_fluc[i, j]
            r, p = stats.spearmanr(fluc, theta_fluc)
            out[i, j] = r
    return out


if __name__ == '__main__':
    SNS, SESSES, BAD_SNS, NAME2L, NAME2SN2L = get_sess_setup()
    PSDs = []
    dfs_l = []
    ars_all = []
    for SN in tqdm(SNS, desc='ERSP explore loop'):
        ars_sn = []
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            ar = explore_sn(SN, SESS)
            ars_sn.append(ar)
        ars_sn = np.nanmean(ars_sn, axis=0)
        ars_all.append(ars_sn)
        # break
    ars_all = np.array(ars_all)
    M = np.nanmean(ars_all, axis=0)

    atlas = get_atlas(combine_regions=True, combine_bilateral=True,
                      lifu_labels=True)
    # print(len(atlas['tick_labels']))
    plot_connectivity(M, atlas=atlas, vmin=-0.1, vmax=0.1, minimal=False)

