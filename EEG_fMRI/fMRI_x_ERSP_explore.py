import os
import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from EEG_fMRI.EEG_fMRI_test import get_EEG_score_sn, get_fMRI_ar, conv
from EEG_fMRI.FC_x_ERP import get_sess_setup
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

    out_ar = np.full((len(region_l), len(region_l), len(region_l),
                      ar_fMRI.shape[-1]), np.nan, dtype=float)
    for i in range(len(region_l)):
        for j in range(i):
            for k in range(len(region_l)):
                region1 = region_l[i]
                region2 = region_l[j]
                region3 = region_l[k]

                idxs1 = ROI2idx[region1]
                idxs2 = ROI2idx[region2]
                idxs3 = ROI2idx[region3]
                v0 = conn_fMRI[np.ix_(idxs1, idxs2)].mean(axis=(0, 1))
                v1 = conn_fMRI[np.ix_(idxs3, idxs2)].mean(axis=(0, 1))
                v0 = stats.zscore(v0, nan_policy='omit')
                v1 = stats.zscore(v1, nan_policy='omit')
                fluc = np.abs(v0 - v1)
                out_ar[i, j, k] = fluc
                out_ar[j, i, k] = fluc
    return out_ar

def explore_sn(sn='02', sess='01_task-rest', avg_before=False,
               high_gamma=False, many_ROI=True,
               super_slow=False, avg_ref=False,
               abs_analysis=False, all_conn=False,
               double_speed=True, Fz_Pz_abs_dif=False):
    ar_fluc = pickle_wrap(get_fMRI_ROI_vals,
                            kwargs={'sn': sn, 'sess': sess,
                                    'combine_regions': True,
                                    'clean': False},
                            easy_override=False, verbose=-1)
    ar_fluc = np.abs(ar_fluc)
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
    out = np.full((ar_fluc.shape[0], ar_fluc.shape[0],
                   ar_fluc.shape[0]), np.nan, dtype=float)
    if EEG_fluc is None:
        return out
    # print(EEG_fluc.shape)
    EEG_fluc = conv(EEG_fluc)
    theta_fluc = np.nanmean(EEG_fluc[:, :, 2:10], axis=(1, 2))

    # print(ar_fluc.shape)
    # quit()

    ar_fluc = stats.zscore(ar_fluc, axis=-1, nan_policy='omit')
    theta_fluc = stats.zscore(theta_fluc, nan_policy='omit')[
                 None, None, None, :]
    out = np.nanmean(ar_fluc * theta_fluc, axis=-1)
    return out
    # print(out.shape)
    # quit()

    # for i in range(ar_fluc.shape[0]):
    #     for j in range(i):
    #         fluc = ar_fluc[i, j]
    #         r, p = stats.spearmanr(fluc, theta_fluc, nan_policy='omit')
    #         out[i, j] = r
    # return out

def find_max(ar):
    l = list(range(27))
    tup_l = []
    score_l = []
    tup2score = {}
    for i in l:
        for j in l:
            for k in l:
                for m in l:
                    # if (i == j or i == k or
                    #         i == m or j == k or
                    #         j == m or k == m):
                    #     continue
                    if i < j < k < m:
                        val00 = ar[i, j, k]
                        val01 = ar[j, i, k]
                        val02 = ar[i, k, j]

                        val10 = ar[i, j, m]
                        val11 = ar[j, i, m]
                        val12 = ar[i, m, j]

                        val20 = ar[i, k, m]
                        val21 = ar[k, i, m]
                        val22 = ar[i, m, k]

                        val30 = ar[j, k, m]
                        val31 = ar[k, j, m]
                        val32 = ar[j, m, k]

                        total = (val00 + val01 + val02 +
                                 val10 + val11 + val12 +
                                 val20 + val21 + val22 +
                                 val30 + val31 + val32)
                        tup_l.append((i, j, k, m))
                        score_l.append(total)
                        tup2score[(i, j, k, m)] = total

    for i, region in enumerate(atlas['ROI_regions']):
        print(f'{i}: {region=}')
    region2i = {region: i for i, region in enumerate(atlas['ROI_regions'])}
    LOC_idx = region2i['LOC']
    IFG_idx = region2i['IFG']
    IPL_idx = region2i['IPL']
    ATL_idx = region2i['ATL']
    interest_tup = tuple(sorted((LOC_idx, IFG_idx, IPL_idx, ATL_idx)))

    print(tup2score[interest_tup])



    score_l = np.array(score_l)
    nan_scores = np.isnan(score_l)
    score_l = score_l[~nan_scores]
    tup_l = np.array(tup_l)[~nan_scores]

    ranks = np.argsort(score_l)[::-1]
    print(tup_l.shape)
    print(tup_l[ranks[:10]])
    print(score_l[ranks[:10]])

    quit()

    print(np.nanmax(score_l))
    quit()

# def get_region2i():


if __name__ == '__main__':
    atlas = get_atlas(combine_regions=True, combine_bilateral=True,
                      lifu_labels=True)

    for i, region in enumerate(atlas['ROI_regions']):
        print(f'{i}: {region=}')
    region2i = {region: i for i, region in enumerate(atlas['ROI_regions'])}
    LOC_idx = region2i['LOC']
    IFG_idx = region2i['IFG']
    IPL_idx = region2i['IPL']
    ATL_idx = region2i['ATL']


    SNS, SESSES, BAD_SNS, NAME2L, NAME2SN2L = get_sess_setup()
    PSDs = []
    dfs_l = []
    ars_all = []
    sanity_all = []
    for SN in tqdm(SNS, desc='ERSP explore loop'):
        ars_sn = []
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            ar = pickle_wrap(explore_sn,
                             kwargs={'sn': SN, 'sess': SESS},
                             easy_override=True, verbose=-1)
            # ar = ar[:, 1, :]
            # except IndexError:
            #     ar = pickle_wrap(explore_sn,
            #                      kwargs={'sn': SN, 'sess': SESS},
            #                      easy_override=True, verbose=-1)
            # print(ar.shape)

            a = ar[LOC_idx, IPL_idx, IFG_idx]
            b = ar[LOC_idx, ATL_idx, IFG_idx]
            c = ar[IPL_idx, IFG_idx, ATL_idx]
            d = ar[IPL_idx, LOC_idx, ATL_idx]
            sanity = a + b + c + d
            sanity_all.append(sanity)

            ars_sn.append(ar)
        ars_sn = np.nanmean(ars_sn, axis=0)
        ars_all.append(ars_sn)
        # break
    # ar = np.nanmean(ars_all, axis=0)
    ars_all = np.array(ars_all)
    M = np.nanmean(ars_all, axis=0)
    SE = stats.sem(ars_all, axis=0, nan_policy='omit')
    t = M / SE

    find_max(t)

    p = stats.t.sf(np.abs(t), ars_all.shape[0] - 1) * 2

    t_sanity, p_sanity = stats.ttest_1samp(sanity_all, 0, nan_policy='omit')
    print(f'{t_sanity=:.3f}, {p_sanity=:.4f}')

    # print(len(atlas['tick_labels']))
    plot_connectivity(t, atlas=atlas, vmin=-3, vmax=3, minimal=False)

