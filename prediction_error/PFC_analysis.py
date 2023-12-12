from collections import defaultdict

import numpy as np

from atlas_utils import get_BN_and_resample
from fMRI_proc import get_ROI_vecs, regress_out_within_across, get_IRAFs, within_run_to_nan
from organize_bhv import get_trial_info
from org_sns import get_all_sns

from stim import get_stim_RDM, get_semantic_vectors, get_DNN_vecs
import utils
import scipy.stats as stats

from tqdm import tqdm
import pandas as pd

def get_stim_RDMs(df_sn, semantic=False, DNN_layer=2):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=DNN_layer, PCA=True)

    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'obj_abs': get_stim_RDM(df_sn, d_vecs, obj_only=True, take_abs=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, scene_only=True),
                 'scn_abs': get_stim_RDM(df_sn, d_vecs, scene_only=True, take_abs=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, dif=True),
                 'dif_abs': get_stim_RDM(df_sn, d_vecs, dif=True, take_abs=True),
                 'prod': get_stim_RDM(df_sn, d_vecs, prod=True),}
    return RDM_stims


def analyze_sn(sn, atlas, fp_fMRI_col='obj_fMRI'):
    df_sn = get_trial_info(sn)
    ROI2vecs = get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn,
                            nan_thresh=.25, org_by_region=False, inc=None)
    if ROI2vecs is None:
        print('None subject')
        return None

    PFC_vecs = []
    dmPFC = {'1 SFG_L_7_1', '2 SFG_R_7_1',
             '5 SFG_L_7_3', '6 SFG_R_7_3',
             '11 SFG_L_7_6', '12 SFG_R_7_6',
             '13 SFG_L_7_7', '14 SFG_R_7_7'}
    vmPFC = {#'179 CG_L_7_3', '180 CG_R_7_3',
             #'177 CG_L_7_2', '178 CG_R_7_2',
             #'187 CG_L_7_7', '188 CG_R_7_7',
             '45 OrG_L_6_3', '46 OrG_R_6_3',
             '47 OrG_L_6_4', '48 OrG_R_6_4',
             '49 OrG_L_6_5', '50 OrG_R_6_5'}
    for ROI, vecs in ROI2vecs.items():
        # print(ROI)
        if ROI in vmPFC or ROI in dmPFC:
            # PFC_vecs.append(vecs)
            continue

        # if ROI in dmPFC:
        #     continue
        # if 'SFG' in ROI:
        #     PFC_vecs.append(vecs)
        if 'IFG' in ROI or 'MFG' in ROI:
            PFC_vecs.append(np.nanmean(vecs, axis=1)[:, None])
            # PFC_vecs.append(vecs)
        # if 'Hipp' in ROI:
        #     PFC_vecs.append(vecs)
    PFC_vecs = np.hstack(PFC_vecs)
    # PFC_vecs = stdize(PFC_vecs, axis=0)
    # PFC_vecs_prod = utils.pb_outer(PFC_vecs, PFC_vecs, flat=False)
    # idxs = np.tril_indices_from(PFC_vecs_prod[0, :], k=-1)
    # PFC_vecs = PFC_vecs_prod[:, idxs[0], idxs[1]]

    RDM_fMRI = np.corrcoef(PFC_vecs)
    RDM_stims = get_stim_RDMs(df_sn, semantic=False, DNN_layer=2)
    key2z = {}
    RDM_fMRI = within_run_to_nan(RDM_fMRI)
    # RDM_fMRI = regress_out_within_across(RDM_fMRI)

    key2IRAF_df = {}
    for key, RDM_stim in RDM_stims.items():
        tril_idx = np.tril_indices_from(RDM_fMRI, k=-1)
        fMRI_vec = RDM_fMRI[tril_idx]
        stim_vec = RDM_stim[tril_idx]
        r, _ = stats.spearmanr(fMRI_vec, stim_vec, nan_policy='omit')
        z = np.arctanh(r)
        key2z[key] = z
        IRAFs = get_IRAFs(RDM_fMRI, RDM_stim, df_sn)
        key2IRAF_df[key] = pd.DataFrame({'IRAF': IRAFs,
                                         'hit_hit': df_sn['hit_hit']}).dropna()
        key2IRAF_df[key]['sn'] = sn
    return key2z, key2IRAF_df

def analyze_all_sn(age=1):
    atlas = get_BN_and_resample(combine_bilateral=False)
    age2sn = get_all_sns()
    key2z_all = defaultdict(list)
    key2IRAF_df_all = defaultdict(lambda: pd.DataFrame())
    for i, sn in tqdm(enumerate(age2sn[age]), desc='PFC looping sn'):
        key2z, key2IRAF_df  = analyze_sn(sn, atlas)
        if key2z is None:
            continue
        for key, z in key2z.items():
            key2z_all[key].append(z)
            key2IRAF_df_all[key] = pd.concat([key2IRAF_df_all[key], key2IRAF_df[key]])

    key2z_all['dif_abs_'] = utils.regress_out_multi([#key2z_all['obj_abs'],
                                                     #key2z_all['scn_abs'],
                                                     key2z_all['obj'],
                                                     key2z_all['scn']],
                                                     key2z_all['dif_abs'])
    key2z_all['obj_abs_'] = utils.regress_out_multi([key2z_all['scn_abs'],
                                                     key2z_all['dif_abs']],
                                                     key2z_all['obj_abs'])
    key2z_all['scn_abs_'] = utils.regress_out_multi([key2z_all['obj_abs'],
                                                     key2z_all['dif_abs'],],
                                                     key2z_all['scn_abs'])


    for key, l in key2z_all.items():
        n = (~pd.isna(l)).sum()
        M = np.nanmean(l)
        SD = np.nanstd(l)
        SE = SD / np.sqrt(n)
        t = M / SE
        p = stats.t.sf(np.abs(t), len(l)-1) # one-sided
        print(f'{key}: M={M:.3f}, SD={SD:.3f}, SE={SE:.3f}, t={t:.3f}, p={p:.3f}')

        from pymer4.models import Lmer
        formula = f'IRAF ~ 1 + (1 | sn)'
        if len(key2IRAF_df_all[key]) == 0:
            continue
        df = key2IRAF_df_all[key]
        df = df[df['hit_hit'] > 0]
        model = Lmer(formula, data=df)
        model.fit(REML=True, verbose=False, summary=False)
        summary = model.coefs
        print(summary.round(3))
        t = summary['T-stat'].loc['(Intercept)']
        p = summary['P-val'].loc['(Intercept)']
        print(f'\tlmer {key}: t={t:.3f}, p={p:.3f}')
        print()


def find_temporal_correlation(ar):
    rs = []
    for i in tqdm(range(len(ar))):
        for j in range(i):
            l0 = ar[i]
            l1 = ar[j]
            r, _ = stats.pearsonr(l0, l1)
            rs.append(r)
    M = np.nanmean(rs)
    med = np.nanmedian(rs)
    print(f'M={M:.4f}, med={med:.4f}')
    quit()


if __name__ == '__main__':
    analyze_all_sn()
    # sanity_test()
