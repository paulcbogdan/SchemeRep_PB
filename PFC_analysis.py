import os.path
import pickle

from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs
from ROIs import get_BN_and_resample, get_combined_BNA
from fMRI_analysis import get_ROI_vecs, get_ROI_vecs_, within_run_to_nan, regress_out_within_across
from old.test_lifu import get_stim_RDM_lifu
from organize_bhv import get_trial_info, get_all_sns
from nilearn import image

from stim_vec import get_stim_RDM
from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
from wordvec_get_vectors import get_semantic_vectors
import utils
import scipy.stats as stats

from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

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

def load_and_get_ROI_vecs(sn, atlas, fp_fMRI_col='scn_fMRI',
                          vec_prod=False, nan_thresh=.25,
                          org_by_region=False, combine_regions=False):
    # # img = image.load_img(df_sn['fp_fMRI']).get_fdata()
    # img = image.load_img(df_sn['fp_f MRI']).get_fdata()
    #
    # ROI2vecs, _ = get_ROI_vecs(atlas['ROIs'], atlas['ROI_nums'], atlas, img,
    #                             atlas['ROI_regions'],
    #                             vec_prod=vec_prod)
    # if combine_regions:
    #     n_ROIs = len(atlas['ROI_regions'])
    # else:
    n_ROIs = len(atlas['ROIs'])
    n_regions = len(np.unique(atlas['ROI_regions']))
    vec_prod_str = '_vp' if vec_prod else ''
    nan_str = f'_nan{nan_thresh}' if nan_thresh != .25 else ''
    org_by_region_str = '_oByR' if org_by_region else ''
    fp_cache = fr'cache\ROI2vecs\sn{sn}_{fp_fMRI_col}_nROI{n_ROIs}_reg{n_regions}' \
               fr'{vec_prod_str}{org_by_region_str}{nan_str}.pkl'
    if not os.path.isfile(fp_cache):
        print(f'ROI2vecs cache not found: {fp_cache}')
        return None, None
    with open(fp_cache, 'rb') as f:
        ROI2vecs, _ = pickle.load(f)

    return ROI2vecs, _

def analyze_sn(sn, atlas):
    df_sn = get_trial_info(sn)
    # n_ROIs = len(atlas['ROIs'])
    # vec_prod_str = '_vec_prod' if False else ''
    ROI2vecs, _ = load_and_get_ROI_vecs(sn, atlas, fp_fMRI_col='vis_fMRI')
    if ROI2vecs is None:
        return None
    # fp_cache = fr'cache\ROI2vecs\sn{sn}_nROI{n_ROIs}{vec_prod_str}.pkl'
    # ROI2vecs, _ = pickle_wrap(fp_cache,
    #                           lambda: load_and_get_ROI_vecs(df_sn,
    #                                                         atlas),
    #                           easy_override=False, verbose=True)
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
            # PFC_vecs.append(np.nanmean(vecs, axis=1)[:, None])
            PFC_vecs.append(vecs)
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
    # RDM_fMRI = within_run_to_nan(RDM_fMRI)
    RDM_fMRI = regress_out_within_across(RDM_fMRI)

    for key, RDM_stim in RDM_stims.items():
        tril_idx = np.tril_indices_from(RDM_fMRI, k=-1)
        fMRI_vec = RDM_fMRI[tril_idx]
        stim_vec = RDM_stim[tril_idx]
        r, _ = stats.spearmanr(fMRI_vec, stim_vec, nan_policy='omit')
        z = np.arctanh(r)
        key2z[key] = z
    return key2z

def analyze_all_sn(age=1):
    atlas = get_BN_and_resample(combine_bilateral=False)
    age2sn = get_all_sns()
    key2z_all = defaultdict(list)
    for i, sn in tqdm(enumerate(age2sn[age]), desc='PFC looping sn'):
        key2z = analyze_sn(sn, atlas)
        if key2z is None:
            continue
        for key, z in key2z.items():
            key2z_all[key].append(z)

    key2z_all['dif_abs_'] = utils.regress_out_multi([key2z_all['obj_abs'],
                                                     key2z_all['scn_abs'],
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
        M = np.nanmean(l)
        SD = np.nanstd(l)
        SE = SD / np.sqrt(len(l))
        t = M / SE
        p = stats.t.sf(np.abs(t), len(l)-1) # one-sided
        print(f'{key}: M={M:.3f}, SD={SD:.3f}, SE={SE:.3f}, t={t:.3f}, p={p:.3f}')

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


def sanity_test(sn='102'):
    atlas = get_BN_and_resample(combine_bilateral=False)
    df_sn = get_trial_info(sn)
    # ROI2vecs, _ = load_and_get_ROI_vecs(sn, atlas, fp_fMRI_col='obj_fMRI')
    img = image.load_img(df_sn['obj_fMRI']).get_fdata()
    ROI2vecs, _ = get_ROI_vecs_(atlas['ROIs'], atlas['ROI_nums'], atlas,
                                img, atlas['ROI_regions'])

    RDM_stims = get_stim_RDMs(df_sn, semantic=False, DNN_layer=2)
    RDM_stim = RDM_stims['obj']
    # RDM_stim = get_stim_RDM_lifu(df_sn)
    # RDM_stim = np.random.normal(size=RDM_stim.shape)
    # RDM_stim_flat = RDM_stim[np.tril_indices_from(RDM_stim, k=-1)]
    all_data = []
    for ROI, vecs in tqdm(ROI2vecs.items(), desc='looping ROIs'):
        fMRI_RDM = np.corrcoef(vecs)


        trial_per_run = fMRI_RDM.shape[0] // 3
        for run in range(3):
            low = run * trial_per_run
            high = (run + 1) * trial_per_run
            fMRI_RDM[low:high, low:high] = np.nan

        trial_rs = []
        for i in range(fMRI_RDM.shape[0]):
            fMRI_vec_std = np.delete(fMRI_RDM[i, :], i)
            stim_vec_std = np.delete(RDM_stim[i, :], i)
            fMRI_vec_std = stdize(fMRI_vec_std, nans=True)
            stim_vec_std = stdize(stim_vec_std, nans=True)

            fMRI_nans = np.isnan(fMRI_vec_std)
            stim_nans = np.isnan(stim_vec_std)
            either_nan = fMRI_nans | stim_nans
            fMRI_vec_std = fMRI_vec_std[~either_nan]
            stim_vec_std = stim_vec_std[~either_nan]
            r = (fMRI_vec_std @ stim_vec_std) / len(fMRI_vec_std)
            trial_rs.append(r)
        all_data.append(trial_rs)
    all_data = np.vstack(all_data)
    plt.imshow(all_data)
    # find_temporal_correlation(all_data)
    plt.xlabel('Item')
    plt.ylabel('ROI')
    plt.title('IRAF based on only within-run\nIRAF for 2nd layer DNN, participant 102')
    # plt.title('Shuffled trials\nIRAF for 2nd layer DNN, participant 102')
    plt.colorbar()
    plt.show()
    quit()

if __name__ == '__main__':
    analyze_all_sn()

    # sanity_test()
