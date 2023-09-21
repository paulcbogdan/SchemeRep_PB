from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs
from ROIs import get_BN_and_resample, get_combined_BNA
from fMRI_analysis import get_ROI_vecs
from organize_bhv import get_trial_info, get_all_sns
from nilearn import image

from stim_vec import get_stim_RDM
from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
from wordvec_get_vectors import get_semantic_vectors
import utils
import scipy.stats as stats

from tqdm import tqdm
from pathlib import Path

def get_stim_RDMs(df_sn, semantic=False, early=True):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(early=early, PCA=True)

    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'obj_abs': get_stim_RDM(df_sn, d_vecs, obj_only=True, take_abs=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, scene_only=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, dif=True),
                 'dif_abs': get_stim_RDM(df_sn, d_vecs, dif=True, take_abs=True),
                 'prod': get_stim_RDM(df_sn, d_vecs, prod=True),}
    return RDM_stims

def load_and_get_ROI_vecs(df_sn, atlas, vec_prod=False):
    img = image.load_img(df_sn['fp_fMRI']).get_fdata()
    ROI2vecs, _ = get_ROI_vecs(atlas['ROIs'], atlas['ROI_nums'], atlas, img,
                               atlas['ROI_regions'],
                               vec_prod=vec_prod)
    return ROI2vecs, _

def analyze_sn(sn, atlas):
    df_sn = get_trial_info(sn)
    n_ROIs = len(atlas['ROIs'])
    vec_prod_str = '_vec_prod' if False else ''
    fp_cache = fr'cache\ROI2vecs\sn{sn}_nROI{n_ROIs}{vec_prod_str}.pkl'
    ROI2vecs, _ = pickle_wrap(fp_cache,
                              lambda: load_and_get_ROI_vecs(df_sn,
                                                            atlas),
                              easy_override=False, verbose=True)
    PFC_vecs = []
    dmPFC = {'1 SFG_L_7_1', '2 SFG_R_7_1',
             '5 SFG_L_7_3', '6 SFG_R_7_3',
             '11 SFG_L_7_6', '12 SFG_R_7_6',
             '13 SFG_L_7_7', '14 SFG_R_7_7'}
    for ROI, vecs in ROI2vecs.items():
        # print(ROI)
        if ROI in dmPFC:
            continue
        if  'MFG' in ROI or 'IFG' in ROI or 'SFG' in ROI:
            PFC_vecs.append(np.nanmean(vecs, axis=1)[:, None])
            # PFC_vecs.append(vecs)
        # if 'Hipp' in ROI:
        #     PFC_vecs.append(vecs)
    # quit()
    PFC_vecs = np.hstack(PFC_vecs)
    # PFC_vecs = stdize(PFC_vecs, axis=0)
    # PFC_vecs_prod = utils.pb_outer(PFC_vecs, PFC_vecs, flat=False)
    # idxs = np.tril_indices_from(PFC_vecs_prod[0, :], k=-1)
    # PFC_vecs = PFC_vecs_prod[:, idxs[0], idxs[1]]

    RDM_fMRI = np.corrcoef(PFC_vecs)
    RDM_stims = get_stim_RDMs(df_sn, semantic=True, early=True)
    key2z = {}
    for key, RDM_stim in RDM_stims.items():
        tril_idx = np.tril_indices_from(RDM_fMRI, k=-1)
        fMRI_vec = RDM_fMRI[tril_idx]
        stim_vec = RDM_stim[tril_idx]
        r, _ = stats.spearmanr(fMRI_vec, stim_vec)
        z = np.arctanh(r)
        key2z[key] = z
    return key2z

def analyze_all_sn(age=1):
    atlas = get_BN_and_resample(combine_bilateral=False)
    age2sn = get_all_sns()
    key2z_all = defaultdict(list)
    for i, sn in tqdm(enumerate(age2sn[age]), desc='PFC looping sn'):
        key2z = analyze_sn(sn, atlas)
        for key, z in key2z.items():
            key2z_all[key].append(z)

    for key, l in key2z_all.items():
        M = np.nanmean(l)
        SD = np.nanstd(l)
        SE = SD / np.sqrt(len(l))
        t = M / SE
        p = stats.t.sf(np.abs(t), len(l)-1) # one-sided
        print(f'{key}: M={M:.3f}, SD={SD:.3f}, SE={SE:.3f}, t={t:.3f}, p={p:.3f}')


if __name__ == '__main__':
    analyze_all_sn()
