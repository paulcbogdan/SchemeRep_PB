from datetime import datetime

from utils import pickle_wrap
from collections import defaultdict

import numpy as np

from atlas_utils import get_atlas
from organize_bhv import get_trial_info
from org_sns import get_sns
from nilearn import image
import matplotlib.pyplot as plt

from stim import get_stim_RDM, get_semantic_vectors, get_DNN_vecs
from utils import stdize, nan_ar, defaultdict_to_dict
import utils
import scipy.stats as stats

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
filterwarnings('ignore', category=RuntimeWarning,
               message='Degrees of freedom <= 0')
filterwarnings('ignore', category=RuntimeWarning,
               message='NaNs of infinite values are')

def within_run_to_nan(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        RDM_[low:high, low:high] = np.nan
    # TODO: Fix, this won't work properly except for on encoding!!
    return RDM_

def RDM_x_RDM_by_run(fMRI_RDM, RSM_stim, corr='spear'):
    RSM_stim[np.diag_indices_from(RSM_stim)] = np.nan
    fMRI_RDM[np.diag_indices_from(fMRI_RDM)] = np.nan
    zs = []
    trial_per_run = RSM_stim.shape[0] // 3
    for run0 in range(3):
        for run1 in range(3):
            if run1 < run0:
                continue
            low0 = run0 * trial_per_run
            high0 = (run0 + 1) * trial_per_run
            low1 = run1 * trial_per_run
            high1 = (run1 + 1) * trial_per_run
            stim_flat = RSM_stim[low0:high0, low1:high1].flatten()
            fMRI_flat = fMRI_RDM[low0:high0, low1:high1].flatten()
            nans = np.isnan(stim_flat) | np.isnan(fMRI_flat)
            n_nans = np.sum(nans)
            assert n_nans == 0 or n_nans == 38, f'RDM x RDM bad nans: {n_nans=}'
            stim_flat = stim_flat[~nans]
            fMRI_flat = fMRI_flat[~nans]
            if corr == 'spear':
                # print(f'{fMRI_flat}')
                # print(f'{stim_flat}')
                r, _ = stats.spearmanr(fMRI_flat, stim_flat)
                z = np.arctanh(r)
            elif corr == 'corr':
                r, _ = stats.pearsonr(fMRI_flat, stim_flat)
                z = np.arctanh(r)
            elif corr == 'euc':
                z = -np.mean(fMRI_flat - stim_flat)
            else:
                raise KeyError(f'conn must be \"spear\", \"corr\", or \"euc\", not {corr}')
            zs.append(z)
            # print(f'{z=}')
    return np.nanmean(zs)

def RDM_x_RDM(fMRI_RDM, stim_RDM, corr='spear', within_to_nan=True):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    if within_to_nan:
        fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    else:
        fMRI_RDM_ = fMRI_RDM
    fMRI_flat = fMRI_RDM_[tril_idx]
    stim_flat = stim_RDM[tril_idx]
    nans = np.isnan(fMRI_flat) | np.isnan(stim_flat)
    fMRI_flat = fMRI_flat[~nans]
    stim_flat = stim_flat[~nans]
    if corr == 'spear':
        r, _ = stats.spearmanr(fMRI_flat, stim_flat)
        z = np.arctanh(r)
    elif corr == 'corr':
        r, _ = stats.pearsonr(fMRI_flat, stim_flat)
        z = np.arctanh(r)
    elif corr == 'euc':
        z = -np.mean(fMRI_flat - stim_flat)
    else:
        raise KeyError(f'conn must be \"spear\", \"corr\", or \"euc\", not {corr}')

    return z

def get_IRAFs(fMRI_RDM, stim_RDM, df_sn, within_to_nan=True,
              by_run=False, second_order='corr'):
    '''
    matmul all took 0.001 seconds
    matmul semi (one loop, inner matmul) took 0.008 seconds
    scipy pearsonr took 0.013 seconds
    scipy spearmanr took 0.055 seconds
    '''

    if within_to_nan:
        fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    elif by_run:
        fMRI_RDM_within_nan = within_run_to_nan(fMRI_RDM)
        fMRI_RDM_ = fMRI_RDM.copy()
        fMRI_RDM_[~np.isnan(fMRI_RDM_within_nan)] = np.nan
        # raise NotImplementedError
    else:
        fMRI_RDM_ = fMRI_RDM
    fMRI_RDM_[np.diag_indices_from(fMRI_RDM)] = np.nan
    stim_RDM[np.diag_indices_from(stim_RDM)] = np.nan
    stim_RDM_ = stim_RDM.copy()
    if second_order == 'corr':
        fMRI_RDM_std = stdize(fMRI_RDM_, axis=0, nans=True)
        stim_RDM_std = stdize(stim_RDM, axis=0, nans=True)
        IRAFs = np.nanmean(fMRI_RDM_std * stim_RDM_std, axis=0)
        IRAFs = np.arctanh(IRAFs)
    elif second_order == 'spear':
        fMRI_RDM_[np.isnan(stim_RDM_)] = np.nan
        stim_RDM_[np.isnan(fMRI_RDM_)] = np.nan
        fMRI_RDM_r = stats.rankdata(fMRI_RDM_, axis=0, nan_policy='omit')
        stim_RDM_r = stats.rankdata(stim_RDM_, axis=0, nan_policy='omit')
        fMRI_RDM_r = stdize(fMRI_RDM_r, axis=0, nans=True)
        stim_RDM_r = stdize(stim_RDM_r, axis=0, nans=True)
        IRAFs = np.nanmean(fMRI_RDM_r * stim_RDM_r, axis=0)
        IRAFs = np.arctanh(IRAFs)
    else:
        raise NotImplementedError(f'get_IRAFs {second_order=}')
    IRAFs = IRAFs[df_sn['obj'].argsort()]
    return IRAFs

def get_triple_connectivity(ROI_to_RDM_fMRI, RDM_stim, ROIs):
    trils = np.tril_indices_from(ROI_to_RDM_fMRI[ROIs[0]], k=-1)
    for ROI, rdm in ROI_to_RDM_fMRI.items():
        ROI_to_RDM_fMRI[ROI] = (rdm - np.nanmean(rdm[trils])) / np.nanstd(rdm[trils])
    RDM_stim = (RDM_stim - np.nanmean(RDM_stim[trils])) / np.nanstd(RDM_stim[trils])

    triple_prod_mat = np.zeros((len(ROIs), len(ROIs)))
    for j, (ROI0, RDM0) in enumerate(ROI_to_RDM_fMRI.items()):
        for k, (ROI1, RDM1) in enumerate(ROI_to_RDM_fMRI.items()):
            if j >= k: continue
            prod = np.multiply(np.multiply(RDM0, RDM1), RDM_stim)
            prod = prod[trils]
            triple_prod_mat[ROIs.index(ROI1), ROIs.index(ROI0)] = \
                triple_prod_mat[ROIs.index(ROI0), ROIs.index(ROI1)] = \
                np.nanmean(prod)
    return triple_prod_mat

def get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn, nan_thresh=.25,
                 org_by_region=False, inc=None, drop_nan_voxels=True,
                 easy_override=False, combine_regions=False,
                 verbose=0):
    ROIs = atlas['ROIs']
    ROI_regions = atlas['ROI_regions']
    shenyang_key = atlas['shenyang']
    # shenyang_key = False
    n_ROIs = len(ROIs)
    n_regions = len(np.unique(ROI_regions))
    nan_str = f'_nan{nan_thresh}' if nan_thresh != .25 else ''
    nan_str += '_dropNaNvox' if drop_nan_voxels else ''
    sh_str = f'_sh' if shenyang_key else ''
    org_by_region_str = '_oByR' if org_by_region else ''
    inc_str = '' if inc is None else \
        '_Con' if inc == 1 else \
            '_Inc' if inc == 2 else '_Neu'
    combine_str = '_comb' if combine_regions else '_noComb'
    stim_order = ''.join(df_sn['obj'].values).replace(' ', '')
    stim_order = stim_order[::len(stim_order) // 10]
    fp_cache = fr'cache\ROI2vecs\sn{sn}_{fp_fMRI_col}{inc_str}_nROI{n_ROIs}' \
               fr'_reg{n_regions}{org_by_region_str}{nan_str}{combine_str}' \
               fr'{sh_str}{stim_order}.pkl'
    if verbose >= 0: print(f'Load ROI2vecs: {fp_cache=}')
    f = lambda: get_ROI_vecs_(df_sn, fp_fMRI_col, atlas, nan_thresh=nan_thresh,
                  org_by_region=org_by_region, drop_nan_voxels=drop_nan_voxels)
    r2vecs = pickle_wrap(f, fp_cache, easy_override=easy_override,
                         verbose=verbose)
    return r2vecs




def get_ROI_vecs_(df_sn, fp_fMRI_col, atlas,
                  nan_thresh=.25, org_by_region=False,
                  drop_nan_voxels=True):
    # print(df_sn[fp_fMRI_col])
    # quit()
    try:
        img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    except TypeError as e:
        sn = df_sn['sn'].values[0]
        print(df_sn[fp_fMRI_col])
        for i in range(10):
            print(f'Error in loading ({fp_fMRI_col}): {e}')
            print(f'\t{sn=}')
        quit()

    n_nans = np.isnan(img).sum()
    print(f'Total number of NaNs: {n_nans/114:.1f}')
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    ROI2vecs = {}
    region2vecs = defaultdict(list)
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        region_vecs = img[atlas_roi]

        voxels_w_nan = np.isnan(region_vecs[:, good_idxs]).any(axis=1)
        voxels_all_nan = (~np.isnan(region_vecs)).any(axis=1)

        n_nans_ROI = np.sum(voxels_w_nan)
        p_nan_any = n_nans_ROI / len(voxels_w_nan)
        n_nan_overall = np.sum(np.isnan(region_vecs[:, good_idxs]))
        p_nan_overall = n_nan_overall / np.prod(region_vecs[:, good_idxs].shape)

        if p_nan_any > nan_thresh:  # more than 10%
            print(f'Skip ({region}): {p_nan_any=:.2f}, {p_nan_overall=:.2f}')
            continue
        if drop_nan_voxels:
            region_vecs = region_vecs[~voxels_w_nan, :]
        region_vecs = region_vecs.T
        ROI2vecs[ROI] = region_vecs
        if org_by_region:
            region2vecs[region].append(np.nanmean(region_vecs, axis=1))

    region2vecs = dict(region2vecs)

    for region, l in region2vecs.items():
        region2vecs[region] = np.array(l).T

    if org_by_region:
        return region2vecs
    else:
        return ROI2vecs

def get_all_stim_RDMs(df_sn, d_vecs):
    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'obj_abs': get_stim_RDM(df_sn, d_vecs, obj_only=True,
                                         take_abs=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, scene_only=True),
                 'scn_abs': get_stim_RDM(df_sn, d_vecs, scene_only=True,
                                         take_abs=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, dif=True),
                 'dif_abs': get_stim_RDM(df_sn, d_vecs, dif=True,
                                         take_abs=True),
                 'prd': get_stim_RDM(df_sn, d_vecs, prod=True),
                 'prd_abs': get_stim_RDM(df_sn, d_vecs, prod=True,
                                         take_abs=True),
                 'add': get_stim_RDM(df_sn, d_vecs, add=True),
                 'add_abs': get_stim_RDM(df_sn, d_vecs, add=True,
                                         take_abs=True),
                 # 'lifu': get_stim_RDM(df_sn, d_vecs, lifu=True),
                 # 'lifu_sem': get_stim_RDM(df_sn, d_vecs, lifu_sem=True)
                 }
    return RDM_stims

def include_bhv(bhv, df_sn_):
    bhv_cols = ['enc_trial',
                'obj_run', 'scn_run',
                'obj', 'scene', 'obj_rename',
                'scene_rename', 'inc', 'perceived_con',
                'hit_hit', 'vis_hit', 'con_hit',
                'vis_outlier', 'con_outlier', 'obj_outlier',
                'scn_outlier', 'bl_outlier', 'obj2_outlier']
    for key in bhv_cols:
        try:
            bhv[key].append(df_sn_[key].values)
        except KeyError:
            bhv[key].append(np.full(len(df_sn_), np.nan))

def prep_variables(sn, cin, atlas, org_by_region, ret):
    print(f'{ret=}')
    df_sn = get_trial_info(sn, easy_override=False)
    if cin is not None:
        df_sn = df_sn[df_sn['inc'] == cin]
        n_trials = 38
    else:
        n_trials = 114
    ROI_to_RDM_fMRI = {}
    ROI_nums = atlas['ROI_nums']
    if org_by_region:
        ROIs = atlas['tick_labels']
    else:
        ROIs = atlas['ROIs']
    return df_sn, n_trials, ROI_to_RDM_fMRI, ROIs, ROI_nums

def shuffle_df_sn(df_sn, fp_fMRI_col):
    cond = fp_fMRI_col.split('_')[0]
    for run in range(1, 4):
        df_run = df_sn.loc[df_sn[f'{cond}_run'] == run]
        df_sn.loc[df_sn['obj_run'] == run, fp_fMRI_col] = \
            np.random.permutation(df_run[fp_fMRI_col].values)

def analyze_subj(sn, cin, d_vecs, atlas, stim_keys,
                 ROI_to_z, ROI_to_IRAF, ROI_to_activity, bhv,
                 fp_fMRI_col='fp_fMRI', org_by_region=False,
                 shuffle=False, ):
    print(f'Onto: {sn}')

    df_sn, n_trials, ROI_to_RDM_fMRI, ROIs, ROI_nums = \
        prep_variables(sn, cin, atlas, org_by_region,
                       ret=('vis' in fp_fMRI_col or 'con' in fp_fMRI_col))
    sess = fp_fMRI_col.split('_')[0].replace('2', '').replace('3', '').\
        replace('4', '').replace('7', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    if shuffle:
        shuffle_df_sn(df_sn, fp_fMRI_col)

    include_bhv(bhv, df_sn.sort_values(by='obj'))
    RDM_stims = get_all_stim_RDMs(df_sn, d_vecs) # TODO: don't repeat every sn
    ROI2vecs = get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn, inc=cin,
                            nan_thresh=.8, org_by_region=org_by_region,
                            easy_override=False)

    for j, (ROI, ROI_num) in enumerate(zip(ROIs, ROI_nums)):
        if ROI not in ROI2vecs:
            M_activity = np.full((1, n_trials), np.nan)
            ROI_to_activity[ROI] = np.append(ROI_to_activity[ROI], M_activity,
                                             axis=0)
            RDM_fMRI = nan_ar((n_trials, n_trials))
            ROI_to_RDM_fMRI[ROI] = RDM_fMRI
            for key in stim_keys:
                z = np.nan
                ROI_to_z[key][ROI].append(z)
                IRAFs = np.full((1, n_trials), np.nan)
                ROI_to_IRAF[key][ROI] = \
                    np.append(ROI_to_IRAF[key][ROI], IRAFs, axis=0)
        else:
            region_vecs = np.copy(ROI2vecs[ROI])
            M_activity = \
                region_vecs[df_sn['obj'].argsort(), :].mean(axis=1)[None, :]
            ROI_to_activity[ROI] = np.append(ROI_to_activity[ROI], M_activity,
                                             axis=0)
            RDM_fMRI = np.corrcoef(region_vecs)
            for key in stim_keys:
                z = RDM_x_RDM(RDM_fMRI, RDM_stims[key])
                ROI_to_z[key][ROI].append(z)
                IRAFs = get_IRAFs(RDM_fMRI, RDM_stims[key], df_sn)
                ROI_to_IRAF[key][ROI] = \
                    np.append(ROI_to_IRAF[key][ROI], IRAFs[None, :], axis=0)
    return df_sn.sort_values(by='obj')



def mass_RDM_x_RDM(age=1, cin=None, semantic=False, DNN_layer=2, PCA_obj=True,
                   bilateral=False, combine_regions=False, org_by_region=False,
                   fp_fMRI_col='fp_fMRI', shuffle=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=DNN_layer, PCA=True,
                              PCA_obj=PCA_obj)

    new_space = '2' in fp_fMRI_col
    atlas = get_atlas(combine_regions=combine_regions, combine_bilateral=bilateral,
                      new_space=new_space)
    # if combine_regions:
    #     atlas = get_combined_BNA(combine_bilateral=bilateral)
    # else:
    #     atlas = get_BN_and_resample(combine_bilateral=bilateral)
    ret = fp_fMRI_col in ['con_fMRI', 'vis_fMRI', 'dif_bl-vis', 'dif_obj-vis',
                          'con2_fMRI', 'vis2_fMRI', 'con3_fMRI', 'vis3_fMRI', ]
    atlas = get_atlas(combine_regions=combine_regions, combine_bilateral=bilateral)
    age2sn = get_sns(fp_fMRI_col)

    n_trials = 114 if cin is None else 38

    ROI_to_z = defaultdict(lambda: defaultdict(list))
    ROI_to_IRAF = defaultdict(lambda: defaultdict(lambda: np.full((0, n_trials),
                                                                  np.nan)))
    ROI_to_activity = defaultdict(lambda: np.full((0, n_trials), np.nan))
    stim_keys = ['obj', 'obj_abs',
                 'scn', 'scn_abs',
                 'dif', 'dif_abs',
                 # 'prd', 'prd_abs',
                 # 'add', 'add_abs',
                 ]#'lifu', 'lifu_sem']


    if org_by_region:
        atlas['n_ROIs'] = len(atlas['tick_labels'])

    bhv = defaultdict(list)
    sns = []
    dfs_sn = []
    for i, sn in tqdm(enumerate(age2sn[age]),
                      desc=f'Looping subjects: age2sn[{age}]'):
        df_sn = analyze_subj(sn, cin, d_vecs, atlas, stim_keys, ROI_to_z,
                             ROI_to_IRAF, ROI_to_activity, bhv,
                             fp_fMRI_col=fp_fMRI_col,
                             org_by_region=org_by_region, shuffle=shuffle)
        sns.append(sn)
        dfs_sn.append(df_sn)

    apply_regress_out_multi(atlas, org_by_region, n_trials, stim_keys,
                            ROI_to_z, ROI_to_IRAF)

    d_out = {'activity': ROI_to_activity,
             'IRAFs_ROI': ROI_to_IRAF,
             'z': ROI_to_z,
             'bhv': bhv,
             'sns': sns,
             'df_sn': dfs_sn}
    d_out = defaultdict_to_dict(d_out)
    return d_out

def apply_regress_out_multi(atlas, org_by_region, n_trials, stim_keys,
                            ROI_to_z, ROI_to_IRAF):
    ROIs = atlas['tick_labels'] if org_by_region else atlas['ROIs']
    keys_sans_obj_scn = [key for key in stim_keys if
                         ('obj' not in key) and ('scn' not in key) and
                         ('lifu' not in key)]
    for ROI in ROIs:
        for key in keys_sans_obj_scn:
            key_mod = f'{key}_'
            ROI_to_z[key_mod][ROI] = utils.regress_out_multi([ROI_to_z['obj'][ROI],
                                                             ROI_to_z['scn'][ROI]],
                                                            ROI_to_z[key][ROI])
            ROI_to_IRAF[key_mod][ROI] = np.full(ROI_to_IRAF[key][ROI].shape, np.nan)
            for trial_i in range(n_trials):
                IRAF_dif_ = utils.regress_out_multi([ROI_to_IRAF['obj'][ROI][:, trial_i],
                                                     ROI_to_IRAF['scn'][ROI][:, trial_i]],
                                                    ROI_to_IRAF[key][ROI][:, trial_i])
                ROI_to_IRAF[key_mod][ROI][:, trial_i] = IRAF_dif_

    for ROI in ROIs:
        key_mod = f'scn_'
        ROI_to_z[key_mod][ROI] = utils.regress_out_multi([ROI_to_z['obj'][ROI]],
                                                         ROI_to_z['scn'][ROI])
        ROI_to_IRAF[key_mod][ROI] = np.full(ROI_to_IRAF['scn'][ROI].shape, np.nan)
        for trial_i in range(n_trials):
            IRAF_dif_ = utils.regress_out_multi([ROI_to_IRAF['obj'][ROI][:, trial_i]],
                                                ROI_to_IRAF['scn'][ROI][:, trial_i])
            ROI_to_IRAF[key_mod][ROI][:, trial_i] = IRAF_dif_


def run_multi_settings():
    combine_regions = False
    bilateral = False # combines bilateral ROIs/regions
    org_by_region = False
    PCA_obj = True
    assert not (org_by_region and combine_regions), \
        'Cannot combine regions and organize by region'
    inc = None
    for age in [2, 1]:
        for inc in [None]:
            for DNN_layer, semantic in [
                #(2, False),
                # (-1, False),
                # (4, False),
                # (6, False),
                # (True, False),
                (False, True),
            ]:  # (True, False),
                for fp_fMRI_col in [
                    'scn7_fMRI',
                    # 'obj7_fMRI',
                    # 'bl7_fMRI',
                    # 'con7_fMRI',
                    # 'vis7_fMRI',
                ]:

                    RSA_fn = utils.get_RSA_fn(inc, age, semantic, DNN_layer,
                                              fp_fMRI_col, PCA_obj=PCA_obj,
                                              combine_regions=combine_regions,
                                              bilateral=bilateral,
                                              vec_prod=False,
                                              org_by_region=org_by_region)
                    fp_out = fr'cache/RSA/{RSA_fn}.pkl'
                    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                    f = lambda: mass_RDM_x_RDM(age=age, cin=inc,
                                               DNN_layer=DNN_layer,
                                               semantic=semantic,
                                               bilateral=bilateral,
                                               combine_regions=combine_regions,
                                               org_by_region=org_by_region,
                                               fp_fMRI_col=fp_fMRI_col,
                                               PCA_obj=PCA_obj,
                                               shuffle=False
                                               )
                    dt_max = datetime(2024, 2, 17, 1, 0, 0, 0)
                    d = pickle_wrap(f, fp_out, easy_override=True,
                                    verbose=True, dt_max=dt_max)


if __name__ == '__main__':
    # TODO: 3 way correlation, Region A RDM x Region B RDM x Stimulus RDM
    run_multi_settings()

