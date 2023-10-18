from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from atlas_utils import get_BN_and_resample, get_combined_BNA, get_atlas
from organize_bhv import get_trial_info, get_all_sns
from nilearn import image

from plot_gen import plot_connectivity
from stim import get_stim_RDM, get_semantic_vectors, get_DNN_vecs
from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
import utils
import scipy.stats as stats

from tqdm import tqdm
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
filterwarnings('ignore', category=RuntimeWarning,
               message='Degrees of freedom <= 0')

def within_run_to_nan(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        RDM_[low:high, low:high] = np.nan
    return RDM_

def regress_out_within_across(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    within_zero = np.zeros(RDM.shape)
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        within_idxs = np.arange(low, high)
        within_zero[np.ix_(within_idxs, within_idxs)] = 1
    within_zero[np.diag_indices_from(within_zero)] = 0
    M_within = np.nanmean(RDM_[within_zero == 1])
    M_between = np.nanmean(RDM_[within_zero == 0])
    RDM_[within_zero == 1] = RDM_[within_zero == 1] - M_within
    RDM_[within_zero == 0] = RDM_[within_zero == 0] - M_between
    return RDM_

def RDM_x_RDM(fMRI_RDM, stim_RDM):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    fMRI_vec = fMRI_RDM_[tril_idx]
    stim_vec = stim_RDM[tril_idx]
    r, _ = stats.spearmanr(fMRI_vec, stim_vec, nan_policy='omit')
    z = np.arctanh(r)
    return z

def get_IRAFs(fMRI_RDM, stim_RDM, df_sn):
    '''
    matmul all took 0.001 seconds
    matmul semi (one loop, inner matmul) took 0.008 seconds
    scipy pearsonr took 0.013 seconds
    scipy spearmanr took 0.055 seconds
    '''
    fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    fMRI_RDM_[np.diag_indices_from(fMRI_RDM)] = np.nan
    stim_RDM[np.diag_indices_from(stim_RDM)] = np.nan
    fMRI_RDM_std = stdize(fMRI_RDM_, axis=0, nans=True)
    stim_RDM_std = stdize(stim_RDM, axis=0, nans=True)
    IRAFs = fMRI_RDM_std * stim_RDM_std
    IRAFs = np.nanmean(IRAFs, axis=0)
    IRAFs = IRAFs[df_sn['obj'].argsort()]
    return IRAFs


def get_IRAF_connectivity_matrix(ROI_to_IRAF, n_ROIs, ROIs):
    first_key = next(iter(ROI_to_IRAF.keys()))
    subj_timeseries_shape = (n_ROIs, ROI_to_IRAF[first_key].shape[1])
    subj2timeseries = defaultdict(lambda: np.full(subj_timeseries_shape,
                                                  np.nan))
    for i, ROI in enumerate(ROIs):
        ar = ROI_to_IRAF[ROI]
        for subj_j in range(ar.shape[0]):
            subj2timeseries[subj_j][i, :] = ar[subj_j, :]
    all_matricies = []
    for sn, ar in subj2timeseries.items():
        mat = np.corrcoef(ar)
        mat[np.diag_indices_from(mat)] = 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i % 2 == j % 2:
                    if (i < mat.shape[0] - 1):
                        mat[i, j] = mat[i+1, j]
        mat[mat > .99] = .99
        mat[mat < -.99] = -.99
        mat = np.arctanh(mat)
        all_matricies.append(mat)
    return all_matricies

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
                 org_by_region=False, inc=None, easy_override=False):
    ROIs = atlas['ROIs']
    ROI_regions = atlas['ROI_regions']
    n_ROIs = len(ROIs)
    n_regions = len(np.unique(ROI_regions))
    nan_str = f'_nan{nan_thresh}' if nan_thresh != .25 else ''
    org_by_region_str = '_oByR' if org_by_region else ''
    inc_str = '' if inc is None else \
        '_Con' if inc == 1 else \
            '_Inc' if inc == 2 else '_Neu'
    fp_cache = fr'cache\ROI2vecs\sn{sn}_{fp_fMRI_col}{inc_str}_nROI{n_ROIs}' \
               fr'_reg{n_regions}{org_by_region_str}{nan_str}.pkl'
    f = lambda: get_ROI_vecs_(df_sn, fp_fMRI_col, atlas,
                  nan_thresh=nan_thresh,
                  org_by_region=org_by_region)
    r2vecs = pickle_wrap(fp_cache, f, verbose=True,
                         easy_override=True)
    return r2vecs


def get_ROI_vecs_(df_sn, fp_fMRI_col, atlas,
                  nan_thresh=.25, org_by_region=False):
    n_nans = pd.isna(df_sn[fp_fMRI_col]).sum()
    if n_nans:
        raise ValueError(f'Found NaNs in {fp_fMRI_col}, {n_nans=}')

    img = image.load_img(df_sn[fp_fMRI_col]).get_fdata()
    n_nans = np.isnan(img).sum()
    print(f'Total number of NaNs: {n_nans}')
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    ROI2vecs = {}
    region2vecs = defaultdict(list)
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        region_vecs = img[atlas_roi]
        voxels_w_nan = np.isnan(region_vecs).any(axis=1)

        # p_nans_per_trial = np.sum(np.isnan(region_vecs), axis=0) / region_vecs.shape[0]

        n_nans_ROI = np.sum(voxels_w_nan)
        # print(f'{ROI}: {n_nans_ROI/len(voxels_w_nan):.4f}')
        if n_nans_ROI / len(voxels_w_nan) > nan_thresh:  # more than 10%
            continue
        # print(f'{ROI}: {np.mean(voxels_w_nan):.4f}')
        region_vecs = region_vecs[~voxels_w_nan, :]
        # print(f'{ROI}: {n_nans_ROI / len(voxels_w_nan)}')

        # region_vecs[:, p_nans_per_trial > nan_thresh] = np.nan

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

def get_mean_activity(region_vecs, sort_by):
    M = np.nanmean(region_vecs, axis=1)
    M_sorted = []
    for a, _ in sorted(zip(M, sort_by), key=lambda x: x[1]):
        M_sorted.append(a)
    return np.array(M_sorted)

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
                 'lifu': get_stim_RDM(df_sn, d_vecs, lifu=True),
                 'lifu_sem': get_stim_RDM(df_sn, d_vecs, lifu_sem=True)}
    return RDM_stims

def include_bhv(bhv, df_sn_):
    bhv_cols = ['enc_trial',
                'obj_run', 'scn_run',
                'obj', 'scene', 'obj_rename',
                'scene_rename', 'inc', 'perceived_con',
                'hit_hit', 'vis_hit', 'con_hit',
                'vis_outlier', 'con_outlier', 'obj_outlier',
                'scn_outlier', 'bl_outlier']
    for key in bhv_cols:
        try:
            bhv[key].append(df_sn_[key].values)
        except KeyError:
            bhv[key].append(np.full(len(df_sn_), np.nan))

def prep_variables(sn, cin, atlas, org_by_region):
    df_sn = get_trial_info(sn, easy_override=True)
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
                 ROI_to_z, ROI_to_IRAF, triple_z, ROI_to_activity, bhv,
                 fp_fMRI_col='fp_fMRI', org_by_region=False,
                 shuffle=False):
    print(f'Onto: {sn}')

    df_sn, n_trials, ROI_to_RDM_fMRI, ROIs, ROI_nums = \
        prep_variables(sn, cin, atlas, org_by_region)

    if shuffle:
        shuffle_df_sn(df_sn, fp_fMRI_col)

    include_bhv(bhv, df_sn)
    RDM_stims = get_all_stim_RDMs(df_sn, d_vecs) # TODO: don't repeat every sn
    ROI2vecs = get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn, inc=cin,
                            nan_thresh=.5, org_by_region=org_by_region)
    # for j, (ROI, ROI_num) in tqdm(enumerate(zip(ROIs, ROI_nums)),
    #                               desc='looping ROIs outer', total=len(ROIs),
    #                               leave=True, ncols=80, position=0):

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
            cond = fp_fMRI_col.split('_')[0]
            outliers = np.array(bhv[f'{cond}_outlier'][-1])
            ROI2vecs[ROI][outliers, :] = np.nan
            region_vecs = np.copy(ROI2vecs[ROI])
            M_activity =  get_mean_activity(region_vecs, df_sn['obj'])[None, :]
            ROI_to_activity[ROI] = np.append(ROI_to_activity[ROI], M_activity,
                                             axis=0)
            # print(ROI_to_activity[ROI])
            RDM_fMRI = np.corrcoef(region_vecs)
            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()

            # print(np.sum(np.isnan(region_vecs)))
            # region_vecs = stdize(region_vecs, axis=1, nans=True)
            # region_vecs_a = region_vecs[None, :, :]
            # region_vecs_b = region_vecs[:, None, :]
            # prod = region_vecs_a * region_vecs_b
            # RDM_fMRI = np.nanmean(prod, axis=2)
            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()
            # print(RDM_fMRI.shape)
            # quit()
            # RDM_fMRI = np.ma.corrcoef(np.ma.masked_invalid(region_vecs))
            # print(np.sum(~np.isnan(RDM_fMRI)))
            # quit()

            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()
            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()
            # IRAFs = np.full((1, n_trials), np.nan)
            # for key in stim_keys:
            #     zs = []
            #     for inc_val in [1, 2, 3]:
            #         region_vecs = np.copy(ROI2vecs[ROI])
            #         inc_vals = bhv['inc'][-1]
            #         inc_bools = inc_vals == inc_val
            #         # inc_bools = np.full(len(inc_vals), True)
            #         region_vecs[~inc_bools, :] = np.nan
            #         RDM_fMRI = np.corrcoef(region_vecs)
            #         z = RDM_x_RDM(RDM_fMRI, RDM_stims[key])
            #         zs.append(z)
            #         IRAFs_inc = get_IRAFs(RDM_fMRI, RDM_stims[key], df_sn)
            #         IRAFs[0, inc_bools] = IRAFs_inc[inc_bools]
            #     z = np.nanmean(zs)
            #     # assert np.sum(np.isnan(zs)) == 0, f'nan in zs: {zs=}'
            #     ROI_to_z[key][ROI].append(z)
            #     ROI_to_IRAF[key][ROI] = \
            #         np.append(ROI_to_IRAF[key][ROI], IRAFs, axis=0)

            for key in stim_keys:
                z = RDM_x_RDM(RDM_fMRI, RDM_stims[key])
                # print(f'[{sn}, {key}] {z=}')
                # plt.imshow(RDM_fMRI)
                # plt.show()
                # quit()
                ROI_to_z[key][ROI].append(z)
                IRAFs = get_IRAFs(RDM_fMRI, RDM_stims[key], df_sn)
                ROI_to_IRAF[key][ROI] = \
                    np.append(ROI_to_IRAF[key][ROI], IRAFs[None, :], axis=0)

    apply_regress_out_multi(atlas, org_by_region, n_trials, stim_keys,
                            ROI_to_z, ROI_to_IRAF)

    # for key in stim_keys:
    #     triple_z[key].append(get_triple_connectivity(ROI_to_RDM_fMRI,
    #                                                  RDM_stims[key], ROIs))

    return df_sn



def mass_RDM_x_RDM(age=1, cin=None, semantic=False, DNN_layer=2, PCA_obj=True,
                   bilateral=False, combine_regions=False, org_by_region=False,
                   fp_fMRI_col='fp_fMRI', shuffle=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=DNN_layer, PCA=True,
                              PCA_obj=PCA_obj)
    atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)
    ret = fp_fMRI_col in ['con_fMRI', 'vis_fMRI', 'dif_bl-vis', 'dif_obj-vis']
    age2sn = get_all_sns(ret=ret)

    n_trials = 114 if cin is None else 38

    ROI_to_z = defaultdict(lambda: defaultdict(list))
    ROI_to_IRAF = defaultdict(lambda: defaultdict(lambda: np.full((0, n_trials),
                                                                  np.nan)))
    ROI_to_activity = defaultdict(lambda: np.full((0, n_trials), np.nan))
    stim_keys = ['obj', 'obj_abs',
                 'scn', 'scn_abs',
                 'dif', 'dif_abs',
                 'prd', 'prd_abs',
                 'add', 'add_abs',
                 'lifu', 'lifu_sem']
    stim_keys = ['obj', 'obj_abs',
                 'scn', 'scn_abs',
                 'dif_abs']
    stim_keys = ['obj', 'scn']#, 'dif_abs', 'lifu']
    triple_z = {}
    rxr_all = {}
    for key in stim_keys:
        triple_z[key] = []
        rxr_all[key] = []

    if org_by_region:
        atlas['n_ROIs'] = len(atlas['tick_labels'])

    bhv = defaultdict(list)
    sns = []
    dfs_sn = []
    for i, sn in tqdm(enumerate(age2sn[age]),
                      desc=f'Looping subjects: age2sn[{age}]'):
        df_sn = analyze_subj(sn, cin, d_vecs, atlas, stim_keys, ROI_to_z,
                             ROI_to_IRAF, triple_z, ROI_to_activity, bhv,
                             fp_fMRI_col=fp_fMRI_col,
                             org_by_region=org_by_region, shuffle=shuffle)
        sns.append(sn)
        dfs_sn.append(df_sn)

    IRAF_conn = {}
    for key in ROI_to_IRAF.keys():
        IRAF_conn[key] = get_IRAF_connectivity_matrix(ROI_to_IRAF[key],
                                                      atlas['n_ROIs'],
                                                      atlas['ROIs'])

    d_out = {'rxr': rxr_all,
            'triple_z': triple_z,
            'IRAF_conn': IRAF_conn,
            'activity': ROI_to_activity,
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

        # for key in keys_sans_obj_scn:
        #     key_mod = f'{key}__'
        #     ROI_to_z[key_mod][ROI] = utils.regress_out_multi([ROI_to_z['obj_abs'][ROI],
        #                                                      ROI_to_z['scn_abs'][ROI]],
        #                                                     ROI_to_z[key][ROI])
        #     ROI_to_IRAF[key_mod][ROI] = np.full(ROI_to_IRAF[key][ROI].shape, np.nan)
        #     for trial_i in range(n_trials):
        #         IRAF_dif_ = utils.regress_out_multi([ROI_to_IRAF['obj_abs'][ROI][:, trial_i],
        #                                              ROI_to_IRAF['scn_abs'][ROI][:, trial_i]],
        #                                             ROI_to_IRAF[key][ROI][:, trial_i])
        #         ROI_to_IRAF[key_mod][ROI][:, trial_i] = IRAF_dif_


def run_multi_settings():
    # semantic = False
    combine_regions = False
    bilateral = False # combines bilateral ROIs/regions
    org_by_region = False
    PCA_obj = True
    assert not (org_by_region and combine_regions), \
        'Cannot combine regions and organize by region'
    # fp_fMRI_col = 'obj_fMRI' # maps onto columns defined in get_trial_info
    # fp_fMRI_col = 'scn_fMRI'
    inc = None
    # if True:
    for age in [1, 2]:
        for inc in [None]:
            for fp_fMRI_col in ['obj_fMRI', 'scn_fMRI',
                                'con_fMRI', 'vis_fMRI',
                                'bl_fMRI']:
            # for fp_fMRI_col in ['scn_fMRI',]:
            # for fp_fMRI_col in ['dif_bl-vis', 'dif_bl-obj', 'dif_obj-vis']:
                for DNN_layer, semantic in [
                                            (2, False),
                                            # (4, False),
                                            # (6, False),
                                            # (-1, False),
                                            (False, True),
                                            ]: # (True, False),
                    # if semantic and PCA_obj:
                    #     continue

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
                    d = pickle_wrap(fp_out, f, easy_override=True,
                                    verbose=True)
                # quit()


# a = [[1, 2, 3, np.nan], [4, 2, 8, 10], [1, 2, 3, 10]]
# a = [[1, 2, 3, ], [4, 2, 8, ], [1, 2, 3, ]]
#
# a_m=ma.masked_invalid(a)
# # print(a_m)
# # print(a[a_m])
# r = np.ma.corrcoef(ma.masked_invalid(a))
# print(r)
# quit()
# r = np.corrcoef(a)
# print(r)
# quit()

if __name__ == '__main__':
    # TODO: 3 way correlation, Region A RDM x Region B RDM x Stimulus RDM
    run_multi_settings()

