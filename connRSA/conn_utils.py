import warnings
from collections import defaultdict
from time import time

import numpy as np
import scipy.stats as stats

from Study1A.load_Study1A_funcs import get_ROI_vecs
from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from Utils.pickle_wrap_funcs import pickle_wrap
from stim import scipy_dist
from utils import tril_flat, stdize, pb_outer_euc, pb_outer

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')


def corr_matrix_last_two_dim(ar, nans=True, euc_dist=False,
                             stdize=True):
    # Second-to-last dim should be your variable
    # Last dim should be a time series
    m = np.nanmean if nans else np.mean
    # Creates nan in rs and rs_flat if every value in time series is identical
    if ar.shape[-1] > 1 and stdize:
        s = np.nanstd if nans else np.std
        M = np.expand_dims(m(ar, axis=-1), axis=-1)
        SD = np.expand_dims(s(ar, axis=-1), axis=-1)
        ar_std = (ar - M) / SD
        ar_std0 = np.expand_dims(ar_std, axis=-2)
        ar_std1 = np.expand_dims(ar_std, axis=-3)
    else:
        ar_std0 = np.expand_dims(ar, axis=-2)
        ar_std1 = np.expand_dims(ar, axis=-3)

    if euc_dist:
        # stdize can't actually make a difference because it applies to both
        #   ar_std0 and ar_std1
        rs = -abs(ar_std0 - ar_std1)
    else:
        rs = ar_std0 * ar_std1
    rs = m(rs, axis=-1)
    rs_flat = tril_flat(rs)
    return rs, rs_flat


def corr_last_dim(ar0, ar1, nans=True, euc_dist=False, stdize=True):
    # Last dim of both should be a time series
    # stdize by last dim for correlation
    m = np.nanmean if nans else np.mean
    if stdize:
        s = np.nanstd if nans else np.std
        M0 = np.expand_dims(m(ar0, axis=-1), axis=-1)
        SD0 = np.expand_dims(s(ar0, axis=-1), axis=-1)
        ar0_ = (ar0 - M0) / SD0
        M1 = np.expand_dims(m(ar1, axis=-1), axis=-1)
        SD1 = np.expand_dims(s(ar1, axis=-1), axis=-1)
        ar1_ = (ar1 - M1) / SD1
    else:
        ar0_ = ar0
        ar1_ = ar1
    if euc_dist:
        rs = -abs(ar0_ - ar1_)
    else:
        rs = ar0_ * ar1_
    rs = m(rs, axis=-1)
    return rs


def get_conn_vecs(vecs, vecs1=None, conn='euc', stdize_by_run=False):
    vecs = stdize(vecs, axis=0, nans=True, stdize_by_run=stdize_by_run)
    if vecs1 is None:
        vecs1 = vecs
        cross = False
    else:
        vecs1 = stdize(vecs1, axis=0, nans=True, stdize_by_run=stdize_by_run)
        cross = True
    if conn == 'euc':
        vecs = pb_outer_euc(vecs, vecs1, tril=not cross,
                            flat=cross, nan_diag=not cross)
    elif conn == 'prod':
        vecs = pb_outer(vecs, vecs1, tril=not cross,
                        flat=cross, nan_diag=not cross)
    else:
        raise ValueError(f'conn={conn} not recognized')
    return vecs


def calculate_cross_region_vecs(ROI2vecs, networks, conn='euc'):
    ROI2vecs_new = {}
    for network, ROIs in networks.items():
        if isinstance(ROIs, tuple):
            ROIs0 = ROIs[0]
            ROIs1 = ROIs[1]
            vecs0_l = []
            for ROI0_0 in ROIs0:
                vecs0_l.append(ROI2vecs[ROI0_0])
            vecs0 = np.concatenate(vecs0_l, axis=1)
            vecs1_l = []
            for ROI1_1 in ROIs1:
                vecs1_l.append(ROI2vecs[ROI1_1])
            vecs1 = np.concatenate(vecs1_l, axis=1)
            vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
            # print(f'{vecs0.shape=}, {vecs1.shape=}, {vecs.shape=}')
            # quit()
        else:
            vecs_l = []
            for ROI0 in ROIs:
                vecs0 = ROI2vecs[ROI0]
                for ROI1 in ROIs:
                    if ROI0 == ROI1:
                        continue
                    vecs1 = ROI2vecs[ROI1]
                    vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
                    vecs_l.append(vecs)
            vecs = np.concatenate(vecs_l, axis=1)
        assert vecs.shape[0] == 114, f'{vecs.shape=}'
        ROI2vecs_new[network] = vecs
    return ROI2vecs_new


def cluster_regions(ROI2vecs, networks):
    ROI2vecs_new = {}

    OrG_ROIs = ['OrG_L_6_1', 'OrG_R_6_1', 'OrG_L_6_2', 'OrG_R_6_2', 'OrG_L_6_3', 'OrG_R_6_3',
                'OrG_L_6_4', 'OrG_R_6_4', 'OrG_L_6_5', 'OrG_R_6_5', 'OrG_L_6_6', 'OrG_R_6_6', ]
    OrG_ROI2idx = {ROI: i for i, ROI in enumerate(OrG_ROIs)}

    for network, ROIs in networks.items():
        vecs_l = []
        for ROI in ROIs:
            if ROI in OrG_ROI2idx:
                ROI_vec = ROI2vecs['OrG'][:, OrG_ROI2idx[ROI]]
                vecs_l.append(ROI_vec[:, None])
            elif ROI == 'ATL_L_7_3':
                ROI_vec = ROI2vecs['ATL'][:, -2]
                vecs_l.append(ROI_vec[:, None])
            elif ROI == 'ATL_R_7_3':
                ROI_vec = ROI2vecs['ATL'][:, -1]
                vecs_l.append(ROI_vec[:, None])
                # vecs_l.append([ROI2vecs['ATL'][:, -2]])
            elif ROI in ROI2vecs:
                vecs_l.append(ROI2vecs[ROI])
            elif f'{ROI}_L' in ROI2vecs:
                vecs_l.append(ROI2vecs[f'{ROI}_L'])
                vecs_l.append(ROI2vecs[f'{ROI}_R'])
            else:
                if ROI != 'CG':
                    warnings.warn(f'ROI={ROI} not found')

        vecs = np.concatenate(vecs_l, axis=1)
        # print(vecs.shape)
        # quit()

        ROI2vecs_new[network] = vecs
    return ROI2vecs_new


def get_BOLD_ctrl(combine_regions, sn, fp0, df_sn, org_by_region,
                  easy_override):
    atlas = get_atlas(combine_regions=False)
    ROIs = get_BNA_ROIs('BNA_region')
    ROIs = [ROI.split('_')[0] for ROI in ROIs[::2]]
    assert combine_regions
    ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                             drop_nan_voxels=False,
                             org_by_region=org_by_region,
                             easy_override=easy_override,
                             combine_regions=False,
                             verbose=-1)

    ROI2vecs0_ = {}
    for ROI in ROIs:
        l = []
        for ROI_small, vecs in ROI2vecs0.items():
            if ROI in ROI_small:
                l.append(vecs)
                vecs -= np.nanmean(vecs, axis=1)[:, None]
        vecs = np.concatenate(l, axis=1)
        ROI2vecs0_[ROI] = vecs
    ROI2vecs0 = ROI2vecs0_
    return ROI2vecs0


def get_ROI_vecs_wrap(sn, atlas, fp0, df_sn, fp1=None, networks=None,
                      org_by_region=True, cross_region=False,
                      conn=None, combine_regions=False,
                      easy_override=False):
    # print(f'{combine_regions=}')
    # print(f'{org_by_region=}')

    assert not (combine_regions and org_by_region)
    # print(conn)
    # quit()

    if conn == 'BOLD_ctrl':
        ROI2vecs0 = get_BOLD_ctrl(combine_regions, sn, fp0, df_sn, org_by_region,
                                  easy_override)
    else:
        ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                                 drop_nan_voxels=False,
                                 org_by_region=org_by_region,
                                 easy_override=easy_override,
                                 combine_regions=combine_regions,
                                 verbose=-1)

    if fp1 is not None:
        if conn == 'BOLD_ctrl':
            ROI2vecs1 = get_BOLD_ctrl(combine_regions, sn, fp1, df_sn, org_by_region,
                                      easy_override)
        else:
            ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=org_by_region,
                                     easy_override=easy_override,
                                     combine_regions=combine_regions,
                                     verbose=-1)
    else:
        ROI2vecs1 = None

    if networks:
        if cross_region:
            ROI2vecs0 = calculate_cross_region_vecs(ROI2vecs0, networks,
                                                    conn=conn)
        else:
            ROI2vecs0 = cluster_regions(ROI2vecs0, networks,
                                        # bold_ctrl=conn == 'BOLD_ctrl'
                                        )

        if fp1 is not None:
            if cross_region:
                ROI2vecs1 = calculate_cross_region_vecs(ROI2vecs1, networks,
                                                        conn=conn)
            else:
                ROI2vecs1 = cluster_regions(ROI2vecs1, networks,
                                            # bold_ctrl=conn == 'BOLD_ctrl'
                                            )

    if fp1 is not None:
        return ROI2vecs0, ROI2vecs1
    else:
        return ROI2vecs0


def get_trial_x_trial(vecs, vecs1=None, trial_similarity='corr'):
    # Fixed bug (Nov 21, where this was stdizing for euc & seuc)
    vecs0 = vecs[None, :, :]
    if vecs1 is not None:
        vecs1 = vecs1[:, None, :]
    else:
        vecs1 = vecs[:, None, :]
    if trial_similarity == 'corr':
        vecs0 = stdize(vecs0, axis=2, nans=True)
        vecs1 = stdize(vecs1, axis=2, nans=True)
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=-1)  # Pearson
    elif trial_similarity == 'mink':
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Euclidean
        # RSM_fMRI = prune_RSM_outliers(RSM_fMRI, z=3)
    elif trial_similarity == 'euc':
        # print('TOAST')
        # import matplotlib.pyplot as plt

        nan_cols = np.isnan(vecs0).any(axis=(0, 1))
        # print(vecs0.shape)
        # print(vecs1.shape)
        # print(nan_cols.shape)
        vecs0 = vecs0[:, :, ~nan_cols]
        vecs1 = vecs1[:, :, ~nan_cols]

        # plt.imshow(vecs0[0, ...], aspect='auto')
        # plt.show()

        # from scipy.spatial import distance
        # RSM_fMRI = -distance.cdist(vecs0[0, ...], vecs0[0, ...], 'euclidean')

        RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
                              metric='euclidean', )
        # RSM_fMRI =
        # print(RSM_fMRI.shape)
        # plt.imshow(RSM_fMRI)
        # plt.show()
        # quit()
        # quit()
        # RSM_fMRI = prune_RSM_outliers(RSM_fMRI, z=3)
    elif trial_similarity == 'spear':
        # print(vecs0)
        vecs0_r = stats.rankdata(vecs0, method='average', axis=-1,
                                 nan_policy='omit')
        # print(vecs0_r)
        vecs1_r = stats.rankdata(vecs1, method='average', axis=-1,
                                 nan_policy='omit')
        vecs0_r = stdize(vecs0_r, axis=2, nans=True)
        vecs1_r = stdize(vecs1_r, axis=2, nans=True)
        RSM_fMRI = np.nanmean(vecs0_r * vecs1_r, axis=-1)  # Spearman
    elif trial_similarity in ['seuclidean', 'mahalanobis']:
        RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
                              metric=trial_similarity)
        # RSM_fMRI = prune_RSM_outliers(RSM_fMRI, z=3)
    else:
        raise ValueError(f'{trial_similarity=} not supported')
    return RSM_fMRI


def get_mean_conn_trialwise(ROIs_l, ROI2vecs, conn='euc'):
    conn_trialwise = np.full((len(ROIs_l), len(ROIs_l), 114), np.nan)
    for i, ROI0 in enumerate(ROIs_l):
        vecs0 = ROI2vecs[ROI0]
        vecs0 = vecs0[:, None]
        for j, ROI1 in enumerate(ROIs_l):
            if ROI1 == ROI0:
                conn_trialwise[i, j, :] = np.nan
                continue
            vecs1 = ROI2vecs[ROI1]
            vecs1 = vecs1[:, None]
            if conn == 'euc':
                vecs = np.abs(vecs0 - vecs1)  # speed-up
            else:
                vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
            conn_trialwise[i, j, :] = np.squeeze(vecs)  # prev (114, 1)
    mean_conn_trialwise = np.nanmean(conn_trialwise, axis=0)
    return mean_conn_trialwise


def prep_for_ROI_analysis(ROI2vecs, PFC, PFC2):
    ROI2vecs_M = {}
    for ROI, vecs in ROI2vecs.items():
        if PFC or PFC2:
            if 'PFC' in ROI or 'OFC' in ROI:
                ROI2vecs_M[ROI] = np.nanmean(vecs, axis=1)
            elif 'SFG' in ROI or 'MFG' in ROI or 'IFG' in ROI or 'OrG' in ROI:
                ROI2vecs_M[ROI] = np.nanmean(vecs, axis=1)
            elif PFC2 and 'ACC' in ROI:
                ROI2vecs_M[ROI] = np.nanmean(vecs, axis=1)
        else:
            ROI2vecs_M[ROI] = np.nanmean(vecs, axis=1)
    return ROI2vecs_M


def prep_for_pairwise(ROI2vecs, atlas):
    region2vecs = defaultdict(list)
    # print(len(ROI2vecs))
    # print(list(ROI2vecs))
    # print(len(atlas['ROIs']))
    # quit()
    # quit()
    # print(f'{list(ROI2vecs)=}')
    for ROI, region_LR in zip(atlas['ROIs'], atlas['ROI_regions_laterality']):
        # print(f'{ROI=} | {region_LR=}')
        if ROI not in ROI2vecs:
            # print(f'{ROI} not in ROI2vecs')
            continue
        # print(f'{ROI} ({region_LR}): {np.nanmean(ROI2vecs[ROI], axis=1)}')
        region2vecs[region_LR].append(np.nanmean(ROI2vecs[ROI], axis=1))
    # print(list(region2vecs))
    # quit()
    for region_LR, vecs in region2vecs.items():
        region2vecs[region_LR] = np.array(vecs).T
    region_order = [(f'{region}_L', f'{region}_R') for
                    region in atlas['tick_labels']]
    # print(f'{region_order=}')
    region_order = [item for sublist in region_order for item in sublist]
    score_ar = np.full((len(region2vecs), len(region2vecs)), np.nan)
    IRAFs_ar = np.full((len(region2vecs), len(region2vecs), 114), np.nan)

    return region2vecs, region_order, score_ar, IRAFs_ar


def mask_img(img, ROIs, blocks=False):
    t_st = time()

    if blocks:
        atlas = get_atlas()
        atlas_data = atlas['maps'].get_fdata()
        # bad_vals = np.unique(img[atlas_data == 0])
        bad_vals, bad_vals_cnts = np.unique(img[atlas_data == 0], return_counts=True)
        max_cnt = np.max(bad_vals_cnts[1])
        bad_vals = [bad_val for bad_val, cnt in
                    zip(bad_vals, bad_vals_cnts) if cnt >= (max_cnt * .8)]
        for bad_val in bad_vals:
            img[(bad_val + 1e-8 > img) & (img > bad_val - 1e-8)] = np.nan
        return img

    atlas_mask = pickle_wrap(get_atlas_mask, RAM_cache=True, easy_override=True)
    img[~atlas_mask] = np.nan  # Needed to basically get only ROIs and no non-ROI
    if ROIs is None:
        print(f'\tTime needed to atlas mask: {time() - t_st=:.2f}')
        return img

    ROIs_mask = pickle_wrap(get_ROI_mask, kwargs={'ROIs': ROIs},
                            RAM_cache=True, easy_override=True)

    img[~ROIs_mask, :] = np.nan
    # print(f'\tTime needed to mask ROIs: {time() - t_st=:.2f}')

    # for i in range(10, 40, 5):
    #     plt.imshow(img[i, :, :, 0])
    #     plt.colorbar()
    #     plt.show()
    # quit()
    return img


def get_atlas_mask():
    atlas = get_atlas()
    atlas_data = atlas['maps'].get_fdata()
    return atlas_data != 0


def get_ROI_mask(ROIs):
    t_st = time()
    atlas = get_atlas()
    atlas_data = atlas['maps'].get_fdata()

    ROIs = set(ROIs)
    atlas_remove = np.zeros(atlas['maps'].shape, dtype=np.bool_)
    for ROI, ROI_num in zip(atlas['ROIs'], atlas['ROI_nums']):
        if ROI in ROIs:
            atlas_remove[atlas_data == ROI_num] = True
    return atlas_remove
