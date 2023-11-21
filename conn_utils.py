import pickle

import matplotlib.pyplot as plt
import numpy as np

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from old.plot_gen import plot_connectivity
from stim import get_stim_RDM, get_DNN_vecs, scipy_dist, prune_RSM_outliers
from utils import get_RSA_fn, tril_flat, stdize, pb_outer_euc, pb_outer
import scipy.stats as stats

import warnings
from time import time

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')


def roimap2np(ROI2ar, only_some=None):
    l = []
    for ROI, ar in ROI2ar.items():
        if only_some:
            for ROI_ in only_some:
                if ROI_ in ROI:
                    break
            else:
                continue
        l.append(ar)
    # print(np.array(l).shape)
    l = np.array(l)
    print(f'Data shape: {l.shape}')
    return l

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


def do_single_trial_conn(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=False, rxr=False):

    # Binary connectivity

    fp = get_RSA_fn(inc=cin, age=age, semantic=semantic, DNN_layer=2,
                    fp_fMRI_col='obj_fMRI',
                    bilateral=bilateral, combine_regions=combine_regions,
                    vec_prod=vec_prod, org_by_region=org_by_region,
                    rxr=rxr)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROI_focus = ['EVC', 'LOC', 'sOcG']
    # ROI_focus = ['IFG', 'MFG']
    obj_a = roimap2np(d['activity'], only_some=ROI_focus)

    fp2 = get_RSA_fn(inc=cin, age=age, semantic=semantic, DNN_layer=2,
                     fp_fMRI_col='scn_fMRI',
                     bilateral=bilateral, combine_regions=combine_regions,
                     vec_prod=vec_prod, org_by_region=org_by_region,
                     rxr=rxr)
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)
    scn_a = roimap2np(d2['activity'], only_some=ROI_focus)

    both_a = np.stack([obj_a, scn_a], axis=-1)
    both_a = both_a.transpose((1, 2, 0, 3))
    # print(both_a.shape)
    # quit()
    _, rs_flat = corr_matrix_last_two_dim(both_a)
    # print(rs_flat.shape)
    # quit()
    _, RDMs_flat = corr_matrix_last_two_dim(rs_flat)

    df_sn = get_trial_info('102')
    df_sn.sort_values(by='obj', inplace=True)
    # d_vecs = get_semantic_vectors()
    d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True)
    RDM_stim = get_stim_RDM(df_sn, d_vecs, add=True, take_abs=False)
    # RDM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
    # RDM_stim = get_stim_RDM(df_sn, d_vecs, dif=True, take_abs=True)


    print(f'stim: {np.sum(np.isnan(RDM_stim))}')
    RDM_stim_flat = RDM_stim[np.tril_indices(RDM_stim.shape[0], k=-1)]

    r2nd_order = corr_last_dim(RDMs_flat, RDM_stim_flat)
    r2nd_order = np.arctanh(r2nd_order)
    M_second_order = np.nanmean(r2nd_order, axis=0)
    SD_second_order = np.nanstd(r2nd_order, axis=0)
    SE_second_order = SD_second_order / np.sqrt(r2nd_order.shape[0])
    t = M_second_order / SE_second_order
    p = 2*(1 - stats.t.cdf(np.abs(t), r2nd_order.shape[0] - 1))
    print(f'{r2nd_order=}')
    print(f'{M_second_order=:.3f}')
    print(f'{SD_second_order=:.3f}')
    print(f'{t=:.3f}, {p=:.3f}')
    # print(np.argwhere(np.isnan(r2nd_order)))
    quit()


    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      combine_bilateral=bilateral or org_by_region)
    title = 'sleepy'
    plot_connectivity(np.mean(rs, axis=0),
                      atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=title, no_avg=True,
                      cbar_label='t-value')




def get_conn_vecs(vecs, vecs1=None, conn='euc'):
    vecs = stdize(vecs, axis=0, nans=True)
    if vecs1 is None:
        vecs1 = vecs
        cross = False
    else:
        # TODO: redo all cross, there may be an error
        vecs = stdize(vecs, axis=0, nans=True)
        cross = True
    if conn == 'euc':
        vecs = pb_outer_euc(vecs, vecs1, tril=not cross,
                            flat=cross, nan_diag=not cross)
    elif conn == 'prod':
        vecs = pb_outer(vecs, vecs1,  tril=not cross,
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
    for network, ROIs in networks.items():
        vecs_l = []
        for ROI in ROIs:
            if ROI in ROI2vecs:
                vecs_l.append(ROI2vecs[ROI])
            elif f'{ROI}_L' in ROI2vecs:
                vecs_l.append(ROI2vecs[f'{ROI}_L'])
                vecs_l.append(ROI2vecs[f'{ROI}_R'])
            else:
                raise ValueError(f'ROI={ROI} not found')
        vecs = np.concatenate(vecs_l, axis=1)
        ROI2vecs_new[network] = vecs
    return ROI2vecs_new


def get_ROI_vecs_wrap(sn, atlas, fp0, df_sn, fp1=None, networks=None,
                      org_by_region=True, cross_region=False,
                      conn=None, combine_regions=False):
    assert not (combine_regions and org_by_region)
    ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                             drop_nan_voxels=False,
                             org_by_region=org_by_region,
                             easy_override=False,
                             combine_regions=combine_regions)
    if fp1 is not None:
        ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                 drop_nan_voxels=False,
                                 org_by_region=org_by_region,
                                 easy_override=False,
                                 combine_regions=combine_regions)
    else:
        ROI2vecs1 = None

    if networks:
        if cross_region:
            ROI2vecs0 = calculate_cross_region_vecs(ROI2vecs0, networks,
                                                    conn=conn)
        else:
            ROI2vecs0 = cluster_regions(ROI2vecs0, networks)

        if fp1 is not None:
            if cross_region:
                ROI2vecs1 = calculate_cross_region_vecs(ROI2vecs1, networks,
                                                        conn=conn)
            else:
                ROI2vecs1 = cluster_regions(ROI2vecs1, networks)

    if fp1 is not None:
        return ROI2vecs0, ROI2vecs1
    else:
        return ROI2vecs0


def get_trial_x_trial(vecs, vecs1=None, trial_similarity='corr'):
    # Fixed bug (Nov 21, where this was stdizing for euc & seuc)
    vecs0 = vecs[None, :, :]
    if vecs1 is not None:
        has_vecs1 = True
        vecs1 = vecs1[:, None, :]
    else:
        has_vecs1 = False
        vecs1 = vecs[:, None, :]
    if trial_similarity == 'corr':
        vecs0 = stdize(vecs0, axis=2, nans=True)
        vecs1 = stdize(vecs1, axis=2, nans=True)
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=-1)  # Pearson
    elif trial_similarity == 'euc':
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Euclidean
    elif trial_similarity == 'spear':
        # t = time()
        vecs0_r = stats.rankdata(vecs0, method='average', axis=-1)
        vecs1_r = stats.rankdata(vecs1, method='average', axis=-1)
        vecs0_r = stdize(vecs0_r, axis=2, nans=True)
        vecs1_r = stdize(vecs1_r, axis=2, nans=True)

        RSM_fMRI = np.nanmean(vecs0_r * vecs1_r, axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # plt.imshow(RSM_fMRI)
        # plt.title('Fast')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        # RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # # prune_RSM_outliers(RSM_fMRI, z=3.5)
        #
        # plt.imshow(RSM_fMRI)
        # plt.title('euc')
        # plt.colorbar()
        # plt.show()
        #
        # RSM_fMRI = np.nanmean(-abs(vecs0_r - vecs1_r), axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # plt.imshow(RSM_fMRI)
        # plt.title('ranked euc')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        #
        # RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
        #                       metric='seuclidean')
        # # prune_RSM_outliers(RSM_fMRI, z=3.5)
        # plt.imshow(RSM_fMRI)
        # plt.title('std euc')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        #
        #
        # print(f'{has_vecs1=}')
        #
        # vecs0, vecs1 = np.squeeze(vecs0), np.squeeze(vecs1)
        # RSM_fMRI = np.zeros((vecs0.shape[0], vecs1.shape[0]))
        # for i in range(vecs0.shape[0]):
        #     for j in range(vecs1.shape[0]):
        #         if i >= j and not has_vecs1:
        #             continue
        #         r, p = stats.spearmanr(vecs0[i], vecs1[j])
        #         z = np.arctanh(r)
        #         RSM_fMRI[j, i] = z
        #         if not has_vecs1:
        #             RSM_fMRI[j, i] = z
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        #
        # plt.imshow(RSM_fMRI)
        # plt.colorbar()
        # plt.title('Slow spearman')
        # plt.show()
        # quit()
        # RSM_fMRI[np.diag_indices_from(RSM_fMRI)] = np.nan
    elif trial_similarity == 'seuclidean':
        RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
                              metric='seuclidean')
        RSM_fMRI = prune_RSM_outliers(RSM_fMRI)
    else:
        raise ValueError(f'{trial_similarity=} not supported')
    return RSM_fMRI
