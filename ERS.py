from tqdm import tqdm

from atlas_utils import get_BN_and_resample, get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_all_sns, get_trial_info
from plot_gen import plot_connectivity
from single_trial_conn import corr_last_dim, corr_matrix_last_two_dim
import numpy as np
import scipy.stats as stats

from stim import get_semantic_vectors, get_DNN_vecs, get_stim_RDM
from utils import stdize, pb_outer
import matplotlib.pyplot as plt
from time import time

def cluster_regions(ROI2vecs):
    networks = {
                # 'Occipital': ['EVC', 'LOC', 'sOcG'],
                # 'Ventral': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG'],
                # 'Dorsal': ['SPL', 'IPL', 'Pcun', 'pSTS'],
                # 'dPFC': ['IFG', 'MFG', 'SFG'],
                # 'PFC_Occ': ['IFG', 'MFG', 'SFG', 'EVC', 'LOC', 'sOcG'],
                }
    # networks['Perception'] = networks['Occipital'] + \
    #                          networks['Ventral'] + \
    #                          networks['Dorsal']
    networks['all'] = list(ROI2vecs.keys())
    ROI2vecs_new = {}
    for network, ROIs in networks.items():
        vecs_l = [ROI2vecs[ROI] for ROI in ROIs]
        vecs = np.concatenate(vecs_l, axis=1)
        ROI2vecs_new[network] = vecs
    keys = list(networks)
    return ROI2vecs_new, keys


def RSA_sn_fp(sn, atlas, d_vecs, fp, org_by_region=True, debug=False,
              networks=True):
    df_sn = get_trial_info(sn)
    # Org by region overrides combine bilateral?
    ROI2vecs = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                            drop_nan_voxels=False, org_by_region=True,
                            easy_override=False)

    n_regions = len(atlas['ROIs'])
    if networks:
        ROI2vecs, keys = cluster_regions(ROI2vecs)
    elif org_by_region:
        keys = atlas['tick_labels']
    else:
        keys = atlas['ROIs']
    RSA_l = []
    sizes = []
    rs_by_edge_ar = None
    for ROI in keys:
        if debug and ('LOC' not in ROI):
            RSA_l.append(np.nan)
            sizes.append(np.nan)
            continue
        vecs = ROI2vecs[ROI]
        keeps = ~np.isnan(vecs).any(axis=0)
        sizes.append(np.sum(keeps))
        # vecs = vecs[:, keeps] # TODO: double check this doesn't break stuff
        vecs = stdize(vecs, axis=0, nans=True)
        vecs = pb_outer(vecs, vecs, tril=True, nan_diag=True)
        vecs = stdize(vecs, axis=1, nans=True)

        vecs0 = vecs[None, :, :]
        vecs1 = vecs[:, None, :]
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1) # could be a Pearson

        trils = np.tril_indices_from(RSM_fMRI, k=-1)
        RSM_fMRI_flat = RSM_fMRI[trils]
        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        RSM_stim_flat = RSM_stim[trils]
        r, p = stats.spearmanr(RSM_fMRI_flat, RSM_stim_flat)
        RSA_l.append(r)

        if ROI == 'all':

            RSM_fMRI_by_edge = -abs(vecs0 - vecs1)
            # print(RSM_fMRI_by_edge.shape)
            RSM_fMRI_by_edge = np.transpose(RSM_fMRI_by_edge, (2, 0, 1))
            # print(RSM_fMRI_by_edge.shape)
            RSM_flat_fMRI_by_edge = RSM_fMRI_by_edge[:, trils[0], trils[1]]
            # print(RSM_flat_fMRI_by_edge.shape)
            # print(RSM_fMRI_by_edge.shape)
            # print(f'{len(trils)=}')
            # print(f'{RSM_flat_fMRI_by_edge.shape=}')
            # print(f'{RSM_stim_flat.shape=}')
            rs_by_edge = corr_last_dim(RSM_flat_fMRI_by_edge, RSM_stim_flat)
            # print(rs_by_edge)
            # quit()
            # print(f'{rs_by_edge.shape=}')
            rs_by_edge_ar = np.zeros((n_regions, n_regions))
            trils_c = np.tril_indices_from(rs_by_edge_ar, k=-1)

            rs_by_edge_ar[trils_c[0], trils_c[1]] = rs_by_edge
            rs_by_edge_ar[trils_c[1], trils_c[0]] = rs_by_edge

    RSA_l = np.array(RSA_l)

    # TODO: look at cross region? e.g., SFG x LOC
    return RSA_l, keys, sizes, rs_by_edge_ar




def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           org_by_region=True):
    df_sn = get_trial_info(sn)
    ROI2vecs_enc = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False, org_by_region=True,
                                easy_override=False)
    ROI2vecs_ret = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False, org_by_region=True,
                                easy_override=False)
    ers_l = []
    # hits = df_sn['hit_bool'].fillna(False).values
    # print(atlas['tick_labels'])
    # quit()
    if org_by_region:
        keys = atlas['tick_labels']
    else:
        keys = atlas['ROIs']
    # print(f'{atlas["ROIs"]}')
    # quit()
    sizes = []
    for ROI in keys:
        # if 'LOC' not in ROI:
        #     ers_l.append(0)
        #     continue
        # if 'LOC' not in ROI and 'EVC' not in ROI:
        #     continue
        # print(ROI)
        # quit()
        try:
            vecs_enc = ROI2vecs_enc[ROI]
            # print(f'{vecs_enc.shape=}')
            # quit()
            vecs_ret = ROI2vecs_ret[ROI]
        except KeyError:
            ers_l.append(np.nan)
            # print('BAD')
            continue
        if vecs_enc.shape != vecs_ret.shape:
            ers_l.append(np.nan)
            print(f'ROI {ROI} has different shapes for enc ({vecs_enc.shape}) '
                  f'and ret ({vecs_ret.shape})')
            continue
        keeps = np.logical_and(~np.isnan(vecs_enc).any(axis=0),
                               ~np.isnan(vecs_ret).any(axis=0))
        sizes.append(keeps.sum())
        vecs_enc = vecs_enc[:, keeps]
        vecs_ret = vecs_ret[:, keeps]

        vecs_enc = stdize(vecs_enc, axis=0)
        vecs_ret = stdize(vecs_ret, axis=0)
        vecs_enc = pb_outer(vecs_enc, vecs_enc, tril=True, nan_diag=True)
        vecs_ret = pb_outer(vecs_ret, vecs_ret, tril=True, nan_diag=True)

        # stdizing along axis=1 here leads to all edges within a given trial
        #   having mean = 0. This makes it more similar to a Pearson correlation
        #   but isn't obviously necessary
        vecs_enc = stdize(vecs_enc, axis=1)
        vecs_ret = stdize(vecs_ret, axis=1)
        ers = np.nanmean(-abs(vecs_enc - vecs_ret), axis=1)

        vecs_enc_ = vecs_enc[:, None, :]
        vecs_ret_ = vecs_ret[None, :, :]
        ers_else = np.nanmean(-abs(vecs_enc_ - vecs_ret_), axis=2)
        ers_else[np.diag_indices_from(ers_else)] = 0
        ers_else = np.sum(ers_else, axis=1) / (ers_else.shape[1] - 1)

        ers_dif = ers - ers_else
        ers_l.append(np.nanmean(ers_dif))

    ers_sn = np.array(ers_l)

    return ers_sn, keys, sizes


def ERS_all_sn(RSA=True, semantic=False):
    if RSA:
        if semantic:
            d_vecs = get_semantic_vectors()
        else:
            d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    else:
        d_vecs = None

    atlas = get_atlas(combine_regions=False, combine_bilateral=False,
                      split=False, split_code='z')
    age2sn = get_all_sns(ret=True)
    ers_l_all = []
    sns = age2sn[1]
    print(f'{len(sns)=}')
    # fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI']
    RSA_str = 'RSA (semantic)' if (RSA and semantic) else \
              'RSA (perceptual)' if (RSA and not semantic) else \
              'ERS'
    print(f'{fps=}')
    rs_by_edge_ar_l = []

    for i, sn in tqdm(enumerate(sns[::-1]), desc='ERS, looping subjects'):
        ers_sn_by_comparison = []
        keys = None
        sizes = None
        if RSA:
            for fp0 in fps:
                ers_sn, keys, sizes, rs_by_edge_ar = \
                    RSA_sn_fp(sn, atlas, d_vecs, fp0)
                ers_sn_by_comparison.append(ers_sn)
                rs_by_edge_ar_l.append(rs_by_edge_ar)
        else:
            for fp0 in fps:
                for fp1 in fps:
                    if fp0 >= fp1:
                        continue
                    ers_sn, keys, sizes = ERS_sn(sn, atlas, fp0, fp1)
                    ers_sn_by_comparison.append(ers_sn)
        ers_sn_by_comparison = np.array(ers_sn_by_comparison)
        ers_sn = np.nanmean(ers_sn_by_comparison, axis=0)
        # ers_sn, keys = ERS_sn(sn, atlas)
        ers_l_all.append(ers_sn)
        ers_all = np.array(ers_l_all)
        rs_by_edge_ar_all = np.array(rs_by_edge_ar_l)
        print()
        if i < 2:
            continue
        for j, ROI in enumerate(keys):
            ers = ers_all[:, j]
            M = np.nanmean(ers)
            SD = np.nanstd(ers)
            N = len(ers[~np.isnan(ers)])
            SE = SD / np.sqrt(N)
            t = M / SE
            p = stats.t.sf(np.abs(t), N-1)
            print(f'{RSA_str} | {ROI} ({sizes[j]}), t[{N-1}]={t:.2f}, p={p:.3f}')

        M_mat = np.nanmean(rs_by_edge_ar_all, axis=0)
        SD_mat = np.nanstd(rs_by_edge_ar_all, axis=0)
        N_mat = np.sum(~np.isnan(rs_by_edge_ar_all), axis=0)
        SE_mat = SD_mat / np.sqrt(N_mat)
        t_mat = M_mat / SE_mat
        plot_connectivity(t_mat, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'],
                          vmin=-3, vmax=3, no_avg=True)



if __name__ == '__main__':
    ERS_all_sn()