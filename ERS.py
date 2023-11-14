from tqdm import tqdm

from atlas_utils import get_BN_and_resample, get_atlas
from fMRI_proc import get_ROI_vecs, RDM_x_RDM, within_run_to_nan, \
    regress_out_within_across, RDM_x_RDM_by_run
from organize_bhv import get_all_sns, get_trial_info
from plot_gen import plot_connectivity
from single_trial_conn import corr_last_dim, corr_matrix_last_two_dim
import numpy as np
import scipy.stats as stats

from stim import get_semantic_vectors, get_DNN_vecs, get_stim_RDM
from utils import stdize, pb_outer, pb_outer_euc, get_default_fp, pickle_wrap
import matplotlib.pyplot as plt
from time import time
import random

def get_conn_vecs(vecs, conn='euc'):
    vecs = stdize(vecs, axis=0, nans=True)
    if conn == 'euc':
        vecs = pb_outer_euc(vecs, vecs, tril=True, nan_diag=True)
    elif conn == 'corr':
        vecs = pb_outer(vecs, vecs, tril=True, nan_diag=True)
    else:
        raise ValueError(f'conn={conn} not recognized')
    return vecs


def cluster_regions(ROI2vecs, networks):
    ROI2vecs_new = {}
    for network, ROIs in networks.items():
        vecs_l = [ROI2vecs[ROI] for ROI in ROIs]
        vecs = np.concatenate(vecs_l, axis=1)
        ROI2vecs_new[network] = vecs
    return ROI2vecs_new

def get_ROI_vecs_wrap(sn, atlas, fp0, df_sn, fp1=None, networks=None):
    ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                            drop_nan_voxels=False, org_by_region=True,
                            easy_override=True)
    if fp1 is not None:
        ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False, org_by_region=True,
                                easy_override=True)
    else:
        ROI2vecs1 = None

    if networks:
        ROI2vecs0 = cluster_regions(ROI2vecs0, networks)
        if fp1 is not None:
            ROI2vecs1 = cluster_regions(ROI2vecs1, networks)

    if fp1 is not None:
        return ROI2vecs0, ROI2vecs1
    else:
        return ROI2vecs0

def get_trial_x_trial(vecs, vecs1=None, trial_similarity='corr'):
    if vecs1:
        vecs1 = stdize(vecs1, axis=1, nans=True)  # Is this needed?
    vecs = stdize(vecs, axis=1, nans=True) # Is this needed?
    vecs0 = vecs[None, :, :]
    if vecs1:
        vecs1 = vecs1[:, None, :]
    else:
        vecs1 = vecs[:, None, :]
    if trial_similarity == 'corr':
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=-1)  # Pearson
    elif trial_similarity == 'euc':
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Euclidean
    else:
        raise ValueError(f'{trial_similarity=} not supported')
    return RSM_fMRI


def RSA_sn(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear'):
    df_sn = get_trial_info(sn)
    sess = fp.split('_')[0].replace('2', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=networks)
    scores = []
    scores_alt = []
    sizes = []
    for ROI, vecs in ROI2vecs.items():
        vecs = ROI2vecs[ROI]
        keeps = ~np.isnan(vecs).any(axis=0)
        sizes.append(np.sum(keeps))
        vecs = stdize(vecs, axis=0, nans=True)
        vecs = get_conn_vecs(vecs, conn=conn)
        RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        # z = RDM_x_RDM_by_run(RSM_fMRI, RSM_stim, corr='spear')

        # RSM_stim_flat = RSM_stim[trils]

        z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr='spear')
        scores_alt.append(z)
        # print(f'\n({ROI}) RDM x RDM: {fp} | {z=:.3f}')

        z = RDM_x_RDM_by_run(RSM_fMRI, RSM_stim, corr='spear')
        # print(f'\t({ROI}) By run: {fp} | {z=:.3f}')
        # r, p = stats.spearmanr(RSM_fMRI_flat, RSM_stim_flat, nan_policy='omit')
        scores.append(z)
        continue

        # trial_per_run = RSM_stim.shape[0] // 3
        # for run0 in range(3):
        #     for run1 in range(3):
        #         if run1 < run0:
        #             continue
        #         low0 = run0 * trial_per_run
        #         high0 = (run0 + 1) * trial_per_run
        #         low1 = run1 * trial_per_run
        #         high1 = (run1 + 1) * trial_per_run
        #         M = np.nanmean(RSM_stim[low0:high0, low1:high1])
        #         SD = np.nanstd(RSM_stim[low0:high0, low1:high1])
        #         print(f'{fp} | {run0}, {run1} | {M=:.3f} ({SD:.3f})')


        # fMRI_l = []
        # stim_l = []
        # trial_per_run = RSM_stim.shape[0] // 3
        # for run0 in range(3):
        #     for run1 in range(3):
        #         if run1 <= run0:
        #             continue
        #         low0 = run0 * trial_per_run
        #         high0 = (run0 + 1) * trial_per_run
        #         low1 = run1 * trial_per_run
        #         high1 = (run1 + 1) * trial_per_run
        #         fMRI_data = RSM_fMRI[low0:high0, low1:high1].flatten()
        #         fMRI_l.extend(fMRI_data)
        #         stim_data = RSM_stim[low0:high0, low1:high1].flatten()
        #         stim_l.extend(stim_data)
        #
        #         # M = np.nanmean(RSM_stim[low0:high0, low1:high1])
        #         # SD = np.nanstd(RSM_stim[low0:high0, low1:high1])
        #         # print(f'{fp} | {run0}, {run1} | {M=:.3f} ({SD:.3f})')
        #
        # RSM_fMRI_flat = np.array(fMRI_l)
        # RSM_stim_flat = np.array(stim_l)


        # print(f'{fMRI_l.shape}')
        # quit()

        # RSM_stim[np.diag_indices_from(RSM_stim)] = np.nan
        # RSM_stim = within_run_to_nan(RSM_stim)

        # RSM_zeros = within_run_to_nan(RSM_fMRI)
        # RSM_zeros_flat = RSM_zeros[trils].astype(bool)
        # plt.imshow(RSM_fMRI)
        # plt.title(f'{fp=}')
        # plt.scatter(RSM_stim_flat[RSM_zeros_flat], RSM_fMRI_flat[RSM_zeros_flat],
        #             color='r', label='diagonal', alpha=.4)
        # plt.scatter(RSM_stim_flat[~RSM_zeros_flat], RSM_fMRI_flat[~RSM_zeros_flat],
        #             color='b', label='else', alpha=.4)
        # plt.legend()
        # plt.show()
        # # quit()


        # plt.imshow(RSM_stim)
        # plt.colorbar()
        # plt.show()
        # quit()

        # nans = np.isnan(RSM_fMRI_flat)
        # withins = RS

        # plt.scatter(RSM_fMRI_flat, RSM_stim_flat)
        # plt.show()

        # r, p = stats.spearmanr(RSM_fMRI_flat, RSM_stim_flat, nan_policy='omit')
        # z = np.arctanh(r)
        # print(f'{fp} | {z=:.3f} ({p:.3f})')
        # scores.append(r)
        continue


        # if ROI != 'LOC':
        #     scores.append(np.nan)
        #     sizes.append(np.nan)
        #     continue
        keeps = ~np.isnan(vecs).any(axis=0)
        sizes.append(np.sum(keeps))
        # print(f'{vecs.shape=}')
        vecs = get_conn_vecs(vecs, conn=conn)
        # print(f'{trial_similarity=}')
        RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
        # print(RSM_fMRI.shape)
        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        score = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order)
        # print(f'{ROI} | {score=:.3f}')
        # print(f'{ROI}, {score=:.3f}')
        scores.append(score)
    scores = np.array(scores)
    scores_alt = np.array(scores_alt)
    scores_bigger = np.nanmean(scores > scores_alt) - 0.5
    print(f'{scores_bigger=:.3f}')

    return scores, sizes


def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc'):
    df_sn = get_trial_info(sn)
    ROI2vecs_enc, ROI2vecs_ret = get_ROI_vecs_wrap(sn, atlas, fp0, df_sn,
                                                   fp1=fp1, networks=networks)
    scores = []
    sizes = []
    for ROI in ROI2vecs_enc.keys():
        try:
            vecs_enc = ROI2vecs_enc[ROI]
            vecs_ret = ROI2vecs_ret[ROI]
        except KeyError:
            scores.append(np.nan)
            continue
        assert vecs_enc.shape == vecs_ret.shape
        keeps = np.logical_and(~np.isnan(vecs_enc).any(axis=0),
                               ~np.isnan(vecs_ret).any(axis=0))
        sizes.append(np.sum(keeps))
        vecs_enc = get_conn_vecs(vecs_enc, conn=conn)
        vecs_ret = get_conn_vecs(vecs_ret, conn=conn)

        ERS_ar = get_trial_x_trial(vecs_enc, vecs_ret, trial_similarity=trial_similarity)
        ERS_sames = np.diag(ERS_ar)
        M_same = np.nanmean(ERS_sames)
        M_else = np.nanmean(ERS_ar[~np.eye(len(ERS_ar), dtype=bool)])
        score = M_same - M_else
        scores.append(score)
    scores = np.array(scores)
    return scores, sizes

def run_sn(fps, RSA, sn, atlas, d_vecs, networks=None,
           conn='euc', trial_similarity='euc', second_order='spear'):
    scores_all = []
    sizes_all = []
    for fp0 in fps:
        if RSA:
            scores, sizes = \
                RSA_sn(sn, atlas, d_vecs, fp0, networks=networks,
                       conn=conn, trial_similarity=trial_similarity,
                       second_order=second_order)
            scores_all.append(scores)
            sizes_all.append(sizes)
        else:
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                scores, sizes = \
                    ERS_sn(sn, atlas, fp0, fp1, networks=networks,
                           conn=conn, trial_similarity=trial_similarity)
                scores_all.append(scores)
                sizes_all.append(sizes)
    M_score_by_ROI = np.nanmean(scores_all, axis=0)
    M_size_by_ROI = np.nanmean(sizes_all, axis=0)
    return M_score_by_ROI, M_size_by_ROI

def prep_vecs(RSA, semantic):
    if RSA:
        if semantic:
            d_vecs = get_semantic_vectors()
        else:
            d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    else:
        d_vecs = None
    return d_vecs

def prep_fps(four_tasks):
    if four_tasks:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    else:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI']
    return fps

def prep_networks():
    networks = {
        'Occipital': ['EVC', 'LOC', 'sOcG'],
        'Ventral': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG'],
        'Dorsal': ['SPL', 'IPL', 'Pcun', 'pSTS'],
        'dPFC': ['IFG', 'MFG', 'SFG'],
        'PFC_Occ': ['IFG', 'MFG', 'SFG', 'EVC', 'LOC', 'sOcG'],
        'FPCN': ['IFG', 'MFG', 'SFG', 'SPL', 'IPL', 'pSTS']
    }
    return networks

def run_settings(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, verbose=1):
    settings = locals().copy()
    # d_vecs = prep_vecs(RSA, semantic)
    # vecs_l = list(d_vecs.values())
    # random.shuffle(vecs_l)
    # d_vecs = dict(zip(d_vecs.keys(), vecs_l))

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False,
                      split=split, split_code='xyz')
    fps = prep_fps(four_tasks)
    age2sn = get_all_sns(ret=True)
    sns = age2sn[1]

    if do_networks:
        networks = prep_networks()
        keys = list(networks)
    else:
        networks = None
        keys = atlas['tick_labels']

    results = {'networks': networks, 'keys': keys,
               'sns': sns, 'scores': [], 'sizes': []}
    results['settings'] = settings
    for i, sn in tqdm(enumerate(sns), desc='ERS, looping subjects'):
        ers_sn_by_comparison = []
        scores, sizes = run_sn(fps, RSA, sn, atlas, d_vecs, networks=networks,
                               conn=conn, trial_similarity=trial_similarity,
                               second_order=second_order)
        results['scores'].append(scores)
        results['sizes'].append(sizes)
        if verbose and i > 1:
            # print('Shuffled')
            report_results(results)
    return results

def print_settings(settings):
    print(f'{settings=}')

def report_results(results):
    print_settings(results['settings'])
    scores = np.array(results['scores'])
    sizes = np.array(results['sizes'])
    for j, ROI in enumerate(results['keys']):
        ROI_scores = scores[:, j]
        M = np.nanmean(ROI_scores)
        SD = np.nanstd(ROI_scores)
        N = len(ROI_scores[~np.isnan(ROI_scores)])
        SE = SD / np.sqrt(N)
        t = M / SE
        p = stats.t.sf(np.abs(t), N - 1)

        M_size = np.nanmean(sizes[:, j])
        print(f'{ROI} ({M_size:.1f}), t[{N - 1}]={t:.2f}, p={p:.3f}')


def plot_edgewise(rs_by_edge_ar_all, atlas, prt_all):
    M_mat = np.nanmean(rs_by_edge_ar_all, axis=0)
    SD_mat = np.nanstd(rs_by_edge_ar_all, axis=0)
    N_mat = np.sum(~np.isnan(rs_by_edge_ar_all), axis=0)
    SE_mat = SD_mat / np.sqrt(N_mat)
    t_mat = M_mat / SE_mat
    plot_connectivity(t_mat, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      vmin=-4, vmax=4, no_avg=True,
                      title=prt_all, cbar_label='t-value')

def run_analysis(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='corr', second_order='spear',
                 four_tasks=False, combine_regions=False, split=False):
    settings = locals().copy()
    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks
    dir_results = r'cache/conn_RSA'
    results = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                          cache_dir=dir_results, easy_override=True)
    report_results(results)


if __name__ == '__main__':
    run_analysis()