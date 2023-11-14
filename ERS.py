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
from collections import defaultdict

def get_conn_vecs(vecs, conn='euc'):
    vecs = stdize(vecs, axis=0, nans=True)
    if conn == 'euc':
        vecs = pb_outer_euc(vecs, vecs, tril=True, nan_diag=True)
    elif conn == 'prod':
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
                            drop_nan_voxels=False,
                            org_by_region=False,
                            easy_override=True)
    if fp1 is not None:
        ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False,
                                 org_by_region=True,
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
    vecs = stdize(vecs, axis=1, nans=True) # Is this needed?
    vecs0 = vecs[None, :, :]
    if vecs1 is not None:
        vecs1 = stdize(vecs1, axis=1, nans=True)  # Is this needed?
        vecs1 = vecs1[:, None, :]
    else:
        vecs1 = vecs[:, None, :]
    if trial_similarity == 'corr':
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=-1)  # Pearson
    elif trial_similarity == 'euc':
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Euclidean
    # elif trial_similarity == 'spear':
    #     pass
    else:
        raise ValueError(f'{trial_similarity=} not supported')
    return RSM_fMRI

def get_trial_x_trial_RSM(vecs, simple_mean=False):
    RSM = np.zeros((vecs.shape[0], vecs.shape[0]))
    # M_by_edge = np.nanmean(vecs, axis=0)
    # print(M_by_edge.shape)
    # quit()

    trial_per_run = vecs.shape[0] // 3
    trial2run = {}
    run2M = {}
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        vecs_run = vecs[low:high]
        M_by_edge = np.nanmean(vecs_run, axis=0)
        run2M[run] = M_by_edge
        for trial in range(trial_per_run):
            trial2run[trial + low] = run

    n_sn = vecs.shape[0]
    for i in range(vecs.shape[0]):
        run_i = trial2run[i]
        for j in range(vecs.shape[0]):
            run_j = trial2run[j]
            vecs_i = vecs[i].copy()
            vecs_j = vecs[j].copy()
            if i == j:
                RSM[i, j] = np.nan
                continue
            elif run_i != run_j:
                if simple_mean:
                    M_by_edge_i = run2M[run_i]
                    M_by_edge_j = run2M[run_j]
                else:
                    M_by_edge_i = (run2M[run_i] * n_sn - vecs_i) / (n_sn - 1)
                    M_by_edge_j = (run2M[run_j] * n_sn - vecs_j) / (n_sn - 1)
                    # M_by_edge_i = (run2M[run_i] * n_sn - vecs_i * 0.5) / (n_sn - 0.5)
                    # M_by_edge_j = (run2M[run_j] * n_sn - vecs_j * 0.5) / (n_sn - 0.5)
                vecs_i -= M_by_edge_i
                vecs_j -= M_by_edge_j
                r, p = stats.pearsonr(vecs_i, vecs_j)
                RSM[i, j] = r
            else:
                if simple_mean:
                    M_by_edge_ij = run2M[run_i]
                else:
                    M_by_edge_ij = (run2M[run_i] * n_sn - vecs_i - vecs_j) / (n_sn - 2)
                    # M_by_edge_ij = (run2M[run_i] * n_sn - vecs_i*0.5 - vecs_j*0.5) / (n_sn - 1)
                vecs_i -= M_by_edge_ij
                vecs_j -= M_by_edge_ij
                r, p = stats.pearsonr(vecs_i, vecs_j)
                RSM[i, j] = r
    # plt.imshow(RSM)
    # plt.colorbar()
    # plt.show()
    # quit()
    return RSM


def conn_autocorrelation(combine_regions=False, split=False, four_tasks=False,
                         conn='euc'):

    # this suggests that to measure the similarity between trials 0 and 1, you
    #   should standardize the two by the mean and std calculated solely with
    #   trials 3-38.

    # Autocorrelation is very small, like .08
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False,
                      split=split, split_code='xyz')
    fps = prep_fps(four_tasks)

    fp = fps[0]
    sess = fp.split('_')[0].replace('2', '')
    age2sn = get_all_sns(ret=True)
    sns = age2sn[1]
    corr_by_sn = []
    corr_by_ROI = defaultdict(list)
    for i, sn in tqdm(enumerate(sns), desc='ERS, looping subjects'):
        df_sn = get_trial_info(sn)
        df_sn.sort_values(by=f'{sess}_trial', inplace=True)
        # for idx, row in df_sn.iterrows():
        #     print(row[fp])
        # print(df_sn[fp])
        # quit()
        ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                     networks=False)
        sn_corr = []
        for i, (ROI, vecs) in enumerate(ROI2vecs.items()):
            if i % 10 != 0: continue
            vecs = ROI2vecs[ROI]

            keeps = ~np.isnan(vecs).any(axis=0)
            vecs = vecs[:, keeps]
            if vecs.shape[-1] < 10: continue
            # print(f'{vecs.shape=}')
            # quit()
            # print(vecs.shape)
            # quit()
            # vecs = stdize(vecs, axis=1, nans=True)
            # print(f'{vecs.shape=}')
            # vecs = stdize(vecs, axis=1, nans=True)
            # vecs = get_conn_vecs(vecs, conn=conn)
            # ROI_corr = []

            # print(vecs.shape)
            # quit()
            # vecs = np.vstack([vecs[:10], vecs[38:48], vecs[76:86]])
            trial_per_run = vecs.shape[0] // 3

            # print(vecs.shape)
            # quit()

            # for run in range(3):
            #     low = run * trial_per_run
            #     high = (run + 1) * trial_per_run
            #     # vecs_run = vecs[low:high, :]
            #     # vecs_run = stdize(vecs_run, axis=0, nans=True)
            #     vecs[low:high, :] = stdize(vecs[low:high, :], axis=0, nans=True)

                # for trial in range(trial_per_run):
                #     if trial == trial_per_run - 1: continue
                #     data0 = vecs_run[trial, :]
                #     data1 = vecs_run[trial + 1, :]
                #     r_order = stats.pearsonr(data0, data1)[0]
                #
                #     alts = []
                #     for trial_alt in range(trial_per_run):
                #         if trial_alt == trial: continue
                #         data1 = vecs_run[trial_alt, :]
                #         r = stats.pearsonr(data0, data1)[0]
                #         alts.append(r)
                #
                #     r_dif = r_order - np.mean(alts)
                #     ROI_corr.append(r_order)
                    # sn_corr.append(r_dif)

            # RDM_fMRI = np.corrcoef(vecs)
            RDM_fMRI = get_trial_x_trial_RSM(vecs)


            # print(RDM_fMRI.shape)
            # print(vecs.shape)
            # quit()

            RDM_stim = np.zeros_like(RDM_fMRI)
            for run0 in range(3):
                for run1 in range(3):
                    if run0 != run1: continue
                    low0 = run0 * trial_per_run
                    high0 = (run0 + 1) * trial_per_run
                    low1 = run1 * trial_per_run
                    high1 = (run1 + 1) * trial_per_run
                    for i in range(trial_per_run):
                        for j in range(trial_per_run):
                            RDM_stim[low0 + i, low1 + j] = 38 - abs(i - j)


            # plt.imshow(RDM_stim)
            # plt.show()
            trils = np.tril_indices_from(RDM_fMRI, k=-1)
            RDM_fMRI_flat = RDM_fMRI[trils]
            RDM_stim_flat = RDM_stim[trils]
            r, p = stats.spearmanr(RDM_fMRI_flat, RDM_stim_flat)
            corr_by_ROI[ROI].append(r)
            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()
            # plt.imshow(RDM_stim)
            # plt.show()
            # z = RDM_x_RDM_by_run(RDM_fMRI, RDM_stim, corr='spear')
            # corr_by_ROI[ROI].append(z)
            # quit()
            # print(f'{r=}')
            # quit()



            # quit()

            # n_edges = vecs.shape[1]
            # for j in range(n_edges):
            #     rs = []
            #     for run in range(3):
            #         low = run * trial_per_run
            #         high = (run + 1) * trial_per_run
            #         time_series = vecs[low:high, j]
            #         # print(f'{vecs.shape=}, {run=},{time_series.shape=}')
            #         time_series0 = time_series[:-1]
            #         time_series1 = time_series[1:]
            #         # plt.scatter(range(len(time_series0)), time_series0)
            #         # plt.show()
            #         r = stats.pearsonr(time_series0, time_series1)[0]
            #         rs.append(r)
            #     sn_corr.append(np.mean(rs))
            #     ROI_corr.append(np.mean(rs))
            # corr_by_ROI[ROI].append(np.mean(ROI_corr))

        for ROI, l in corr_by_ROI.items():
            print(f'{ROI}, {np.mean(l)=:.3f} ({np.std(l)=:.3f})')


        sn_corr = np.array(sn_corr)
        # plt.title(f'{sn=}')
        # plt.hist(sn_corr)
        # corr_sn_region_val.append(corr_by_sn)
        # print(f'{corr_by_sn.shape=}')
        # plt.imshow(corr_by_sn)
        # plt.colorbar()
        # plt.show()
        continue
        M = np.mean(autocorrelation_all_sn)
        SD = np.std(autocorrelation_all_sn)
        print(f'N = {i + 1} ({len(autocorrelation_all_sn)}): '
              f'{M=:.5f} ({SD=:.5f})')
        #     print(f'{ROI}, {autocorrelation=:.3f}')
        # quit()


def RSA_sn(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run'):
    df_sn = get_trial_info(sn)
    sess = fp.split('_')[0].replace('2', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=networks)
    scores = []
    sizes = []
    for ROI, vecs in ROI2vecs.items():
        vecs = ROI2vecs[ROI]
        keeps = ~np.isnan(vecs).any(axis=0)
        sizes.append(np.sum(keeps))
        vecs = stdize(vecs, axis=0, nans=True)
        vecs = get_conn_vecs(vecs, conn=conn)
        RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        if RDM_method == 'by_run':
            z = RDM_x_RDM_by_run(RSM_fMRI, RSM_stim, corr=second_order)
        else:
            raise ValueError(f'{RDM_method=} not supported')
        scores.append(z)
        continue

    scores = np.array(scores)

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
           conn='euc', trial_similarity='euc', second_order='spear',
           RDM_method='by_run'):
    scores_all = []
    sizes_all = []
    for fp0 in fps:
        if RSA:
            scores, sizes = \
                RSA_sn(sn, atlas, d_vecs, fp0, networks=networks,
                       conn=conn, trial_similarity=trial_similarity,
                       second_order=second_order, RDM_method=RDM_method)
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
                 combine_regions=False, split=False, RDM_method='by_run',
                 verbose=1):
    settings = locals().copy()
    d_vecs = prep_vecs(RSA, semantic)
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
                               second_order=second_order, RDM_method=RDM_method)
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

def run_analysis(RSA=False, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='corr', second_order='spear',
                 four_tasks=False, combine_regions=True, split=False,
                 RDM_method='by_run'):
    settings = locals().copy()
    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks
    assert RSA or second_order == 'spear', 'Leave second_order as \"spear\" for ERS'
    dir_results = r'cache/conn_RSA'
    results = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                          cache_dir=dir_results, easy_override=True)
    report_results(results)

def run_analysis_toggles():
    RSA = False
    semantic = False
    do_networks = True

    conn_toggle = ['euc', 'prod']
    trial_similarity_toggle = ['euc', 'corr']
    four_tasks_toggle = [False, True]
    split_toggle = [False, True]
    # combine_regions_toggle = [True]

    for four_tasks in four_tasks_toggle:
        for conn in conn_toggle:
            for trial_similarity in trial_similarity_toggle:
                for split in split_toggle:
                    try:
                        run_analysis(conn=conn,
                                     trial_similarity=trial_similarity,
                                     four_tasks=four_tasks, split=split,
                                     RSA=RSA, semantic=semantic,
                                     do_networks=do_networks)
                    except AssertionError:
                        pass

if __name__ == '__main__':
    conn_autocorrelation()
    # run_analysis_toggles()