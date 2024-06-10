from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as stats

from fMRI_proc import RDM_x_RDM_by_run, RDM_x_RDM, get_IRAFs
from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial
from stim import get_stim_RDM, prune_RSM_outliers
from utils import stdize
from scipy.spatial import distance
from time import time
import matplotlib.pyplot as plt

def RSA_sn(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run', combine_regions=False,
           stdize_by_run=False, semantic=False):
    BOLD = 'BOLD' in conn
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True) # added to help with run-wise sorting
    # TODO: Implement toggle to be high density

    # print(f'{BOLD=}')
    # print(f'{networks=}')
    # print(f'{combine_regions=}')
    # org_by_region = (not BOLD) or (networks and not combine_regions)
    # org_by_region = (not BOLD) and (networks or not combine_regions)
    org_by_region = (not BOLD) or (networks and not combine_regions)



    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=networks, org_by_region=org_by_region,
                                 cross_region=cross_region, conn=conn,
                                 combine_regions=combine_regions,
                                 easy_override=False)
    # for ROI, vecs in ROI2vecs.items():
    #     if ROI != 'MTL2': continue
    #     print(ROI, ':', vecs.shape)
    # quit()

    scores = []
    sizes = []
    IRAFs_all_ROI = []
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                            dist=trial_similarity)
    print('RSA_sn')
    if 'norm' in conn and 'avg' in conn:
        networks = {'MTL': ['Hipp', 'PhG', 'ATL'],
                    'Occipital': ['EVC', 'LOC', 'sOcG'],
                    'PFC': ['SFG', 'MFG', 'IFG', 'OrG', ]
                    }
        networks2l = defaultdict(list)
        i2network = {}

        for i, (ROI, vecs) in enumerate(ROI2vecs.items()):
            for network, keys in networks.items():
                for key in keys:
                    if key in ROI:
                        networks2l[network].append(ROI)
                        assert i not in i2network, \
                            f'Mixed: {i=}, {i2network[i]=}'
                        i2network[i] = network

        network2M = {}
        network2SD = {}
        for network, ROIs in networks2l.items():
            vecs = np.vstack([np.nanmean(ROI2vecs[ROI], axis=-1)
                              for ROI in ROIs])
            # ROI2vecs[network] = vecs
            # print(vecs.shape)
            network2M[network] = np.nanmean(vecs, axis=0)[:, None]
            network2SD[network] = np.nanstd(vecs, axis=0)[:, None]
        # quit()

    RSM_fmri_l = []
    dir_out = fr'cache/conn_RSA/ars/RSA'
    dir_out = (f'{dir_out}/{fp}_{trial_similarity}_'
               f'{second_order}_{RDM_method}_{stdize_by_run}')
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    for i, (ROI, vecs) in enumerate(ROI2vecs.items()):
        # print('A')
        t_st = time()
        vecs_BOLD = ROI2vecs[ROI]
        # TODO: Implement toggle to disable connectivity
        # if sn == '132' and ('obj' in fp or 'scn' in fp): # corrupted run 3
        #     keeps = ~np.isnan(vecs_BOLD[:76]).any(axis=0)
        # else:
        #     keeps = ~np.isnan(vecs_BOLD).any(axis=0)
        # print(vecs_BOLD.shape)
        # print(keeps.shape)
        keeps = np.isnan(vecs_BOLD).sum(axis=0) < 39 # keep sns missing 1 run

        if np.sum(keeps) < 2:
            sizes.append(0)
            scores.append(np.nan)
            IRAFs_all_ROI.append(np.full(len(df_sn), np.nan))
            print(f'No keeps ({ROI}): {sn}')
            continue

        vecs_BOLD = vecs_BOLD[:, keeps]
        sizes.append(np.sum(keeps))

        vecs_BOLD = stdize(vecs_BOLD, axis=0, nans=True)
        if BOLD or cross_region:
            vecs = stdize(vecs_BOLD, axis=0, nans=True,
                          stdize_by_run=stdize_by_run)
        else:
            vecs = get_conn_vecs(vecs_BOLD, conn=conn,
                                 stdize_by_run=stdize_by_run)


        if 'avg' in conn:
            vecs = np.nanmean(vecs, axis=-1)[..., None]

        if '_norm' in conn:
            if 'avg' in conn:
                if i in i2network:
                    network = i2network[i]

                    if 'half' in conn:
                        vecs = vecs - (network2M[network] / 2)
                    else:
                        vecs = (vecs - network2M[network])# / network2SD[network]

            else:
                vecs = stdize(vecs, axis=1, nans=True)

        if RDM_method == 'by_run':
            RSM_fMRI = get_trial_x_trial(vecs,
                                         trial_similarity=trial_similarity)
            z = RDM_x_RDM_by_run(RSM_fMRI, RSM_stim, corr=second_order)
        elif RDM_method == 'clever_std':
            RSM_fMRI = get_trial_x_trial_RSM(vecs, simple_mean=True,
                                             trial_similarity=trial_similarity)
            z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                          within_to_nan=False)
        elif RDM_method == 'clever_std_complex_mean':
            RSM_fMRI = get_trial_x_trial_RSM(vecs, simple_mean=False,
                                             trial_similarity=trial_similarity)
            z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                          within_to_nan=False)
        elif RDM_method == 'within_nan':
            RSM_fMRI = get_trial_x_trial(vecs,
                                         trial_similarity=trial_similarity)


            z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                          within_to_nan=True)
        else:
            raise ValueError(f'{RDM_method=} not supported')

        scores.append(z)

        RSM_fmri_l.append(RSM_fMRI)
        IRAFs = get_IRAFs(RSM_fMRI, RSM_stim, df_sn,
                          within_to_nan=RDM_method == 'within_nan',
                          by_run=RDM_method == 'by_run')
        IRAFs_all_ROI.append(IRAFs)

        cmb = '_cmb' if (combine_regions and BOLD) else ''
        fn_RSM = f'{sn}_{ROI}_{conn}{cmb}.npy'
        fp_RSM = f'{dir_out}/{fn_RSM}'
        print(f'{fp_RSM=}')


        with open(fp_RSM, 'wb') as f:
            np.save(f, RSM_fMRI)

    fn_RSM = f'{sn}_stim_{semantic}.npy'
    fp_RSM = f'{dir_out}/{fn_RSM}'
    with open(fp_RSM, 'wb') as f:
        np.save(f, RSM_stim)

    scores = np.array(scores)
    return scores, sizes, IRAFs_all_ROI


def get_trial_x_trial_RSM(vecs, vecs1=None,
                          simple_mean=False, trial_similarity='corr'):
    do_vecs1 = vecs1 is not None
    # RSM = np.zeros((vecs.shape[0], vecs.shape[0]))

    RSM = np.full((vecs.shape[0], vecs.shape[0]), np.nan)

    trial_per_run = vecs.shape[0] // 3
    trial2run = {}
    run2M = {}
    run2non_nans = {}
    run2non_nans0 = {}
    run2non_nans1 = {}
    # print(f'{vecs.shape=}')
    # quit()
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        vecs_run = vecs[low:high]
        # print(vecs_run.shape)
        if do_vecs1:
            run2non_nans0[run] = np.sum(~np.isnan(vecs_run), axis=0)
            M_run0 = np.nanmean(vecs_run, axis=0)
            vecs_run1 = vecs1[low:high]
            M_run1 = np.nanmean(vecs_run1, axis=0)

            # vecs_run = np.concatenate((vecs_run, vecs_run1), axis=0)
            # print(vecs_run1.shape)
            # print(vecs_run.shape)
            # quit()
            n_non_nans1 = np.sum(~np.isnan(vecs_run1), axis=0)
            run2non_nans1[run] = n_non_nans1
            run2M[run] = (M_run0 + M_run1) / 2
        else:
            n_non_nans = np.sum(~np.isnan(vecs_run), axis=0)
            # assert np.all(n_non_nans == n_non_nans[0]), f'{n_non_nans=}'
            # n_non_nans = n_non_nans[0]
            # print(f'{run=} {n_non_nans=}')
            M_by_edge = np.nanmean(vecs_run, axis=0)
            run2M[run] = M_by_edge
            run2non_nans[run] = n_non_nans

        for trial in range(trial_per_run):
            trial2run[trial + low] = run

    between_run_vecs = {}
    if trial_similarity == 'mahalanobis' and vecs.shape[1] > vecs.shape[0]:
        raise ValueError(f'Too many features '
                         f'({vecs.shape[1]} > {vecs.shape[0]}) '
                         f'for Mahalanobis distance')
    elif trial_similarity == 'mahalanobis':
        V = np.cov(vecs.T)
        IV = np.linalg.inv(V)
        arg = IV
    elif trial_similarity == 'seuclidean':
        V_by_edge = np.nanvar(vecs, axis=0)
        arg = V_by_edge
    else:
        arg = None


    # cov = np.cov(vecs.T)
    # IV = np.linalg.inv(cov)
    # print(f'{cov.shape=}')
    # print(f'{IV.shape=}')
    # quit()

    # print(f'{vecs[4:, :]=}')
    # quit()
    n_sn = vecs.shape[0]
    # print(vecs.shape)
    # quit()
    for i in range(vecs.shape[0]):
        run_i = trial2run[i]
        for j in range(vecs.shape[0]):
            # if j != 0: continue
            run_j = trial2run[j]
            vecs_i = vecs[i].copy()
            # print(f'{vecs_i=}')
            if do_vecs1:
                vecs_j = vecs1[j].copy()
            else:
                vecs_j = vecs[j].copy()

            if i > j and not do_vecs1:
                continue
            elif i == j and not do_vecs1:
                RSM[i, j] = np.nan
                continue
            elif run_i != run_j:
                # if simple_mean:
                #     M_by_edge_i = run2M[run_i]
                #     M_by_edge_j = run2M[run_j]
                #     if i in between_run_vecs:
                #         vecs_i = between_run_vecs[i]
                #     else:
                #         vecs_i -= M_by_edge_i
                #         between_run_vecs[i] = vecs_i
                #     if j in between_run_vecs:
                #         vecs_j = between_run_vecs[j]
                #     else:
                #         vecs_j -= M_by_edge_j
                #         between_run_vecs[j] = vecs_j
                # else:
                if i in between_run_vecs:
                    vecs_i = between_run_vecs[i]
                else:

                    if do_vecs1:
                        n_sn = run2non_nans0[run_i]
                        M_by_edge_i = (run2M[run_i] * n_sn * 2 - vecs_i) / (2 * n_sn - 1)
                    else:
                        n_sn = run2non_nans[run_i]
                        M_by_edge_i = (run2M[run_i] * n_sn - vecs_i) / (n_sn - 1)
                    # M_by_edge_i = (run2M[run_i] * n_sn - vecs_i*0.5) / (n_sn - 0.5)

                    vecs_i -= M_by_edge_i
                    between_run_vecs[i] = vecs_i

                if j in between_run_vecs and not do_vecs1:
                    vecs_j = between_run_vecs[j]
                else:
                    if do_vecs1:
                        n_sn = run2non_nans0[run_i]
                        M_by_edge_j = (run2M[run_j] * n_sn * 2 - vecs_j) / (2 * n_sn - 1)
                    else:
                        n_sn = run2non_nans[run_j]
                        M_by_edge_j = (run2M[run_j] * n_sn - vecs_j) / (n_sn - 1)
                    # M_by_edge_j = (run2M[run_j] * n_sn - vecs_j*0.5) / (n_sn - 0.5)

                    vecs_j -= M_by_edge_j
                    between_run_vecs[j] = vecs_j
                has_nan = np.isnan(vecs_i).any() or np.isnan(vecs_j).any()
                if has_nan:
                    nans = np.isnan(vecs_i) | np.isnan(vecs_j)
                    vecs_i = vecs_i[~nans]
                    vecs_j = vecs_j[~nans]
                if len(vecs_i):
                    # r, p = stats.pearsonr(vecs_i, vecs_j)
                    r = pdist(vecs_i, vecs_j, trial_similarity, arg)
                    RSM[i, j] = r
                    if not do_vecs1:
                        RSM[j, i] = r # This was missing as of 11/20/2023 at 4:17 PM
                else:
                    r = np.nan
                    RSM[i, j] = np.nan
                    if not do_vecs1:
                        RSM[j, i] = np.nan
            else:
                if do_vecs1:
                    n_sn = run2non_nans0[run_i] + run2non_nans1[run_i]

                else:
                    n_sn = run2non_nans[run_i]

                if simple_mean:
                    M_by_edge_ij = run2M[run_i]
                else:
                    M_by_edge_ij = ((run2M[run_i] * n_sn - vecs_i - vecs_j) /
                                    (n_sn - 2))
                    # M_by_edge_ij = (run2M[run_i] * n_sn - (vecs_i - vecs_j)*0.5) / (n_sn - 1)

                vecs_i -= M_by_edge_ij
                vecs_j -= M_by_edge_ij
                has_nan = np.isnan(vecs_i).any() or np.isnan(vecs_j).any()
                if has_nan:
                    nans = np.isnan(vecs_i) | np.isnan(vecs_j)
                    vecs_i = vecs_i[~nans]
                    vecs_j = vecs_j[~nans]
                    if trial_similarity in ['mahalanobis', 'seuclidean', 'euc']:
                        raise ValueError(f'{trial_similarity=} not supported with NaNs')
                if len(vecs_i):
                    try:
                        r = pdist(vecs_i, vecs_j, trial_similarity, arg)
                    except ValueError:
                        print('ValueError RSM pdist')
                        r = np.nan
                    RSM[i, j] = r
                    if not do_vecs1:
                        RSM[j, i] = r
                else:
                    r = np.nan
                    RSM[i, j] = np.nan
                    if not do_vecs1:
                        RSM[j, i] = np.nan

    RSM[np.diag_indices_from(RSM)] = np.nan

    if trial_similarity == 'mahalanobis' or trial_similarity == 'seuclidean':
        RSM = prune_RSM_outliers(RSM)

    return RSM

def pdist(vec_i, vec_j, measure, arg=None):
    if measure == 'mahalanobis':
        r = -distance.mahalanobis(vec_i, vec_j, arg)
    elif measure == 'seuclidean':
        r = -distance.seuclidean(vec_i, vec_j, arg)
        # r = -distance.euclidean(vec_i, vec_j)
    elif measure == 'euc':
        r = -distance.euclidean(vec_i, vec_j)
    elif measure == 'spear':
        r, p = stats.spearmanr(vec_i, vec_j)
        r = np.arctanh(r)
    elif measure == 'corr':
        r, p = stats.pearsonr(vec_i, vec_j)
        r = np.arctanh(r)
    else:
        raise ValueError(f'First order {measure=} not supported')
    return r
