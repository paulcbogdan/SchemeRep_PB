import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as stats

from fMRI_proc import RDM_x_RDM_by_run, RDM_x_RDM, get_IRAFs
from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial
from stim import get_stim_RDM, prune_RSM_outliers
from utils import stdize, pb_outer_euc
from scipy.spatial import distance

def regress_out_edgewise_RSM():
    trials_per_run = 38
    for run0 in range(3):
        for run1 in range(3):
            low0 = run0 * trials_per_run
            high0 = (run0 + 1) * trials_per_run
            low1 = run1 * trials_per_run
            high1 = (run1 + 1) * trials_per_run
            M01 = np.nanmean(RSM_fMRI_by_edge[low0:high0, low1:high1, :],
                             axis=(0, 1))
            RSM_fMRI_by_edge[low0:high0, low1:high1, :]

def RSA_edgewise(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run', combine_regions=False):
    df_sn = get_trial_info(sn)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=False, org_by_region=True,
                                 cross_region=False, conn=conn,
                                 combine_regions=False)

    vecs_BOLD = []
    for ROI, vecs in ROI2vecs.items():
        vecs_BOLD.append(vecs)
    vecs_BOLD = np.concatenate(vecs_BOLD, axis=1)
    n_regions = vecs_BOLD.shape[1]
    vecs = get_conn_vecs(vecs_BOLD, conn=conn)
    vecs = stdize(vecs, axis=1, nans=True)

    trials_per_run = 38
    for run0 in range(3):
        low0 = run0 * trials_per_run
        high0 = (run0 + 1) * trials_per_run
        M0 = np.nanmean(vecs[low0:high0, :], axis=0)
        vecs[low0:high0, :] -= M0
        # SD = np.nanstd(M0)
        # print(f'{M0=}')
        # print(f'{SD=}')
        # M0_sans_trials = (M0 * trials_per_run - vecs[low0:high0, :]) / (trials_per_run - 1)
        # print(M0_sans_trials.shape)
        # print(vecs[low0:high0, 0])
        # print(M0_sans_trials[:, 0])
        # print(np.mean(vecs[low0:high0, 0]))
        # quit()
        # vecs[low0:high0, :] -= M0_sans_trials

    vecs0 = vecs[None, :, :]
    vecs1 = vecs[:, None, :]

    trial_trils = np.tril_indices_from(vecs0[0], k=-1)
    RSM_fMRI_by_edge = -abs(vecs0 - vecs1)
    assert RSM_fMRI_by_edge.shape == (114, 114, 30135)
    # test = np.nanmean(RSM_fMRI_by_edge, axis=2)
    # plt.imshow(test)
    # plt.show()
    # quit()


    RSM_flat_fMRI_by_edge = RSM_fMRI_by_edge[trial_trils[0], trial_trils[1], :]
    assert RSM_flat_fMRI_by_edge.shape == (6441, 30135)
    RSM_flat_fMRI_by_edge = stdize(RSM_flat_fMRI_by_edge, axis=0, nans=True)

    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                            dist=trial_similarity)
    RSM_stim_flat = RSM_stim[trial_trils[0], trial_trils[1]]
    RSM_stim_flat = RSM_stim_flat[:, None]

    assert RSM_stim_flat.shape == (6441, 1)
    RSM_stim_flat = stdize(RSM_stim_flat, axis=0, nans=True)

    score_by_edge_flat = np.nanmean(RSM_flat_fMRI_by_edge * RSM_stim_flat,
                                    axis=0)
    score_by_edge_flat = np.arctanh(score_by_edge_flat)
    score_by_edge = np.full((n_regions, n_regions), np.nan)
    edge_trils = np.tril_indices(n_regions, k=-1)
    score_by_edge[edge_trils[0], edge_trils[1]] = score_by_edge_flat
    score_by_edge[edge_trils[1], edge_trils[0]] = score_by_edge_flat
    return score_by_edge, np.nan, np.nan

    # TODO:

    #
    # plt.imshow(RSA_by_edge)
    # plt.show()
    #
    #
    # quit()
    #
    # # print(RSM_fMRI_by_edge.shape)
    # RSM_fMRI_by_edge = np.transpose(RSM_fMRI_by_edge, (2, 0, 1))
    # # print(RSM_fMRI_by_edge.shape)
    # RSM_flat_fMRI_by_edge = RSM_fMRI_by_edge[:, trils[0], trils[1]]
    # # print(RSM_flat_fMRI_by_edge.shape)
    # # print(RSM_fMRI_by_edge.shape)
    # # print(f'{len(trils)=}')
    # # print(f'{RSM_flat_fMRI_by_edge.shape=}')
    # # print(f'{RSM_stim_flat.shape=}')
    # rs_by_edge = corr_last_dim(RSM_flat_fMRI_by_edge, RSM_stim_flat)
    # # print(rs_by_edge)
    # # quit()
    # # print(f'{rs_by_edge.shape=}')
    # rs_by_edge_ar = np.zeros((n_regions, n_regions))
    # trils_c = np.tril_indices_from(rs_by_edge_ar, k=-1)
    #
    # rs_by_edge_ar[trils_c[0], trils_c[1]] = rs_by_edge
    # rs_by_edge_ar[trils_c[1], trils_c[0]] = rs_by_edge


def RSA_sn(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run', combine_regions=False):
    BOLD = conn == 'BOLD'
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn)
    sess = fp.split('_')[0].replace('2', '').replace('3', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    # TODO: Implement toggle to be high density
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=networks, org_by_region=not BOLD,
                                 cross_region=cross_region, conn=conn,
                                 combine_regions=combine_regions)
    scores = []
    sizes = []
    IRAFs_all_ROI = []
    for ROI, vecs in ROI2vecs.items():
        vecs_BOLD = ROI2vecs[ROI]
        # TODO: Implement toggle to disable connectivity
        keeps = ~np.isnan(vecs_BOLD).any(axis=0)
        if np.sum(keeps) < 2:
            sizes.append(0)
            scores.append(np.nan)
            IRAFs_all_ROI.append(np.full(len(df_sn), np.nan))
            continue
        vecs_BOLD = vecs_BOLD[:, keeps]
        sizes.append(np.sum(keeps))

        vecs_BOLD = stdize(vecs_BOLD, axis=0, nans=True)
        if BOLD or cross_region:
            vecs = vecs_BOLD
        else:
            vecs = get_conn_vecs(vecs_BOLD, conn=conn)

        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                                dist=trial_similarity)
        if RDM_method == 'by_run':
            RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
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
            RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
            z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                          within_to_nan=True)
        else:
            raise ValueError(f'{RDM_method=} not supported')
        scores.append(z)
        IRAFs = get_IRAFs(RSM_fMRI, RSM_stim, df_sn, within_to_nan=False,
                          by_run=RDM_method == 'by_run')
        IRAFs_all_ROI.append(IRAFs)

    scores = np.array(scores)
    return scores, sizes, IRAFs_all_ROI


def get_trial_x_trial_RSM(vecs, simple_mean=False, trial_similarity='corr'):
    RSM = np.zeros((vecs.shape[0], vecs.shape[0]))

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

    n_sn = vecs.shape[0]
    for i in range(vecs.shape[0]):
        run_i = trial2run[i]
        for j in range(vecs.shape[0]):
            run_j = trial2run[j]
            vecs_i = vecs[i].copy()
            vecs_j = vecs[j].copy()
            if i > j:
                continue
            elif i == j:
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
                    M_by_edge_i = (run2M[run_i] * n_sn - vecs_i) / (n_sn - 1)
                    vecs_i -= M_by_edge_i
                    between_run_vecs[i] = vecs_i

                if j in between_run_vecs:
                    vecs_j = between_run_vecs[j]
                else:
                    M_by_edge_j = (run2M[run_j] * n_sn - vecs_j) / (n_sn - 1)
                    vecs_j -= M_by_edge_j
                    between_run_vecs[j] = vecs_j
                has_nan = np.isnan(vecs_i).any() or np.isnan(vecs_j).any()
                if has_nan:
                    nans = np.isnan(vecs_i) | np.isnan(vecs_j)
                    vecs_i = vecs_i[~nans]
                    vecs_j = vecs_j[~nans]
                # r, p = stats.pearsonr(vecs_i, vecs_j)
                r = pdist(vecs_i, vecs_j, trial_similarity, arg)
                RSM[i, j] = r
                RSM[j, i] = r # This was missing as of 11/20/2023 at 4:17 PM
            else:
                if simple_mean:
                    M_by_edge_ij = run2M[run_i]
                else:
                    M_by_edge_ij = (run2M[run_i] * n_sn - vecs_i - vecs_j) / (n_sn - 2)
                vecs_i -= M_by_edge_ij
                vecs_j -= M_by_edge_ij
                has_nan = np.isnan(vecs_i).any() or np.isnan(vecs_j).any()
                if has_nan:
                    nans = np.isnan(vecs_i) | np.isnan(vecs_j)
                    vecs_i = vecs_i[~nans]
                    vecs_j = vecs_j[~nans]
                    if trial_similarity in ['mahalanobis', 'seuclidean', 'euc']:
                        raise ValueError(f'{trial_similarity=} not supported with NaNs')
                r = pdist(vecs_i, vecs_j, trial_similarity, arg)

                RSM[i, j] = r
                RSM[j, i] = r

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
