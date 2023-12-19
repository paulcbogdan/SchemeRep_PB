import numpy as np
from scipy import stats as stats

from fMRI_proc import RDM_x_RDM_by_run, RDM_x_RDM, get_IRAFs
from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial, prep_for_pairwise, get_BNA_ROIs, \
    prep_for_ROI_analysis, get_mean_conn_trialwise
from stim import get_stim_RDM, prune_RSM_outliers
from utils import stdize
from scipy.spatial import distance


def RSA_ROI(sn, atlas, d_vecs, fp, networks=True, conn='euc',
            trial_similarity='corr', second_order='spear',
            RDM_method='by_run', combine_regions=False,
            PFC=False, PFC2=False, ROI_ctrl=False):
    df_sn = get_trial_info(sn)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=False, org_by_region=False,
                                 cross_region=False, conn=conn,
                                 combine_regions=False)
    ROI2vecs = prep_for_ROI_analysis(ROI2vecs, PFC, PFC2)

    if len(ROI2vecs) == 997:
        ROIs_l = get_BNA_ROIs(code='schaefer')
    elif len(ROI2vecs) == 189:
        ROIs_l = get_BNA_ROIs(code='PFC_schaef')
    elif len(ROI2vecs) > 400:
        ROIs_l = get_BNA_ROIs(code='PFC_8')
    elif len(ROI2vecs) == 60:
        ROIs_l = get_BNA_ROIs(code='PFC_ACC')
    elif len(ROI2vecs) == 52:
        ROIs_l = get_BNA_ROIs(code='PFC')
    elif len(ROI2vecs) == 246:
        ROIs_l = get_BNA_ROIs(code=None)
    else:
        raise ValueError(f'Bad get_BNA_ROIs code: {len(ROI2vecs)=}')

    ROIs_l = [ROI for ROI in ROIs_l if ROI in ROI2vecs.keys()]


    if ROI_ctrl:
        mean_conn_trialwise = get_mean_conn_trialwise(ROIs_l, ROI2vecs, conn)

    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                            dist=trial_similarity)

    scores = []
    sizes = []
    scores_trialwise = []
    for ROI0 in ROIs_l:
        vecs0 = ROI2vecs[ROI0]
        vecs0 = vecs0[:, None]
        vecs_else = []
        for ROI1 in ROIs_l:
            if ROI1 == ROI0:
                vecs_else.append(np.full(vecs0.shape[0], np.nan))
                continue
            vecs1 = ROI2vecs[ROI1]
            vecs_else.append(vecs1)
        vecs_else = np.vstack(vecs_else).T
        sizes.append(vecs_else.shape[1])
        vecs = get_conn_vecs(vecs0, vecs_else, conn=conn)
        # print(f'{vecs.shape=}')
        # quit()
        if ROI_ctrl:# or (not PFC):
            vecs -= mean_conn_trialwise.T# * 246 - vecs) / 245
        # print(f'{vecs.shape=}')
        # quit()
        z, IRAFs = get_RSM_subtract_run_mean_fast(vecs, RSM_stim, df_sn,
                                                  trial_similarity=trial_similarity,
                                                  second_order=second_order)
        # print(IRAFs)
        # quit()
        scores.append(z)
        # print(f'{z=:.3f}')
        scores_trialwise.append(IRAFs)

    # quit()
    return scores, sizes, scores_trialwise


def RSA_ROI_pairwise(sn, atlas, d_vecs, fp, networks=True, conn='euc',
                     trial_similarity='corr', second_order='spear',
                     RDM_method='by_run', combine_regions=False):
    # print(f'{second_order=}')
    # quit()
    df_sn = get_trial_info(sn)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=False, org_by_region=False,
                                 cross_region=False, conn=conn,
                                 combine_regions=False)
    key0 = list(ROI2vecs)[2]
    # print(ROI2vecs[key0])
    # print(f'{key0=}')
    # quit()

    ROI2vecs, region_order, score_ar, IRAFs_ar = \
        prep_for_pairwise(ROI2vecs, atlas)
    # print(list(ROI2vecs))
    # print(ROI2vecs)
    # quit()

    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                            dist=trial_similarity)
    # ROI_pair2conn_vecs = {}
    # score_ar = np.full((len(ROI2vecs), len(ROI2vecs)), np.nan)
    # IRAFs_ar = np.full((len(ROI2vecs), len(ROI2vecs), 114), np.nan)
    # print(ROI2vecs)
    for i, ROI0 in enumerate(region_order):

        vecs0 = ROI2vecs[ROI0]
        # print(ROI0, vecs0)
        # quit()
        # print(f'{np.array(vecs0).shape=}')

        for j, ROI1 in enumerate(region_order):
    # for i, (ROI0, vecs0) in enumerate(ROI2vecs.items()):
    #     for j, (ROI1, vecs1) in enumerate(ROI2vecs.items()):
            if ROI1 > ROI0:
                continue
            elif ROI0 == ROI1:
                vecs = get_conn_vecs(vecs0, conn=conn)
            else:
                vecs1 = ROI2vecs[ROI1]
                # print(f'{ROI1} | {np.array(vecs1).shape=}')
                vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
            if vecs.shape[1] == 1:
                score_ar[i, j] = np.nan
                score_ar[j, i] = np.nan
                IRAFs_ar[i, j, :] = np.full((114), np.nan)
                IRAFs_ar[j, i, :] = np.full((114), np.nan)
                continue
            # if ROI1 == 'PCL_L' and ROI0 == 'PCL_L':
                # print(vecs.shape)
                # quit()
            # else:
            #     continue
                # print(f'{vecs=}')
                # quit()
            # print(f'{ROI1} | {ROI0}')
            z, IRAFs = get_RSM_subtract_run_mean_fast(vecs, RSM_stim, df_sn,
                                    trial_similarity=trial_similarity,
                                                      second_order=second_order)
            score_ar[i, j] = z
            score_ar[j, i] = z
            IRAFs_ar[i, j, :] = IRAFs
            IRAFs_ar[j, i, :] = IRAFs
                # quit()
            # print(f'{ROI0} | {ROI1}: {z=:.3f}')
            # ROI_pair2conn_vecs[(ROI0, ROI1)] = z
            # ROI_pair2conn_vecs[(ROI1, ROI0)] = z
    return score_ar, np.nan, IRAFs_ar




def get_RSM_subtract_run_mean_fast(vecs, RSM_stim, df_sn,
                                   trial_similarity='corr',
                                   second_order='spear'):
    trials_per_run = 38
    for run in range(3):
        low = run * trials_per_run
        high = (run + 1) * trials_per_run
        M = np.nanmean(vecs[low:high, :], axis=0)
        vecs[low:high, :] -= M

    if trial_similarity == 'corr':
        vecs = stdize(vecs, axis=1, nans=True)
        vecs0 = vecs[None, :, :]
        vecs1 = vecs[:, None, :]
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=2)
    elif trial_similarity == 'spear':
        vecs_r = stats.rankdata(vecs, axis=1, nan_policy='omit')
        vecs0_r = vecs_r[None, :, :]
        vecs1_r = vecs_r[:, None, :]
        vecs0_r = stdize(vecs0_r, axis=2, nans=True)
        vecs1_r = stdize(vecs1_r, axis=2, nans=True)
        RSM_fMRI = np.nanmean(vecs0_r * vecs1_r, axis=2)  # Spearman
    else:
        raise NotImplementedError(f'{trial_similarity=}')
    # print(vecs0 * vecs1)

    # try:
    z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                  within_to_nan=False)
    # IRAFs = np.full((114), np.nan)
    IRAFs = get_IRAFs(RSM_fMRI, RSM_stim, df_sn,
                      within_to_nan=False,
                      second_order=second_order)
    # except:
    #     print(f'{RSM_fMRI=}')
    #     quit()
    return z, IRAFs


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

    df_sn = get_trial_info(sn, easy_override=True)
    sess = fp.split('_')[0].replace('2', '').replace('3', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True) # added to help with runw-wise sorting
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
                    # if do_vecs1:
                    # print(f'dif: {n_sn=}')
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
                        # print(f'dif: {n_sn=}')
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
                    # print(f'do vecs1: {n_sn=}')
                    # n_sn = n_sn / 2
                else:
                    n_sn = run2non_nans[run_i]

                    # print(f'-- : {n_sn=}')
                # print(f'same: {n_sn=}')
                # if do_vecs1:
                #     n_sn = run2non_nans[run_i]
                #
                # print(f'{vecs_j=}')
                # print(f'{vecs_i=}')

                if simple_mean:
                    M_by_edge_ij = run2M[run_i]
                else:
                    M_by_edge_ij = (run2M[run_i] * n_sn - vecs_i - vecs_j) / (n_sn - 2)
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
            # print(f'{i} | {j} | {r=}')
    # quit()
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
