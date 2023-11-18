import numpy as np
from scipy import stats as stats

from fMRI_proc import RDM_x_RDM_by_run, RDM_x_RDM, get_IRAFs
from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial
from stim import get_stim_RDM
from utils import stdize


def RSA_sn(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run'):
    BOLD = conn == 'BOLD'
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn)
    sess = fp.split('_')[0].replace('2', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    # TODO: Implement toggle to be high density
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=networks, org_by_region=not BOLD,
                                 cross_region=cross_region, conn=conn)
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
        sizes.append(np.sum(keeps))

        vecs_BOLD = stdize(vecs_BOLD, axis=0, nans=True)
        if BOLD or cross_region:
            vecs = vecs_BOLD
        else:
            vecs = get_conn_vecs(vecs_BOLD, conn=conn)
        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        if RDM_method == 'by_run':
            RSM_fMRI = get_trial_x_trial(vecs, trial_similarity=trial_similarity)
            z = RDM_x_RDM_by_run(RSM_fMRI, RSM_stim, corr=second_order)
        elif RDM_method == 'clever_std':
            RSM_fMRI = get_trial_x_trial_RSM(vecs, simple_mean=True)
            z = RDM_x_RDM(RSM_fMRI, RSM_stim, corr=second_order,
                          within_to_nan=False)
        else:
            raise ValueError(f'{RDM_method=} not supported')
        scores.append(z)
        IRAFs = get_IRAFs(RSM_fMRI, RSM_stim, df_sn)
        IRAFs_all_ROI.append(IRAFs)
        # print(f'{z=}')
    scores = np.array(scores)
    return scores, sizes, IRAFs_all_ROI


def get_trial_x_trial_RSM(vecs, simple_mean=False):
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
                r, p = stats.pearsonr(vecs_i, vecs_j)
                RSM[i, j] = r
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
                r, p = stats.pearsonr(vecs_i, vecs_j)
                RSM[i, j] = r
                RSM[j, i] = r
    # plt.imshow(RSM)
    # plt.colorbar()
    # plt.show()
    # quit()
    return RSM
