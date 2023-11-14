from tqdm import tqdm

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs, RDM_x_RDM_by_run, RDM_x_RDM, get_IRAFs
from organize_bhv import get_all_sns, get_trial_info
from plot_gen import plot_connectivity
import numpy as np
import scipy.stats as stats

from stim import get_semantic_vectors, get_DNN_vecs, get_stim_RDM
from utils import stdize, pb_outer, pb_outer_euc, pickle_wrap


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
                            org_by_region=True,
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
    IRAFs_all_ROI = []
    for ROI, vecs in ROI2vecs.items():
        vecs_BOLD = ROI2vecs[ROI]
        keeps = ~np.isnan(vecs_BOLD).any(axis=0)
        sizes.append(np.sum(keeps))
        vecs_BOLD = stdize(vecs_BOLD, axis=0, nans=True)
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

    scores = np.array(scores)

    return scores, sizes, IRAFs_all_ROI


def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc'):
    df_sn = get_trial_info(sn)
    ROI2vecs_enc, ROI2vecs_ret = get_ROI_vecs_wrap(sn, atlas, fp0, df_sn,
                                                   fp1=fp1, networks=networks)
    scores = []
    sizes = []
    scores_by_trial = []
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
        ERS_ar_ = ERS_ar.copy()
        ERS_ar_[~np.eye(len(ERS_ar), dtype=bool)] = np.nan
        ERS_elses = np.nanmean(ERS_ar_, axis=1)

        ERS_dif = ERS_sames - ERS_elses
        score = np.nanmean(ERS_dif)
        scores.append(score)
        scores_by_trial.append(ERS_dif)
    scores = np.array(scores)
    return scores, sizes, scores_by_trial

def run_sn(fps, RSA, sn, atlas, d_vecs, networks=None,
           conn='euc', trial_similarity='euc', second_order='spear',
           RDM_method='by_run'):
    scores_all = []
    sizes_all = []
    scores_by_ROI = []
    for fp0 in fps:
        if RSA:
            scores, sizes, score_by_ROI = \
                RSA_sn(sn, atlas, d_vecs, fp0, networks=networks,
                       conn=conn, trial_similarity=trial_similarity,
                       second_order=second_order, RDM_method=RDM_method)
            scores_all.append(scores)
            sizes_all.append(sizes)
            scores_by_ROI.append(score_by_ROI)
        else:
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                scores, sizes, score_by_ROI = \
                    ERS_sn(sn, atlas, fp0, fp1, networks=networks,
                           conn=conn, trial_similarity=trial_similarity)
                scores_all.append(scores)
                sizes_all.append(sizes)
                scores_by_ROI.append(score_by_ROI)
    M_score_by_ROI = np.nanmean(scores_all, axis=0)
    M_size_by_ROI = np.nanmean(sizes_all, axis=0)
    scores_by_ROI = np.array(scores_by_ROI)
    return M_score_by_ROI, M_size_by_ROI, scores_by_ROI

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
               'sns': sns, 'scores': [], 'sizes': [],
               'scores_by_ROI': []}
    results['settings'] = settings
    for i, sn in tqdm(enumerate(sns), desc='ERS, looping subjects'):
        ers_sn_by_comparison = []
        scores, sizes, scores_by_ROI = \
            run_sn(fps, RSA, sn, atlas, d_vecs, networks=networks,
                               conn=conn, trial_similarity=trial_similarity,
                               second_order=second_order, RDM_method=RDM_method)
        results['scores'].append(scores)
        results['sizes'].append(sizes)
        results['scores_by_ROI'].append(scores_by_ROI)
        if verbose and i > 1:
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
                 four_tasks=False, combine_regions=True, split=False,
                 RDM_method='clever_std'):
    settings = locals().copy()
    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks
    assert RSA or second_order == 'spear', 'Leave second_order as \"spear\" for ERS'
    dir_results = r'cache/conn_RSA'
    results = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                          cache_dir=dir_results, easy_override=False)
    report_results(results)

def run_analysis_toggles():
    RSA = False
    semantic = False
    do_networks = True
    RDM_method='clever_std'

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
                                     do_networks=do_networks,
                                     RDM_method=RDM_method)
                    except AssertionError:
                        pass

if __name__ == '__main__':
    # run_analysis_toggles()
    run_analysis(RSA=False, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='corr', second_order='spear',
                 four_tasks=False, combine_regions=False, split=True,
                 RDM_method='clever_std')