import numpy as np
import pandas as pd
from scipy import stats as stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.conn_report import report_results
from connRSA.conn_utils import get_ROI_vecs_wrap, prep_for_ROI_analysis, get_BNA_ROIs, get_mean_conn_trialwise, \
    get_conn_vecs, prep_for_pairwise
from fMRI_proc import RDM_x_RDM, get_IRAFs

from old.plot_gen import plot_connectivity, my_plot_surf
from organize_bhv import get_trial_info
from stim import get_stim_RDM
from utils import get_default_fp, pickle_wrap, make_title_str, stdize


def get_lmer_matrix(results):
    print(f'{results["scores_by_ROI"].shape=}')
    n_regions = results['scores_by_ROI'].shape[2]
    print(f'{n_regions=}')
    lmer_ar = np.full((n_regions, n_regions), np.nan)
    for i in tqdm(range(n_regions), desc='running lmers'):
        for j in range(n_regions):
            if i > j:
                continue
            else:
                pair_scores = results['scores_by_ROI'][:, :, i, j, :]
                # print(f'{i} | {j} ')
                pair_scores_raveled = pair_scores.ravel()
                # print(f'{pair_scores_raveled.shape=}')
                df = pd.DataFrame({'IRAF': pair_scores_raveled})
                # print(f'{pair_scores.shape=}')
                idxs = np.ndindex(pair_scores.shape)
                idxs = np.array(list(idxs))

                df[['sn', 'fp', 'stim']] = idxs
                for key in ['sn', 'fp', 'stim']:
                    df[key] = df[key].astype(str)
                # print(df)
                df.dropna(inplace=True)
                if len(df) < 10000:
                    print(f'{i}, {j} | many na drops {len(df)=}')
                    lmer_ar[i, j] = np.nan
                    lmer_ar[j, i] = np.nan
                    continue

                # print(df)
                from pymer4.models import Lmer
                # st = time()
                formula = 'IRAF ~ 1 + (1|sn) + (1|fp)'
                model = Lmer(formula, data=df)
                # model.fit()
                # print(model.summary())

                model.fit(REML=True, verbose=False, summary=False)
                summary = model.coefs
                lmer_t = summary['T-stat'].loc['(Intercept)']
                lmer_ar[i, j] = lmer_t
                lmer_ar[j, i] = lmer_t
    return lmer_ar


def visualize_region_matrix(results, plot_lmer=False):
    M_all = np.nanmean(results['scores'], axis=0)
    SD_all = np.nanstd(results['scores'], axis=0)
    N_all = np.sum(~np.isnan(results['scores']), axis=0)
    N = np.max(N_all)
    SE_all = SD_all / np.sqrt(N_all)
    t_all = M_all / SE_all

    # ticks = results['ticks']
    # tick_labels = results['tick_labels']
    # tick_lows = results['tick_lows']

    # print(f'{ticks=}')
    # print(f'{tick_lows=}')

    conn_str = f'conn={results["settings"]["conn"]}'
    second_order_str = f'second_order={results["settings"]["second_order"]}'
    trial_similarity_str = f'trial_similarity={results["settings"]["trial_similarity"]}'
    analysis_str = f'RSA={results["settings"]["RSA"]}, ' \
                   f'arg={results["settings"]["semantic"]}'
    title_str = f'n = {N}, {analysis_str}, \n' \
                f'{conn_str}, {second_order_str}, {trial_similarity_str}'

    plot_connectivity(t_all, results['ticks'], results['tick_labels'], results['tick_lows'],
                      title=f't-test: {title_str}', no_avg=True, cbar_label='t-value', vmin=-4, vmax=4)

    if N == 24 or N >= 30:
        results['scores_by_ROI'] = np.array(results['scores_by_ROI'])
        scores_trialwise = results['scores_by_ROI']
        for fp in range(scores_trialwise.shape[1]):
            scores_fp = np.mean(scores_trialwise[:, fp, :, :, :],
                                axis=-1)
            M_scores_fp = np.nanmean(scores_fp, axis=0)
            SD_scores_fp = np.nanstd(scores_fp, axis=0)
            N_scores_fp = np.sum(~np.isnan(scores_fp), axis=0)
            SE_scores_fp = SD_scores_fp / np.sqrt(N_scores_fp)
            t_scores_fp = M_scores_fp / SE_scores_fp
            plot_connectivity(t_scores_fp, results['ticks'], results['tick_labels'], results['tick_lows'],
                              title=f't-test (fp={fp}): {title_str}', no_avg=True, cbar_label='t-value', vmin=-4,
                              vmax=4)
        # quit()

    if not plot_lmer:
        return

    settings = results['settings']
    lmer_fp = get_default_fp(None, settings, get_lmer_matrix,
                             r'../cache/lmer_ar', False)

    lmer_ar = pickle_wrap(lambda: get_lmer_matrix(results), lmer_fp)

    plot_connectivity(lmer_ar, results['ticks'], results['tick_labels'], results['tick_lows'],
                      title=f'lmer: {title_str}', no_avg=True, cbar_label='t-value', vmin=-4, vmax=4)


def visualize_ROIs(results, do_lmer=False):
    from connsearch.report import plot_ROI_scores
    if 'atlas' in results['settings'] and results['settings']['atlas'] == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=results['settings']['combine_regions'],
                          combine_bilateral=False,
                          split=results['settings']['split'], split_code='xyz')
    # print(atlas['ROIs'])
    # quit()
    ROI2coord = atlas['ROI2coord']
    # print(ROI2coord)
    # print(f'{len(ROI2coord)=}')
    # print(list(atlas['ROI2coord']))
    # quit()
    results['keys'] = [ROI.replace('LH_', 'L_').replace('RH_', 'R_')
                       for ROI in results['keys']]
    # print(list(ROI2coord))
    # print(results['keys'])
    results_coords = [ROI2coord[ROI] for ROI in results['keys']]
    ts, ts_by_fp = report_results(results, do_lmer=do_lmer)
    print(f'{do_lmer=}')



    # print(f'{len(atlas["ROIs"])=}')
    # quit()

    ROI2atlas_idx = {ROI: i for i, ROI in enumerate(atlas['ROIs'])}
    scores = np.full((len(atlas['ROIs']),), np.nan)
    for i, ROI in enumerate(results['keys']):
        scores[ROI2atlas_idx[ROI]] = ts[i]
    if results['settings']['RSA']:
        title_short = make_title_str('', 'obj', 1, False,
                                     results['settings']['semantic'],
                                     short=True)
    else:
        title_short = 'Object IPS. YA.'
    my_plot_surf(scores, atlas, title_short)
    quit()

    # ROI2score = {}
    # for i, ROI in enumerate(results['keys']):
    #     ROI2score[ROI] = ts[i]
    # scores_w_NaNs = []
    # print(f'{len(ROI2score)=}')
    # for ROI in atlas['ROIs']:
    #     if ROI in ROI2score:
    #         scores_w_NaNs.append(ROI2score[ROI])
    #     else:
    #         scores_w_NaNs.append(np.nan)
    # print(f'{len(scores_w_NaNs)=}')
    # print(ts)
    # quit()
    plot_ROI_scores(ts, results_coords, fp_out='trash.png', show=True,
                    vmin=0, vmax=2, title='all')

    for k in range(ts_by_fp.shape[0]):
        plot_ROI_scores(ts_by_fp[k, :], results_coords, fp_out='trash.png',
                        show=True, vmin=0, vmax=3, title=f'fp: {k}')

    quit()


def RSA_ROI(sn, atlas, d_vecs, fp, networks=True, conn='euc',
            trial_similarity='corr', second_order='spear',
            RDM_method='by_run', combine_regions=False,
            PFC=False, PFC2=False, ROI_ctrl=False):
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
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

        if ROI_ctrl:# or (not PFC):
            vecs -= mean_conn_trialwise.T# * 246 - vecs) / 245

        z, IRAFs = get_RSM_subtract_run_mean_fast(vecs, RSM_stim, df_sn,
                                                  trial_similarity=trial_similarity,
                                                  second_order=second_order)

        scores.append(z)
        scores_trialwise.append(IRAFs)

    # quit()
    return scores, sizes, scores_trialwise


def RSA_ROI_pairwise(sn, atlas, d_vecs, fp, networks=True, conn='euc',
                     trial_similarity='corr', second_order='spear',
                     RDM_method='by_run', combine_regions=False):

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                 networks=False, org_by_region=False,
                                 cross_region=False, conn=conn,
                                 combine_regions=False)


    ROI2vecs, region_order, score_ar, IRAFs_ar = \
        prep_for_pairwise(ROI2vecs, atlas)

    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True,
                            dist=trial_similarity)

    for i, ROI0 in enumerate(region_order):

        vecs0 = ROI2vecs[ROI0]

        for j, ROI1 in enumerate(region_order):

            if ROI1 > ROI0:
                continue
            elif ROI0 == ROI1:
                vecs = get_conn_vecs(vecs0, conn=conn)
            else:
                vecs1 = ROI2vecs[ROI1]
                vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
            if vecs.shape[1] == 1:
                score_ar[i, j] = np.nan
                score_ar[j, i] = np.nan
                IRAFs_ar[i, j, :] = np.full((114), np.nan)
                IRAFs_ar[j, i, :] = np.full((114), np.nan)
                continue

            z, IRAFs = get_RSM_subtract_run_mean_fast(vecs, RSM_stim, df_sn,
                                    trial_similarity=trial_similarity,
                                                      second_order=second_order)
            score_ar[i, j] = z
            score_ar[j, i] = z
            IRAFs_ar[i, j, :] = IRAFs
            IRAFs_ar[j, i, :] = IRAFs

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

    return z, IRAFs


def RSA_edgewise(sn, atlas, d_vecs, fp, networks=True,
           conn='euc', trial_similarity='corr', second_order='spear',
           RDM_method='by_run', combine_regions=False):
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
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
