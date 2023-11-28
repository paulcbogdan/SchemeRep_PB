import pandas as pd
from tqdm import tqdm

from atlas_utils import get_atlas
from conn_RSA import RSA_sn, RSA_edgewise, RSA_ROI_pairwise
from conn_ERS import ERS_sn
from conn_report import report_results
from networks import prep_networks
from organize_bhv import get_all_sns
from old.plot_gen import plot_connectivity
import numpy as np

from stim import get_semantic_vectors, get_DNN_vecs
from utils import pickle_wrap
from datetime import datetime
from time import time
from colorama import Fore

import matplotlib.pyplot as plt

def run_sn(fps, RSA, sn, atlas, d_vecs, networks=None,
           conn='euc', trial_similarity='euc', second_order='spear',
           RDM_method='by_run', combine_regions=False, plotting=None):
    scores_all = []
    sizes_all = []
    trialwise_all = []
    for fp0 in fps:
        if RSA:
            if plotting is None:
                f = RSA_sn
            elif plotting == 'edges':
                f = RSA_edgewise
            elif plotting == 'regions':
                f = RSA_ROI_pairwise
            else:
                raise ValueError(f'run_sn RSA unknown: {plotting=}')
            scores, sizes, trialwise = \
                f(sn, atlas, d_vecs, fp0, networks=networks,
                       conn=conn, trial_similarity=trial_similarity,
                       second_order=second_order, RDM_method=RDM_method,
                       combine_regions=combine_regions,
                       )
            scores_all.append(scores)
            sizes_all.append(sizes)
            trialwise_all.append(trialwise)
        else:
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                scores, sizes, trialwise = \
                    ERS_sn(sn, atlas, fp0, fp1, networks=networks,
                           conn=conn, trial_similarity=trial_similarity,
                           combine_regions=combine_regions,)
                scores_all.append(scores)
                sizes_all.append(sizes)
                trialwise_all.append(trialwise)
    M_score_by_ROI = np.nanmean(scores_all, axis=0)

    M_size_by_ROI = np.nanmean(sizes_all, axis=0)
    trialwise_all = np.array(trialwise_all)
    # print(f'{scores_by_ROI.shape=}')
    # quit()
    return M_score_by_ROI, M_size_by_ROI, trialwise_all

def prep_vecs(RSA, semantic):
    if RSA:
        if isinstance(semantic, bool):
            if semantic:
                d_vecs = get_semantic_vectors(normalize=True)
            else:
                d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
        elif isinstance(semantic, int):
            last_layer = semantic == -1
            d_vecs = get_DNN_vecs(DNN_layer=semantic, PCA=not last_layer,
                                  PCA_obj=True)
        else:
            raise ValueError(f'Invalid {semantic=}')
        for obj, vec in d_vecs.items():
            print(f'RSA stimulus vector size: {vec.shape=}')
            break
    else:
        d_vecs = None
    return d_vecs

def prep_fps(four_tasks):
    if four_tasks == '3_4':
        fps = ['bl3_fMRI', 'obj3_fMRI', 'vis3_fMRI', 'con3_fMRI']
    elif four_tasks == '3_3':
        fps = ['bl3_fMRI', 'obj3_fMRI', 'vis3_fMRI']
    elif four_tasks:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    else:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI']
    return fps


def run_settings(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, RDM_method='by_run',
                 age=1, plotting=None):
    settings = locals().copy()
    print(f'Run settings start: {settings=}')
    d_vecs = prep_vecs(RSA, semantic)
    # vecs_l = list(d_vecs.values())
    # random.shuffle(vecs_l)
    # d_vecs = dict(zip(d_vecs.keys(), vecs_l))
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False,
                      split=split, split_code='xyz')
    fps = prep_fps(four_tasks)
    age2sn = get_all_sns(ret=True)
    sns = age2sn[age]

    if (isinstance(do_networks, int) and do_networks) or \
            isinstance(do_networks, str):
        networks = prep_networks(do_networks)
        keys = list(networks)
    else:
        networks = None
        if conn == 'BOLD':
            # if combine_regions:
            #     keys = atlas['tick_labels']
            # else:
            keys = atlas['ROIs']
        else:
            keys = atlas['tick_labels']


    # print(atlas['ROIs'])
    # quit()
    # print(f'{len(keys)=}')
    # quit()

    # else:
    #     raise ValueError(f'Unknown do_networks: {do_networks}')

    results = {'networks': networks, 'keys': keys,
               'sns': sns, 'scores': [], 'sizes': [],
               'scores_by_ROI': [],
               'tick_labels': atlas['tick_labels'],
               }
    if plotting == 'regions':
        results['ticks'] = 0.5 + np.arange(26) * 2
        results['tick_lows'] = 0 + np.arange(26) * 2
    elif plotting == 'edges':
        results['ticks'] = atlas['ticks']
        results['tick_lows'] = atlas['tick_lows']
    results['settings'] = settings
    for i, sn in tqdm(enumerate(sns), desc='ERS, looping subjects'):
        ers_sn_by_comparison = []
        scores, sizes, scores_by_ROI = \
            run_sn(fps, RSA, sn, atlas, d_vecs, networks=networks,
                   conn=conn, trial_similarity=trial_similarity,
                   second_order=second_order, RDM_method=RDM_method,
                   combine_regions=combine_regions, plotting=plotting)
        results['scores'].append(scores)
        results['sizes'].append(sizes)
        results['scores_by_ROI'].append(scores_by_ROI)
        results['trialwise'] = np.array(results['scores_by_ROI'])
        if plotting is None:
            report_results(results)
        else:
            if i > 1:
                visualize_region_matrix(results)

    return results

def visualize_region_matrix(results, plot_lmer=False):
    M_all = np.nanmean(results['scores'], axis=0)
    SD_all = np.nanstd(results['scores'], axis=0)
    N_all = np.sum(~np.isnan(results['scores']), axis=0)
    N = np.max(N_all)
    SE_all = SD_all / np.sqrt(N_all)
    t_all = M_all / SE_all

    ticks = results['ticks']
    tick_labels = results['tick_labels']
    tick_lows = results['tick_lows']

    conn_str = f'conn={results["settings"]["conn"]}'
    second_order_str = f'second_order={results["settings"]["second_order"]}'
    trial_similarity_str = f'trial_similarity={results["settings"]["trial_similarity"]}'
    analysis_str = f'RSA={results["settings"]["RSA"]}, ' \
                   f'arg={results["settings"]["semantic"]}'
    title_str = f'n = {N}, {analysis_str}, \n' \
                f'{conn_str}, {second_order_str},{trial_similarity_str}'

    plot_connectivity(t_all,
                      ticks,
                      tick_labels,
                      tick_lows,
                      # atlas['ticks'],
                      # atlas['tick_labels'],
                      # atlas['tick_lows'],
                      no_avg=True,
                      title=f't-test: {title_str}',
                      cbar_label='t-value',
                      vmin=-4, vmax=4)

    if not plot_lmer:
        return

    print(f'{results["trialwise"].shape=}')
    n_regions = results['trialwise'].shape[2]
    print(f'{n_regions=}')
    lmer_ar = np.full((n_regions, n_regions), np.nan)
    for i in tqdm(range(n_regions), desc='running lmers'):
        for j in range(n_regions):
            if i > j:
                continue
            else:
                pair_scores = results['trialwise'][:, :, i, j, :]
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
                df.dropna(inplace=True)
                if len(df) < 10000:
                    print(f'{i}, {j} | many na drops {len(df)=}')
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

    plot_connectivity(lmer_ar,
                      ticks,
                      tick_labels,
                      tick_lows,
                      # atlas['ticks'],
                      # atlas['tick_labels'],
                      # atlas['tick_lows'],
                      no_avg=True,
                      title=f'lmer: {title_str}',
                      cbar_label='t-value',
                      vmin=-4, vmax=4)

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

def run_settings_healthy(settings, ISPC=False):
    # if ISPC:
    #     f = run_settings_healthy
    # else:
    #     f = run_settings
    dir_results = r'cache/conn_RSA'
    if settings['semantic'] == -1:
        dt_max = datetime(2023, 11, 24, 11, 0, 0, 0)
    else:
        dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)

    if 'cross_' in settings['conn']:
        dt_max = datetime(2023, 11, 26, 11, 0, 0, 0)

    settings1 = settings.copy()
    del settings1['age']
    results1 = pickle_wrap(None, run_settings, kwargs=settings1,
                          cache_dir=dir_results, easy_override=False,
                          dt_max=dt_max, verbose=0)
    print(f'{Fore.CYAN}Young people:{Fore.RESET}')
    report_results(results1)

    settings2 = settings.copy()
    settings2['age'] = 2
    results2 = pickle_wrap(None, run_settings, kwargs=settings2,
                          cache_dir=dir_results, easy_override=False,
                          dt_max=dt_max, verbose=0)
    print(f'{Fore.LIGHTYELLOW_EX}Old people:{Fore.RESET}')
    report_results(results2)

    results_both = {'networks': results1['networks'],
                    'keys': results1['keys'],
                    'sns': results1['sns'] + results2['sns'],
                    'scores': np.concatenate([results1['scores'],
                                              results2['scores']]),
                    'sizes': np.concatenate([results1['sizes'],
                                             results2['sizes']]),
                    'scores_by_ROI': np.concatenate([results1['scores_by_ROI'],
                                                     results2['scores_by_ROI']]),
                    'settings': settings}
    # print(results_both['scores'].shape)
    # print(results_both['scores_by_ROI'].shape)
    # quit()
    return results_both


def run_analysis(RSA=True, semantic=False, do_networks=1,
                 conn='euc', trial_similarity='corr', second_order='corr',
                 four_tasks=False, combine_regions=False, split=True,
                 RDM_method='clever_std', age=1, plotting=None):
    settings = locals().copy()

    if plotting is None:
        del settings['plotting']
    else:
        settings['do_networks'] = -1
        settings['RDM_method'] = 'clever_std'
    # if not edgewise:
    #     del settings['edgewise']
    # else:
    #     settings['do_networks'] = -1

    # assert not edgewise or (trial_similarity == 'corr' and second_order == 'corr')

    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks or conn == 'BOLD'
    assert RSA or second_order == 'spear', 'Leave second_order as \"spear\" for ERS'
    assert not (conn == 'BOLD' and do_networks), 'Not conn=BOLD and do networks'
    assert not (split and do_networks in [2, 7]), 'Too computationally intense'
    if do_networks == 3:
        if 'cross' not in conn:
            settings['conn'] = f'cross_{conn}'
            print(f'Missing \"cross_\" for networks 3, Changed conn to {conn}')
    # assert do_networks != 3 or 'cross' in conn
    # assert not split, 'No split!'

    if semantic == -1: # added due to discovered issue at one point with semantic
        dt_max = datetime(2023, 11, 24, 11, 0, 0, 0)
    else:
        dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)

    if 'cross_' in conn:
        dt_max = datetime(2023, 11, 26, 11, 0, 0, 0)

    dir_results = r'cache/conn_RSA'
    if age == 'healthy':
        print('\n')
        print('*' + '-*' * 120)
        results = pickle_wrap(None, run_settings_healthy,
                              kwargs={'settings': settings},
                              cache_dir=dir_results, easy_override=True,
                              dt_max=dt_max)
        print(f'{Fore.RED}Combined people:{Fore.RESET}')
    else:
        if age == 2:
            assert isinstance(four_tasks, str) and '3_' in four_tasks, 'Bad OA'
        else:
            del settings['age']
        results = pickle_wrap(None, run_settings, kwargs=settings,
                              cache_dir=dir_results, easy_override=False,
                              dt_max=dt_max)
        print(f'Finished!')
    if 'plotting' in settings:
        visualize_region_matrix(results, plot_lmer=True)
    else:
        report_results(results)

def run_analysis_toggles():
    # RSA = False
    # semantic = False
    # do_networks = 1
    # RDM_method = 'clever_std'
    # RDM_method = 'within_nan'
    RDM_method = 'clever_std_complex_mean'

    # conn_toggle = ['cross_euc', 'cross_prod']
    # conn_toggle = ['BOLD']
    # trial_similarity_toggle = ['seuclidean']
    trial_similarity_toggle = ['corr']#, 'spear']#, 'seuclidean']


    four_tasks_toggle = ['3_4']
    # four_tasks_toggle = ['3_3']
    # four_tasks_toggle = [True]
    # four_tasks_toggle = [False, True]
    # conn_toggle = ['cross_euc', 'cross_prod']
    # conn_toggle = ['euc']#, 'prod']
    conn_toggle = ['euc']
    # conn_toggle = ['BOLD']
    split_toggle = [False]
    age = 1
    combine_regions = False
    plotting = 'regions'
    # age = 1
    # split_toggle = [False]

    # analyses = [(True, False), (False, False)]#, (True, True)]
    # analyses = [(True, True), (True, False)]
    # analyses = [(False, False)]
    # analyses = [(True, False)]
    # analyses = [(True, -1)]#, (True, False), (False, False)]
    # analyses = [(True, False), ]
    # analyses = [(True, True), (False, False)]
    analyses = [(True, True)]#, (False, False), (True, False)]
    # analyses = [(True, False), (False, False)]
    for trial_similarity in trial_similarity_toggle:
        for four_tasks in four_tasks_toggle:
            # for do_networks in [6]:
            # for do_networks in [False]:
            for do_networks in [12]:
            # for do_networks in [1, 3, 4, 5, 6, 8, 9, False, 2, 7]: # 8, 1, 3, 4, 5, 6, 7, False
                for conn in conn_toggle:
                    for split in split_toggle:
                        for (RSA, semantic) in analyses:
                            try:
                                run_analysis(conn=conn,
                                             trial_similarity=trial_similarity,
                                             four_tasks=four_tasks,
                                             split=split,
                                             RSA=RSA, semantic=semantic,
                                             do_networks=do_networks,
                                             RDM_method=RDM_method,
                                             combine_regions=combine_regions,
                                             age=age, plotting=plotting)
                            except AssertionError as e:
                                print(f'Assertion no bueno: {e}')
                                pass

if __name__ == '__main__':
    # import scipy.stats as stats
    # r, _ = stats.spearmanr([], [])
    # r, _ = stats.pearsonr([], [])
    # quit()

    # print(isinstance(1, bool))
    # quit()
    run_analysis_toggles()

    # print(prep_networks(setting=2))
    # quit()

    # run_analysis(RSA=True, semantic=False, do_networks=True,
    #              conn='cross_euc', trial_similarity='corr', second_order='spear',
    #              four_tasks=True, combine_regions=False, split=False,
    #              RDM_method='clever_std')
    # run_analysis(RSA=False, semantic=False, do_networks=True,
    #              conn='BOLD', trial_similarity='corr', second_order='spear',
    #              four_tasks=False, combine_regions=False, split=False,
    #              RDM_method='clever_std')