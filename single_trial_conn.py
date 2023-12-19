from tqdm import tqdm

from atlas_utils import get_atlas
from conn_RSA import RSA_sn, RSA_edgewise, RSA_ROI_pairwise, RSA_ROI
from conn_ERS import ERS_sn, ERS_ROI_pairwise#, ERS_ROI
from conn_report import report_results, visualize_region_matrix, visualize_ROIs
from conn_utils import get_BNA_ROIs
from networks import prep_networks
from org_sns import get_all_sns
from old.plot_gen import plot_connectivity
import numpy as np

from stim import get_semantic_vectors, get_DNN_vecs
from utils import pickle_wrap
from datetime import datetime
from colorama import Fore
from functools import partial

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
            elif 'ROIs' in plotting:
                PFC2 = 'PFC2' in plotting
                PFC = ('PFC' in plotting) and ('PFC2' not in plotting)
                ctrl = 'ctrl' in plotting
                f = partial(RSA_ROI, PFC=PFC, PFC2=PFC2, ROI_ctrl=ctrl)
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
            if plotting is None:
                f = ERS_sn
            elif 'ROIs' in plotting:
                PFC2 = 'PFC2' in plotting
                PFC = ('PFC' in plotting) and ('PFC2' not in plotting)
                ctrl = 'ctrl' in plotting
                f = partial(ERS_ROI, PFC=PFC, PFC2=PFC2, ROI_ctrl=ctrl)
            # elif plotting == 'ROIs_PFC_ctrl':
            #     f = partial(ERS_ROI, PFC=True, ROI_ctrl=True)
            # elif plotting == 'ROIs_PFC':
            #     f = partial(ERS_ROI, PFC=True, ROI_ctrl=False)
            # elif plotting == 'ROIs':
            #     f = ERS_ROI
            elif plotting == 'edges':
                raise ValueError(f'run_sn ERS unknown: {plotting=}')
            elif plotting == 'regions':
                f = ERS_ROI_pairwise
            else:
                raise ValueError(f'run_sn ERS unknown: {plotting=}')
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                scores, sizes, trialwise = \
                    f(sn, atlas, fp0, fp1, networks=networks,
                           conn=conn, trial_similarity=trial_similarity,
                           combine_regions=combine_regions,)
                scores_all.append(scores)
                sizes_all.append(sizes)
                trialwise_all.append(trialwise)
    M_score_by_ROI = np.nanmean(scores_all, axis=0)

    M_size_by_ROI = np.nanmean(sizes_all, axis=0)
    trialwise_all = np.array(trialwise_all)

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
    if four_tasks == '3b_4':
        fps = ['bl3_fMRI', 'cmb3_fMRI', 'vis3_fMRI', 'con3_fMRI']
    elif four_tasks == '3b_3':
        fps = ['bl3_fMRI', 'cmb3_fMRI', 'vis3_fMRI']
    elif four_tasks == '3_4':
        fps = ['bl3_fMRI', 'obj3_fMRI', 'vis3_fMRI', 'con3_fMRI']
    elif four_tasks == '3_3':
        fps = ['bl3_fMRI', 'obj3_fMRI', 'vis3_fMRI']
    elif four_tasks:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    else:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI']
    return fps

def prep_results_d(settings, networks, atlas, sns):
    if networks is None:
        if settings['conn'] == 'BOLD':
            keys = atlas['ROIs']
        else:
            keys = atlas['tick_labels']
    else:
        if 'plotting' in settings and settings['plotting'] is not None:
            if 'ROIs_PFC2' in settings['plotting']:
                if settings['split']:
                    raise NotImplementedError('No keys for PFC2 and split')
                else:
                    keys = get_BNA_ROIs(code='PFC_ACC')
            elif 'ROIs_PFC' in settings['plotting']:
                if settings['atlas'] == 'schaefer':
                    keys = get_BNA_ROIs(code='PFC_schaefer')
                elif settings['split']:
                    keys = get_BNA_ROIs(code='PFC_8')
                else:
                    keys = get_BNA_ROIs(code='PFC')
            elif 'ROIs' in settings['plotting']:
                if settings['atlas'] == 'schaefer':
                    keys = get_BNA_ROIs(code='schaefer')
                else:
                    keys = get_BNA_ROIs()
            else:
                keys = list(networks)
        else:
            keys = list(networks)

    results = {'networks': networks, 'keys': keys,
               'sns': sns, 'scores': [], 'sizes': [],
               'tick_labels': atlas['tick_labels'],
               }
    if settings['plotting'] == 'regions':
        if settings['atlas'] == 'schaefer':
            results['ticks'] = 0.5 + np.arange(38) * 2
            results['tick_lows'] = 0 + np.arange(38) * 2
        else:
            results['ticks'] = 0.5 + np.arange(26) * 2
            results['tick_lows'] = 0 + np.arange(26) * 2
            results['ticks'] = 0.5 + np.arange(27) * 2
            results['tick_lows'] = 0 + np.arange(27) * 2

    elif settings['plotting'] == 'edges':
        results['ticks'] = atlas['ticks']
        results['tick_lows'] = atlas['tick_lows']
    results['settings'] = settings
    return results

def run_settings(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, RDM_method='by_run',
                 age=1, plotting=None, atlas='BNA'):
    settings = locals().copy()
    print(f'Run settings start: {settings=}')
    d_vecs = prep_vecs(RSA, semantic)
    # print(f'{atlas=}')
    # quit()
    if atlas == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=False,
                          split=split, split_code='xyz',)
                          # shenyang='_sh' in atlas)
    fps = prep_fps(four_tasks)
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn[age]
    if isinstance(do_networks, str):
        raise ValueError(f'Why is do_networks a string? {do_networks=}')
    networks = prep_networks(do_networks)
    results = prep_results_d(settings, networks, atlas, sns)

    trialwise_all = []
    for i, sn in tqdm(enumerate(sns), desc='run_settings, looping subjects'):
        ers_sn_by_comparison = []
        scores, sizes, scores_trialwise = \
            run_sn(fps, RSA, sn, atlas, d_vecs, networks=networks,
                   conn=conn, trial_similarity=trial_similarity,
                   second_order=second_order, RDM_method=RDM_method,
                   combine_regions=combine_regions, plotting=plotting)
        results['scores'].append(scores)
        results['sizes'].append(sizes)
        trialwise_all.append(scores_trialwise)
        results['scores_by_ROI'] = np.array(trialwise_all)
        results['trialwise'] = np.array(trialwise_all)
        if plotting is None or ('ROIs' in plotting):
            report_results(results)
        else:
            if i > 1:
                visualize_region_matrix(results)
    return results


def run_settings_healthy(settings, ISPC=False):
    dir_results = r'cache/conn_RSA'
    # if settings['semantic'] == -1:
    #     dt_max = datetime(2023, 11, 24, 11, 0, 0, 0)
    # else:
    #     dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)
    #
    # if 'cross_' in settings['conn']:
    #     dt_max = datetime(2023, 11, 26, 11, 0, 0, 0)

    dt_max = datetime(2023, 12, 13, 20, 0, 0, 0)


    settings1 = settings.copy()
    del settings1['age']
    results1 = pickle_wrap(None, run_settings, kwargs=settings1,
                          cache_dir=dir_results, easy_override=EASY_OVERRIDE,
                          dt_max=dt_max, verbose=0)
    print(f'{Fore.CYAN}Young people:{Fore.RESET}')
    report_results(results1)

    settings2 = settings.copy()
    settings2['age'] = 2
    results2 = pickle_wrap(None, run_settings, kwargs=settings2,
                          cache_dir=dir_results, easy_override=EASY_OVERRIDE,
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
    if 'ticks' in results1:
        results_both['ticks'] = results1['ticks']
        results_both['tick_lows'] = results1['tick_lows']
        results_both['tick_labels'] = results1['tick_labels']

                    # 'ticks': results2['ticks'],
                    # 'tick_lows': results2['tick_lows'],
                    # 'tick_labels': results2['tick_labels'],}

    return results_both


def run_analysis(RSA=True, semantic=False, do_networks=1,
                 conn='euc', trial_similarity='corr', second_order='spear',
                 four_tasks=False, combine_regions=False, split=True,
                 RDM_method='clever_std', age=1, plotting=None, atlas='BNA'):
    settings = locals().copy()

    if plotting is None:
        del settings['plotting']
    else:
        settings['do_networks'] = -1
        settings['RDM_method'] = 'clever_std'

    if atlas == 'BNA':
        del settings['atlas']

    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks or conn == 'BOLD'
    assert RSA or second_order == 'spear', 'Leave second_order as \"spear\" for ERS'
    assert not (conn == 'BOLD' and do_networks), 'Not conn=BOLD and do networks'
    assert not (split and do_networks in [2, 7]), 'Too computationally intense'
    assert not (split and 'PFC2' in plotting)
    if do_networks == 3:
        if 'cross' not in conn:
            settings['conn'] = f'cross_{conn}'
            print(f'Missing \"cross_\" for networks 3, Changed conn to {conn}')

    # if semantic == -1: # added due to discovered issue at one point with semantic
    #     dt_max = datetime(2023, 11, 24, 11, 0, 0, 0)
    # else:
    #     dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)
    #
    # if 'cross_' in conn:
    #     dt_max = datetime(2023, 11, 26, 11, 0, 0, 0)
    #
    # if 'plotting' in settings and settings['plotting'] == 'ROIs':
    #     dt_max = datetime(2023, 11, 29, 17, 0, 0, 0)
    dt_max = datetime(2023, 12, 13, 20, 0, 0, 0)
    #
    # settings = {'RSA': True, 'semantic': True, 'do_networks': -1, 'conn': 'euc', 'trial_similarity': 'spear',
    #             'second_order': 'spear', 'four_tasks': '3_4', 'combine_regions': False, 'split': False,
    #             'RDM_method': 'clever_std', 'age': 1, 'plotting': 'regions'}#, 'atlas': 'BNA'}

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
                              cache_dir=dir_results, easy_override=EASY_OVERRIDE,
                              dt_max=dt_max, verbose=1)
        print(f'Finished!')

    if ('plotting' in settings) and ('ROIs' in settings['plotting']):
        visualize_ROIs(results)
    if 'plotting' not in settings or 'ROIs' in settings['plotting']:
        report_results(results, do_lmer=True)
    else:
        visualize_region_matrix(results, plot_lmer=True)

def run_analysis_toggles():
    RDM_method = 'clever_std_complex_mean'
    # RDM_method = 'clever_std'
    trial_similarity_toggle = ['spear']
    four_tasks_toggle = ['3_4']
    conn_toggle = ['euc']
    split_toggle = [False]
    # atlas = 'schaefer'
    atlas = 'BNA_sh'
    # atlas = None
    age = 'healthy'
    # age = 1
    combine_regions = False
    # plotting = 'regions'
    # plotting = 'ROIs_PFC2_ctrl'
    # plotting = 'ROIs'
    plotting = None
    # 'ROIs_ctrl' # no ROI is above t=2.1 for whole-brain
    # plotting = 'regions'
    # second_order = 'spear'
    second_order = 'spear'
    # analyses = [(True, True)]
    # analyses = [(True, False)]
    analyses = [(False, False), (True, True), (True, False)]
    # analyses = [(True, True)]
    for trial_similarity in trial_similarity_toggle:
        for four_tasks in four_tasks_toggle:
            for do_networks in [14, 15]:
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
                                             second_order=second_order,
                                             age=age, plotting=plotting,
                                             atlas=atlas)
                            except AssertionError as e:
                                print(f'Assertion no bueno: {e}')
                                pass

if __name__ == '__main__':
    EASY_OVERRIDE = True
    run_analysis_toggles()

