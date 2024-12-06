import os

from networks.old.networks import prep_networks

os.chdir(r'C:\PycharmProjects\SchemeRep')

from tqdm import tqdm
from pprint import pprint

from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from connRSA.old_Sep29.conn_RSA import RSA_sn
from connRSA.old_Sep29.conn_ERS import ERS_sn
from connRSA.old_Oct29.conn_report import report_results
from connRSA.old.conn_old import visualize_region_matrix, visualize_ROIs, RSA_ROI, RSA_ROI_pairwise, RSA_edgewise
# from old.networks import prep_networks
from org_sns import get_sns
import numpy as np

from stim import get_semantic_vectors, get_DNN_vecs
from Utils.pickle_wrap_funcs import pickle_wrap
from datetime import datetime
from colorama import Fore
from functools import partial, cache

def run_sn(fps, RSA, sn, atlas, d_vecs, networks=None,
           conn='euc', trial_similarity='euc', second_order='spear',
           RDM_method='by_run', combine_regions=False, plotting=None,
           stdize_by_run=False, semantic=False):
    scores_all = []

    sizes_all = []
    trialwise_all = []
    # print(fps)
    # print(RSA)
    # quit()
    for fp0 in fps:
        # if fp0 != 'obj7_fMRI':
        #     continue

        if RSA:
            if plotting is None:
                f = RSA_sn
                # print('test')
                # quit()
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
                  combine_regions=combine_regions, stdize_by_run=stdize_by_run,
                  semantic=semantic)
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
                           combine_regions=combine_regions,
                      stdize_by_run=stdize_by_run)
                scores_all.append(scores)
                sizes_all.append(sizes)
                trialwise_all.append(trialwise)
    M_score_by_ROI = np.nanmean(scores_all, axis=0)

    M_size_by_ROI = np.nanmean(sizes_all, axis=0)
    trialwise_all = np.array(trialwise_all)

    return M_score_by_ROI, M_size_by_ROI, trialwise_all

@cache
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
    if four_tasks == '7':
        fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']

    elif four_tasks == '8':
        fps = ['bl8_fMRI', 'obj8_fMRI', 'con8_fMRI', 'vis8_fMRI']
    else:
        raise NotImplementedError
    return fps

def prep_results_d(settings, networks, atlas, sns):
    if networks is None:
        if 'BOLD' in settings['conn']:
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

def  run_settings(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, RDM_method='by_run',
                 age=1, plotting=None, atlas='BNA',
                 stdize_by_run=False):
    # assert not ((conn == 'BOLD') and RDM_method and ('clever' in RDM_method) and
    #             (not do_networks))
    settings = locals().copy()
    print(f'Run settings start:')
    pprint(settings)
    d_vecs = prep_vecs(RSA, semantic)
    # print(f'{atlas=}')
    # quit()
    if atlas == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=combine_regions,
                          # split=split,
                          # split_code='xyz',
                          )
                          # shenyang='_sh' in atlas)
    fps = prep_fps(four_tasks)
    age2sn = get_sns('all', sh=False)
    sns = age2sn[age]
    # sns = ['126']

    # sns = sns[4::5]
    # print(RSA)
    # quit()

    if isinstance(do_networks, str):
        raise ValueError(f'Why is do_networks a string? {do_networks=}')
    networks = prep_networks(do_networks)
    results = prep_results_d(settings, networks, atlas, sns)

    trialwise_all = []
    for i, sn in tqdm(enumerate(sns), desc='run_settings, looping subjects'):
        print(f'connRSA: {sn=}')
        ers_sn_by_comparison = []
        scores, sizes, scores_trialwise = \
            run_sn(fps, RSA, sn, atlas, d_vecs, networks=networks,
                   conn=conn, trial_similarity=trial_similarity,
                   second_order=second_order, RDM_method=RDM_method,
                   combine_regions=combine_regions, plotting=plotting,
                   stdize_by_run=stdize_by_run, semantic=semantic)
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

    dt_max = datetime(2023, 12, 13, 20,
                      0, 0, 0)


    settings1 = settings.copy()
    del settings1['age']
    results1 = pickle_wrap(run_settings, None, kwargs=settings1,
                           easy_override=EASY_OVERRIDE, verbose=0,
                           cache_dir=dir_results, dt_max=dt_max)
    print(f'{Fore.CYAN}Young people:{Fore.RESET}')
    report_results(results1)

    settings2 = settings.copy()
    settings2['age'] = 2
    results2 = pickle_wrap(run_settings, None, kwargs=settings2,
                           easy_override=EASY_OVERRIDE, verbose=0,
                           cache_dir=dir_results, dt_max=dt_max)
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
                 RDM_method='clever_std', age=1, plotting=None, atlas='BNA',
                 stdize_by_run=False):
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

    dt_max = datetime(2023, 12, 13, 20, 0, 0, 0)

    # settings = {'RSA': True, 'semantic': True, 'do_networks': -1, 'conn': 'euc', 'trial_similarity': 'spear',
    #             'second_order': 'spear', 'four_tasks': '3_4', 'combine_regions': False, 'split': False,
    #             'RDM_method': 'clever_std', 'age': 1, 'plotting': 'regions'}#, 'atlas': 'BNA'}
    # pprint(f'{settings=}')
    dir_results = r'cache/conn_RSA'
    if age == 'healthy':
        print('\n')
        print('*' + '-*' * 120)
        results = pickle_wrap(run_settings_healthy, None,
                              kwargs={'settings': settings}, easy_override=True,
                              cache_dir=dir_results, dt_max=dt_max)
        print(f'{Fore.RED}Combined people:{Fore.RESET}')
    else:
        if age == 2:
            assert isinstance(four_tasks, str) and '3_' in four_tasks, 'Bad OA'
        else:
            del settings['age']
        results = pickle_wrap(run_settings, None, kwargs=settings,
                              easy_override=EASY_OVERRIDE, verbose=1,
                              cache_dir=dir_results, dt_max=dt_max)
        print(f'Finished!')

    if ('plotting' in settings) and ('ROIs' in settings['plotting']):
        visualize_ROIs(results)
    if 'plotting' not in settings or 'ROIs' in settings['plotting']:
        report_results(results, do_lmer=True)
    else:
        visualize_region_matrix(results, plot_lmer=True)
