from tqdm import tqdm

from atlas_utils import get_atlas
from conn_RSA import RSA_sn
from conn_ERS import ERS_sn
from organize_bhv import get_all_sns
from old.plot_gen import plot_connectivity
import numpy as np
import scipy.stats as stats

from stim import get_semantic_vectors, get_DNN_vecs
from utils import pickle_wrap


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
                       second_order=second_order, RDM_method=RDM_method,
                       )
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
    # print(f'{scores_by_ROI.shape=}')
    # quit()
    return M_score_by_ROI, M_size_by_ROI, scores_by_ROI

def prep_vecs(RSA, semantic):
    if RSA:
        if isinstance(semantic, bool):
            if semantic:
                d_vecs = get_semantic_vectors(normalize=True)
            else:
                d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
        elif isinstance(semantic, int):
            d_vecs = get_DNN_vecs(DNN_layer=semantic, PCA=True, PCA_obj=True)
        else:
            raise ValueError(f'Invalid {semantic=}')
    else:
        d_vecs = None
    return d_vecs

def prep_fps(four_tasks):
    if four_tasks:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    else:
        fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI']
    return fps

def prep_networks(setting=1):
    if setting == 1:
        networks = {
            'Occipital': ['EVC', 'LOC', 'sOcG'],
            'Ventral': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG'],
            'Dorsal': ['SPL', 'IPL', 'Pcun', 'pSTS'],
            'dPFC': ['IFG', 'MFG', 'SFG'],
            'PFC_Occ': ['IFG', 'MFG', 'SFG', 'EVC', 'LOC', 'sOcG'],
            'FPCN': ['IFG', 'MFG', 'SFG', 'SPL', 'IPL', 'pSTS']
        }
    elif setting == 2:
        networks = {
            # 'Frontal': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', ],
            # 'PFC_sub': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'Amyg', 'Hipp',
            #             'Str', 'Tha'],
            'else': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS', 'SPL',
                     'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'Amyg', 'Hipp', 'Str',
                     'Tha'],
            'else_cortical': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS',
                              'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'CG'],
            'sub': ['Amyg', 'Hipp', 'Str', 'Tha'],
        }
    # elif setting == 3:
    #     networks = {
    #         'Sanity': (['IFG', 'MFG', 'INS']),
    #         'PFC_Occ': (['IFG', 'MFG', 'SFG', 'OrG', 'EVC', 'LOC']),
    #         'Hipp_Occ': (['Hipp', 'EVC', 'LOC']),
    #         'Parietal_Occ': (['SPL', 'IPL', 'pSTS', 'Pcun', 'EVC', 'LOC']),
    #         'Ventral_Occ': (['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'EVC', 'LOC']),
    #     }
    elif setting == 3:
        networks = {
            'Sanity': (['IFG', 'MFG'], ['INS']),
            'PFC_Occ': (['IFG', 'MFG', 'SFG', 'OrG'], ['EVC', 'LOC']),
            'Hipp_Occ': (['Hipp'], ['EVC', 'LOC']),
            'Parietal_Occ': (['SPL', 'IPL', 'pSTS', 'Pcun'], ['EVC', 'LOC']),
            'Ventral_Occ': (['ITG', 'FuG', 'PhG', 'ATL', 'MTG'], ['EVC', 'LOC']),
        }
    elif setting == 4:
        networks = {
            'PFC': ['SFG', 'MFG', 'IFG', 'OrG',],
        }
    elif setting == 5:
        networks = {
            'Frontal_CG': ['SFG', 'MFG', 'IFG', 'OrG'],
        }
    elif setting == 6:
        networks = {
            'Frontal_CG': ['SFG', 'MFG', 'IFG', 'OrG'],
            # 'dPFC_Occ': (['IFG', 'MFG', 'SFG'], ['EVC', 'LOC']),
            # 'dlPFC_Occ': (['IFG', 'MFG'], ['EVC', 'LOC']),
            'PFC_Hipp': ['IFG', 'MFG', 'SFG', 'Hipp', 'OrG'],
            'dPFC': ['IFG', 'MFG', 'SFG'],
            'DMN': ['OrG', 'CG', 'Pcun', 'IPL'],
            'Salience': ['INS', 'CG'],
            'FPCN': ['MFG', 'IFG', 'IPL'],
            'FPCN_CG': ['MFG', 'IFG', 'IPL', 'CG'],
            'dlPFC': ['IFG', 'MFG'],
            'mPFC_hipp': ['OrG', 'Hipp'],
        }
    elif setting == 7:
        networks = {'else_ventral': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL',
                                     'pSTS', 'SPL', 'IPL', 'Pcun', 'PoG', 'INS',
                                     'CG', 'Amyg', 'Hipp', 'Str', 'Tha', 'ITG',
                                     'FuG', 'PhG', 'ATL', 'MTG']}
    elif setting == 8:
        networks = {'ventral_hipp': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'Hipp'],
                    'temporal': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'STG', 'pSTS'],
                    'dorsal_proper': ['SPL', 'IPL', 'Pcun', 'PoG'],
                    }
    elif setting == 9:
        networks = {'MTL': ['ITG', 'FuG', 'ATL', 'Hipp']}
    # elif setting == 7:
    #     networks = {
    #         'dPFC_Occ': (['IFG', 'MFG', 'SFG'], ['EVC', 'LOC']),
    #         'dlPFC_Occ': (['IFG', 'MFG'], ['EVC', 'LOC']),
    #     }
    else:
        raise ValueError(f'Unknown setting: {setting}')
    return networks

def run_settings(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, RDM_method='by_run',
                 ):
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
    sns = age2sn[1]

    if do_networks:
        networks = prep_networks(do_networks)
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
        report_results(results)
    return results

def print_settings(settings):
    print(f'{settings=}')

def report_results(results):
    print_settings(results['settings'])
    scores = np.array(results['scores'])
    sizes = np.array(results['sizes'])
    if scores.shape[1] == 246:
        print('\t Results printing for BOLD is not yet implemented')
        return

    assert len(results['keys']) == scores.shape[1], \
        f'{len(results["keys"])=}, {scores.shape=}'

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

def run_analysis(RSA=True, semantic=False, do_networks=1,
                 conn='euc', trial_similarity='corr', second_order='spear',
                 four_tasks=False, combine_regions=False, split=True,
                 RDM_method='clever_std'):
    settings = locals().copy()
    assert RSA or (not RSA and not semantic), 'semantic only for RSA'
    assert not (combine_regions and split), 'cannot combine and split'
    assert (not combine_regions) or do_networks
    assert RSA or second_order == 'spear', 'Leave second_order as \"spear\" for ERS'
    assert not (conn == 'BOLD' and do_networks), 'Not conn=BOLD and do networks'
    assert not (split and do_networks in [2, 7]), 'Too computationally intense'
    if do_networks == 3:
        if 'cross' not in conn:
            settings['conn'] = f'cross_{conn}'
            print(f'Missing \"cross_\" for networks 3, Changed conn to {conn}')
    # assert do_networks != 3 or 'cross' in conn
    # assert not split, 'No split!'
    dir_results = r'cache/conn_RSA'
    results = pickle_wrap(None, run_settings, kwargs=settings,
                          cache_dir=dir_results, easy_override=False)
    print(f'Finished!')
    report_results(results)

def run_analysis_toggles():
    # RSA = False
    # semantic = False
    # do_networks = 1
    RDM_method = 'clever_std'
    # RDM_method = 'within_nan'

    # conn_toggle = ['cross_euc', 'cross_prod']
    # conn_toggle = ['BOLD']
    # trial_similarity_toggle = ['seuclidean']
    trial_similarity_toggle = ['spear']#, 'seuclidean']

    four_tasks_toggle = [True, False]
    # conn_toggle = ['cross_euc', 'cross_prod']
    conn_toggle = ['euc']
    # conn_toggle = ['BOLD']
    split_toggle = [False, True]
    # split_toggle = [False]

    # analyses = [(True, False), (False, False)]#, (True, True)]
    # analyses = [(True, True), (True, False)]
    # analyses = [(False, False)]
    # analyses = [(True, False)]
    analyses = [(True, -1), (True, False), (False, False), (True, 4)]
    for trial_similarity in trial_similarity_toggle:

        for four_tasks in four_tasks_toggle:
            for do_networks in [9, 8]: # 8, 1, 3, 4, 5, 6, 7, False
                for conn in conn_toggle:
                    for split in split_toggle:
                        for (RSA, semantic) in analyses:
                            try:
                                run_analysis(conn=conn,
                                             trial_similarity=trial_similarity,
                                             four_tasks=four_tasks, split=split,
                                             RSA=RSA, semantic=semantic,
                                             do_networks=do_networks,
                                             RDM_method=RDM_method,
                                             )
                            except AssertionError as e:
                                print(f'Assertion no bueno: {e}')
                                pass

if __name__ == '__main__':

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