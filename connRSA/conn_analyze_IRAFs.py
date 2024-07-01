import os
from datetime import datetime

from connRSA.old.conn_plot import pie_charts
from connRSA.conn_report import lmer_stats

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from collections import defaultdict

from conn_utils import get_BNA_ROIs
from organize_bhv import get_trial_info
from single_trial_conn import run_settings
from conn_report import report_results
from old.networks import prep_networks
from utils import pickle_wrap
import pandas as pd
import numpy as np
from tqdm import tqdm

ROI2NETWORK = {'Occipital': 1, 'Ventral': 1, 'Dorsal': 1, 'else_cortical': 2,
               'PFC': 16, 'PFC_ACC': 14, 'FP': 14, 'perceptual': 17,
               'full_frontal': 11, 'full_frontal_CG': 11, 'MTL': 9, 'MTL2': 20,
               'subcort': 21 , 'INScc': 22, 'cingulate': 22, 'cortical': 23,
               'IT': 24, 'ITL': 25, 'Parietal': 26, 'OC_IT': 27, 'OC_T': 27,
               'cortex': 28}
ROI2network_anat = {'SFG': 19, 'MFG': 19, 'IFG': 19, 'OrG': 19, 'PrG': 19,
                    'PCL': 19, 'ATL': 19, 'STG': 19, 'MTG': 19, 'ITG': 19,
                    'FuG': 19, 'PhG': 19, 'pSTS': 19, 'SPL': 19, 'IPL': 19,
                    'Pcun': 19, 'PoG': 19, 'INS': 19, 'PCC': 19, 'ACC': 19,
                    'EVC': 19, 'LOC': 19, 'sOcG': 19, 'Amyg': 19, 'Hipp': 19,
                    'Str': 19, 'Tha': 19}
ROI2NETWORK.update(ROI2network_anat)

def get_idx_from_key(l, substring):
    idxs = []
    keys = []
    not_idxs = []
    not_keys = []
    for i, key in enumerate(l):
        if substring in key:
            idxs.append(i)
            keys.append(key)
        else:
            not_idxs.append(i)
            not_keys.append(key)
    return idxs, keys

def load_for_lmer(RSA=True, semantic=False, do_networks=False,
                 conn='euc', trial_similarity='euc',
                 second_order='spear', four_tasks=False,
                 combine_regions=False, split=False, RDM_method='by_run',
                 age=1, plotting=None, atlas='BNA', stdize_by_run=False):
    settings = locals().copy()
    if settings['stdize_by_run'] == False:
        del settings['stdize_by_run']

    dir_results = r'cache/conn_RSA'

    if not RSA:
        settings['RDM_method'] = None
        settings['second_order'] = 'spear'

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)
    # dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)

    # if not settings['combine_regions'] and False:
    #
    #     results_conn = pickle_wrap(run_settings, None, kwargs=settings,
    #                                easy_override=False, verbose=1,
    #                                cache_dir=dir_results)
    #     report_results(results_conn, do_lmer=False)
        # quit()

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        # settings['RDM_method'] = 'clever_std'
        settings['RDM_method'] = 'within_nan'


    settings['conn'] = 'BOLD'
    print(f'BOLD ' * 10)
    results_bold_comb = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results,
                                   dt_max=dt_max)
    report_results(results_bold_comb, do_lmer=False)

    if settings['do_networks'] != 19: # this is covered by combine one below
        print(f'COMBINE BIG ' * 10)
        settings['combine_regions'] = True
        results_bold_cmb_big = pickle_wrap(run_settings, None,
                                           kwargs=settings, easy_override=False,
                                           verbose=1, cache_dir=dir_results,
                                           dt_max=dt_max)
        report_results(results_bold_cmb_big, do_lmer=False)

    # dt_max = datetime(2024, 6, 10, 0, 0, 0, 0)

    settings['do_networks'] = False
    settings['combine_regions'] = False
    results_bold_sep = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results,
                                   dt_max=dt_max)
    # report_results(results_bold_sep, do_lmer=False)
    # print(settings)
    # quit()


    settings['do_networks'] = False
    settings['combine_regions'] = True
    print(f'COMBINE ' * 10)

    results_bold_sep_big = pickle_wrap(run_settings, None,
                                       kwargs=settings, easy_override=False,
                                       verbose=1, cache_dir=dir_results)
    report_results(results_bold_sep_big, do_lmer=False)

    quit()
    return (results_conn, results_bold_comb, results_bold_sep_big,
            results_bold_sep)

def organize_df(results_conn, results_bold_sep, results_bold_com,
                do_networks, target_ROI,):
    scores_conn = results_conn['scores_by_ROI']
    print(f'{scores_conn.shape=}')
    scores_conn = np.array(scores_conn)
    # if not ISPC:
    scores_sep = results_bold_sep['scores_by_ROI']
    scores_sep = np.array(scores_sep)
    scores_comb = results_bold_com['scores_by_ROI']

    network2ROI, BOLD_keys_short = prep_network2ROI(do_networks, target_ROI)

    ROI_to_bold_keys = {}
    ROI_to_idxs = {}
    df_as_d = defaultdict(list)
    print('Organizing df for IRAF analysis')
    for sn_i, sn in tqdm(enumerate(results_conn['sns']), desc='Adding subjects...'):
        df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
        df_sn.sort_values(by='bl_trial', inplace=True)
        for ROI_j, ROI_name in enumerate(results_conn['keys']):
            if ROI_name != target_ROI:
                continue
            for fp_idx in range(scores_conn.shape[1]):
                scores_conn_fp = scores_conn[sn_i, fp_idx, ROI_j, :]
                n = scores_conn_fp.shape[0]
                df_as_d['fp_idx'].extend([str(fp_idx)] * n)
                df_as_d['ROI'].extend([ROI_name] * n)
                df_as_d['conn_score'].extend(scores_conn_fp)
                df_as_d['BOLD_score'].extend(
                    scores_comb[sn_i, fp_idx, ROI_j, :])
                # if ISPC: continue

                if ROI_name in ROI_to_idxs:
                    ROI_bold_keys_short = ROI_to_bold_keys[ROI_name]
                    idxs = ROI_to_idxs[ROI_name]
                else:
                    if do_networks and ROI_name in network2ROI:
                        # if ROI_name not in ROI_to_bold_keys:
                        idxs = []
                        ROI_bold_keys_short = []
                        for ROI_name2 in network2ROI[ROI_name]:
                            idxs2, ROI_bold_keys_short2 = \
                                get_idx_from_key(BOLD_keys_short, ROI_name2)
                            idxs.extend(idxs2)
                            ROI_bold_keys_short.extend(ROI_bold_keys_short2)
                    else:
                        idxs, ROI_bold_keys_short = \
                            get_idx_from_key(BOLD_keys_short, ROI_name)
                    ROI_to_idxs[ROI_name] = idxs
                    ROI_to_bold_keys[ROI_name] = ROI_bold_keys_short

                for idx, key_short in zip(idxs, ROI_bold_keys_short):
                    scores_bold_fp = scores_sep[sn_i, fp_idx, idx, :]
                    df_as_d[f'{key_short}'].extend(scores_bold_fp)

                for col in df_sn.columns:
                    # print(f'{col=} | {df_sn[col].values}')
                    df_as_d[col].extend(df_sn[col].values)


    ROI_cols = ROI_to_bold_keys[target_ROI]

    df = pd.DataFrame(df_as_d)
    # print(list(df.columns))
    # print(df[['ROI', 'conn_score', 'fp_idx', 'BOLD_score']])
    return df, ROI_cols

def prep_network2ROI(do_networks, target_ROI):
    if do_networks is None:
        do_networks = ROI2NETWORK[target_ROI]
    network2ROI = prep_networks(network_setting=do_networks)

    for key, l in network2ROI.items():
        if isinstance(l, tuple):
            network2ROI[key] = l[0] + l[1]
    assert target_ROI in network2ROI.keys(), \
        f'{target_ROI} not in settings choice'

    BOLD_keys = get_BNA_ROIs()
    key2short = {key: key.split(' ')[1] for key in BOLD_keys}
    BOLD_keys_short = [key2short[key] for key in BOLD_keys]

    return network2ROI, BOLD_keys_short


def rsum(row):
    return np.sqrt(np.sum((row ** 2) * np.sign(row)))


def run_lmer_PFC_RSA():
    RSA = True
    semantic = True
    conn = 'prod'
    trial_similarity = 'corr'
    second_order = 'spear'
    four_tasks = '7'
    combine_regions = False
    split = False
    # RDM_method = 'clever_std_complex_mean' # clever_std_complex_mean
    RDM_method = 'within_nan'
    # RDM_method = 'clever_std'
    age = 'healthy'
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = False

    # target_ROI = 'else_cortical'
    # target_ROI = 'perceptual'
    # target_ROI = 'Occipital'
    # target_ROI = 'Hipp'
    target_ROI = 'Parietal'
    # target_ROI = 'pSTS'
    # target_ROI = 'Ventral'
    # target_ROI = 'MTL'
    # target_ROI = 'Dorsal'
    # target_ROI = 'PFC'
    # target_ROI = 'subcort'


    do_networks = ROI2NETWORK[target_ROI]

    kwargs = {'RSA': RSA, 'semantic': semantic, 'do_networks': do_networks,
              'conn': conn, 'trial_similarity': trial_similarity,
              'second_order': second_order, 'four_tasks': four_tasks,
              'combine_regions': combine_regions, 'split': split,
              'RDM_method': RDM_method, 'age': age,
              'stdize_by_run': stdize_by_run
              }
    results_conn, results_bold_comb, results_bold_sep_big, results_bold_sep \
        = pickle_wrap(load_for_lmer, None, kwargs=kwargs,
                      easy_override=True)
    quit()

    report_results(results_conn, do_lmer=False)#True)
    report_results(results_bold_comb, do_lmer=False)#True)

    df, ROI_cols = organize_df(results_conn, results_bold_sep,
                               results_bold_comb, do_networks, target_ROI)
    df['img'] = list(range(114)) * (len(df) // 114)

    if RSA:
        if semantic:
            title = f'RSA (semantic): {target_ROI}'
        else:
            title = f'RSA (perceptual): {target_ROI}'
    else:
        title = f'NPS: {target_ROI}'
    if RSA and not semantic:
        title = title.replace('MTL', 'Med. Temp. Lobe')
    else:
        title = title.replace('MTL', 'Medial Temporal Lobe')
    title = title.replace('PFC_ACC', 'Prefrontal')

    only_con = False
    if only_con:
        df = df[df['fp_idx'].isin(['2',])]
        title += '\n(Only conceptual memory)'

    pie_charts(df, ROI_cols, title)
    lmer_stats(df, ROI_cols)


if __name__ == '__main__':
    # atlas = get_atlas(False, False)
    # print(atlas['tick_labels'])
    # quit()
    run_lmer_PFC_RSA()
