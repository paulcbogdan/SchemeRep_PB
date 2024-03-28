import os
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

from colorama import Fore
from pprint import pprint

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

    results_conn = pickle_wrap(run_settings, None, kwargs=settings,
                               easy_override=False, verbose=1,
                               cache_dir=dir_results)
    # print(list(results_conn))
    # pprint(results_conn)



    # print('-' * 100)
    report_results(results_conn, do_lmer=True)
    # quit()

    if 'complex_mean' in settings['RDM_method']:
        # settings['RDM_method'] = 'clever_std'
        settings['RDM_method'] = 'within_nan'

    settings['conn'] = 'BOLD'
    results_bold_comb = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)

    report_results(results_bold_comb, do_lmer=True)


    settings['do_networks'] = False
    results_bold_sep = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)

    return results_conn, results_bold_comb, results_bold_sep

def organize_df(results_conn, results_bold_sep, results_bold_com,
                do_networks, target_ROI):
    scores_conn = results_conn['scores_by_ROI']
    scores_conn = np.array(scores_conn)
    scores_sep = results_bold_sep['scores_by_ROI']
    scores_sep = np.array(scores_sep)
    scores_comb = results_bold_com['scores_by_ROI']
    scores_comb = np.array(scores_comb)
    # print(f'{scores_conn.shape=}')
    # print(f'{scores_sep.shape=}')
    # print(f'{scores_comb.shape=}')
    # quit()

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
        # break

    # for key, l in df_as_d.items():
    #     print(f'{key}: {len(l)}')


    ROI_cols = ROI_to_bold_keys[target_ROI]

    df = pd.DataFrame(df_as_d)
    # print(list(df.columns))
    # print(df[['ROI', 'conn_score', 'fp_idx', 'BOLD_score']])
    return df, ROI_cols

def prep_network2ROI(do_networks, target_ROI):
    network2ROI = prep_networks(network_setting=do_networks)
    for key, l in network2ROI.items():
        if isinstance(l, tuple):
            network2ROI[key] = l[0] + l[1]
    assert target_ROI in network2ROI.keys(), f'{target_ROI} not in settings choice'

    BOLD_keys = get_BNA_ROIs()
    key2short = {key: key.split(' ')[1] for key in BOLD_keys}
    BOLD_keys_short = [key2short[key] for key in BOLD_keys]

    return network2ROI, BOLD_keys_short


def lmer_stats(df, ROI_cols):
    df_M = df.groupby('sn')['conn_score'].mean()
    M_score = df_M.mean()
    SD_score = df_M.std()
    SE_score = SD_score / np.sqrt(df_M.shape[0])
    t_score = M_score / SE_score

    print(f'Single FP: {M_score=:.3f}, {SD_score=:.3f}, {SE_score=:.3f}, '
          f'{t_score=:.3f}')
    # quit()

    # print(f'{list(ROI_to_bold_keys)=}')
    # ROI_cols = ROI_to_bold_keys[target_ROI]
    # print(f'{ROI_cols=}')
    # print(f'{len(ROI_cols)=}')
    # print()
    cols = ['ROI', 'sn', 'conn_score', 'hit_hit', 'inc', 'vis_hit', 'con_hit',
            'inc_str', 'per_inc_str', 'fp_idx'] + \
           ROI_cols
    df = df[cols]
    df.dropna(inplace=True)
    df['vis_hit'] = df['vis_hit'].astype(int)
    df['con_hit'] = df['con_hit'].astype(int)

    from pymer4.models import Lmer

    print('-' * 120)
    formula = 'conn_score ~ 1 + ' + ' + '.join(ROI_cols) + '+  (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())
    quit()

    print('-' * 120)
    formula = 'conn_score ~ 1 + hit_hit + (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + con_hit + (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())


    print('-' * 120)
    formula = 'conn_score ~ 1 + inc_str + (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|sn)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + fp_idx + (1|sn)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    for fp, df_fp in df.groupby('fp_idx'):
        print('-*-' * 20 + Fore.CYAN + fp + Fore.RESET + '-*-' * 20)
        formula = 'conn_score ~ 1 + (1|sn)'
        print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
        model = Lmer(formula, data=df_fp)
        model.fit(REML=True, verbose=True, summary=True)
        print(model.summary())

        print('-' * 20 + Fore.CYAN + ' ' + fp + ' ' + Fore.RESET + '-' * 20)
        formula = 'conn_score ~ 1 + ' + ' + '.join(ROI_cols) + '+  (1|sn)'
        print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
        model = Lmer(formula, data=df_fp)
        model.fit(REML=True, verbose=True, summary=True)
        print(model.summary())

        print('-' * 20 + Fore.CYAN + ' ' + fp + ' ' + Fore.RESET + '-' * 20)
        formula = 'conn_score ~ 1 + con_hit + (1|sn)'
        print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
        model = Lmer(formula, data=df_fp)
        model.fit(REML=True, verbose=True, summary=True)
        print(model.summary())

        print('-' * 20 + Fore.CYAN + ' ' + fp + ' ' + Fore.RESET + '-' * 20)
        formula = 'conn_score ~ 1 + inc_str + (1|sn)'
        print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
        model = Lmer(formula, data=df_fp)
        model.fit(REML=True, verbose=True, summary=True)
        print(model.summary())


def run_lmer_PFC_RSA():
    RSA = True
    semantic = True
    do_networks = 1
    conn = 'prod'
    trial_similarity = 'corr'
    second_order = 'spear'
    four_tasks = '7'
    combine_regions = False
    split = False
    # RDM_method = 'clever_std_complex_mean' # clever_std_complex_mean
    RDM_method = 'by_run'
    age = 'healthy'
    stdize_by_run = True

    target_ROI = 'Occipital'

    results_conn, results_bold_comb, results_bold_sep = (
        load_for_lmer(RSA=RSA, semantic=semantic, do_networks=do_networks,
                      conn=conn, trial_similarity=trial_similarity,
                      second_order=second_order, four_tasks=four_tasks,
                      combine_regions=combine_regions, split=split,
                      RDM_method=RDM_method,age=age,
                      stdize_by_run=stdize_by_run))



    df, ROI_cols = organize_df(results_conn, results_bold_sep,
                               results_bold_comb, do_networks, target_ROI)

    lmer_stats(df, ROI_cols)

            

if __name__ == '__main__':
    # atlas = get_atlas(False, False)
    # print(atlas['tick_labels'])
    # quit()
    run_lmer_PFC_RSA()
    # run_lmer_PFC_ERS()

