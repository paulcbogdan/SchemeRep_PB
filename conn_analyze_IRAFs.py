from collections import defaultdict

from conn_utils import get_BNA_ROIs
from organize_bhv import get_trial_info
from single_trial_conn import run_settings
from conn_report import report_results
from networks import prep_networks
from utils import pickle_wrap
import pandas as pd
import numpy as np
from tqdm import tqdm

from colorama import Fore

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


def load_for_lmer(RSA, semantic, split, four_tasks, do_networks,
                  trial_similarity, RDM_method, age):
    dir_results = r'cache/conn_RSA'

    settings = {'RSA': RSA, 'semantic': semantic, 'do_networks': do_networks,
                'conn': 'euc', 'trial_similarity': trial_similarity,  # change trial_similarity=spear
                'second_order': 'spear', 'four_tasks': four_tasks,
                'combine_regions': False, 'split': split,
                'RDM_method': RDM_method, 'age': age}

    if age == 1:
        del settings['age']

    results_conn = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                               cache_dir=dir_results, easy_override=False)
    print('-' * 100)
    report_results(results_conn, do_lmer=True)
    print('-' * 100)

    settings = {'RSA': RSA, 'semantic': semantic, 'do_networks': False,
                'conn': 'BOLD', 'trial_similarity': 'corr',
                'second_order': 'spear', 'four_tasks': four_tasks,
                'combine_regions': False, 'split': False,
                'RDM_method': RDM_method, 'age': age}
    results_bold = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                               cache_dir=dir_results, easy_override=False)
    # report_results(results_bold, do_lmer=True)
    return results_bold, results_conn

def organize_df(results_conn, results_bold, do_networks, target_ROI):
    scores_conn = results_conn['scores_by_ROI']
    scores_conn = np.array(scores_conn)
    scores_bold = results_bold['scores_by_ROI']
    scores_bold = np.array(scores_bold)

    network2ROI, BOLD_keys_short = prep_network2ROI(do_networks, target_ROI)

    ROI_to_bold_keys = {}
    df_as_d = defaultdict(list)
    print('Organizing df for IRAF analysis')
    for sn_i, sn in tqdm(enumerate(results_conn['sns']), desc='Adding subjects...'):
        df_sn = get_trial_info(sn, easy_override=True)

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
                if do_networks and ROI_name in network2ROI:
                    idxs = []
                    ROI_bold_keys_short = []
                    for ROI_name2 in network2ROI[ROI_name]:
                        # if ROI_name2 not in ['LOC', 'EVC', 'sOcG']:
                        #     continue
                        idxs2, ROI_bold_keys_short2 = \
                            get_idx_from_key(BOLD_keys_short, ROI_name2)
                        idxs.extend(idxs2)
                        ROI_bold_keys_short.extend(ROI_bold_keys_short2)
                else:
                    idxs, ROI_bold_keys_short = \
                        get_idx_from_key(BOLD_keys_short, ROI_name)

                ROI_to_bold_keys[ROI_name] = ROI_bold_keys_short
                for idx, key_short in zip(idxs, ROI_bold_keys_short):
                    scores_bold_fp = scores_bold[sn_i, fp_idx, idx, :]
                    df_as_d[f'{key_short}'].extend(scores_bold_fp)

                for col in df_sn.columns:
                    # print(f'{col=} | {df_sn[col].values}')
                    df_as_d[col].extend(df_sn[col].values)
        # break

    # for key, l in df_as_d.items():
    #     print(f'{key}: {len(l)}')

    df = pd.DataFrame(df_as_d)

    print(len(df))
    def mean_or_first(l):
        try:
            M = l.mean()
        except TypeError:
            M = l.iloc[0]
        return M

    # df = df.groupby(['sn', 'obj', 'ROI']).agg(mean_or_first).reset_index()
    # print(len(df))
    # print(list(df.columns))
    # quit()
    return df, ROI_to_bold_keys

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


def lmer_stats(df, ROI_to_bold_keys, target_ROI):
    df_M = df.groupby('sn').mean()
    M_score = df_M['conn_score'].mean()
    SD_score = df_M['conn_score'].std()
    SE_score = SD_score / np.sqrt(df_M.shape[0])
    t_score = M_score / SE_score

    print(f'Single FP: {M_score=:.3f}, {SD_score=:.3f}, {SE_score=:.3f}, {t_score=:.3f}')
    # quit()

    # print(f'{list(ROI_to_bold_keys)=}')
    target_bold_keys = ROI_to_bold_keys[target_ROI]
    print(f'{target_bold_keys=}')
    print(f'{len(target_bold_keys)=}')
    print()
    cols = ['ROI', 'sn', 'conn_score', 'hit_hit', 'inc', 'vis_hit', 'con_hit',
            'inc_str', 'per_inc_str', 'fp_idx'] + \
           target_bold_keys
    df = df[cols]
    df.dropna(inplace=True)
    df['vis_hit'] = df['vis_hit'].astype(int)
    df['con_hit'] = df['con_hit'].astype(int)

    from pymer4.models import Lmer

    print('-' * 120)
    formula = 'conn_score ~ 1 + ' + ' + '.join(target_bold_keys) + '+  (1|sn) + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

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
        formula = 'conn_score ~ 1 + ' + ' + '.join(target_bold_keys) + '+  (1|sn)'
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
    split = False
    four_tasks = '3_4'
    do_networks = 6
    target_ROI = 'Frontal_CG'
    trial_similarity = 'spear'
    # RDM_method = 'clever_std'
    RDM_method = 'clever_std_complex_mean'
    age = 1

    results_bold, results_conn = load_for_lmer(RSA, semantic, split,
                                               four_tasks, do_networks,
                                               trial_similarity,
                                               RDM_method,
                                               age)
    df, ROI_to_bold_keys = organize_df(results_conn, results_bold, do_networks,
                                       target_ROI)

    lmer_stats(df, ROI_to_bold_keys, target_ROI)


def run_lmer_PFC_ERS():
    RSA = False
    semantic = False
    split = False
    four_tasks = '3_4'
    do_networks = 6
    target_ROI = 'Frontal_CG'
    trial_similarity = 'spear'
    RDM_method = 'clever_std'
    # RDM_method = 'clever_std_complex_mean'

    age = 1

    results_bold, results_conn = load_for_lmer(RSA, semantic, split,
                                               four_tasks, do_networks,
                                               trial_similarity,
                                               RDM_method,
                                               age)
    df, ROI_to_bold_keys = organize_df(results_conn, results_bold, do_networks,
                                       target_ROI)

    lmer_stats(df, ROI_to_bold_keys, target_ROI)


def run_lmer():
    RSA = False
    semantic = False
    split = False
    four_tasks = '3_4'
    do_networks = 6
    target_ROI = 'Frontal_CG'
    trial_similarity = 'corr'

    results_bold, results_conn = load_for_lmer(RSA, semantic, split,
                                               four_tasks, do_networks,
                                               trial_similarity)
    df, ROI_to_bold_keys = organize_df(results_conn, results_bold, do_networks,
                                       target_ROI)

    lmer_stats(df, ROI_to_bold_keys, target_ROI)

            

if __name__ == '__main__':
    # atlas = get_atlas(False, False)
    # print(atlas['tick_labels'])
    # quit()
    run_lmer_PFC_RSA()
    # run_lmer_PFC_ERS()

