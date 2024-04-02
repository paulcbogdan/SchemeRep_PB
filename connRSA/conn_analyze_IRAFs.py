import os

from connRSA.conn_ISPC import run_settings_ISPC
from corr_RSA_x_vendor import get_plain_df_sn

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

import scipy.stats as stats
from colorama import Fore
import statsmodels.formula.api as smf
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

def load_for_lmer_ISPC(four_tasks='8', conn='euc', combine_regions=False,
			 split=False, do_networks=0, trial_similarity='corr',
			 age=1):
    settings = locals().copy()
    dir_results = r'cache/conn_RSA'
    results_conn = pickle_wrap(run_settings_ISPC, None, kwargs=settings,
                               easy_override=False, cache_dir=dir_results)
    report_results(results_conn, do_lmer=True, ISPC=True)

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        settings['RDM_method'] = 'within_nan'
    #
    settings['conn'] = 'BOLD'
    results_bold_comb = pickle_wrap(run_settings_ISPC, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)

    report_results(results_bold_comb, do_lmer=True, ISPC=True)

    results_bold_sep = {}

    # settings['do_networks'] = False
    # results_bold_sep = pickle_wrap(run_settings_ISPC, None,
    #                                kwargs=settings, easy_override=False,
    #                                verbose=1, cache_dir=dir_results)

    # report_results(results_bold_sep, do_lmer=False)


    return results_conn, results_bold_comb, results_bold_sep

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

    report_results(results_conn, do_lmer=False)#True)
    # quit()

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        # settings['RDM_method'] = 'clever_std'
        settings['RDM_method'] = 'within_nan'

    settings['conn'] = 'BOLD'
    results_bold_comb = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)
    print(f'BOLD ' * 10)
    report_results(results_bold_comb, do_lmer=True)

    settings['do_networks'] = False
    results_bold_sep = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)

    # report_results(results_bold_sep, do_lmer=False)


    return results_conn, results_bold_comb, results_bold_sep

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
    assert target_ROI in network2ROI.keys(), \
        f'{target_ROI} not in settings choice'

    BOLD_keys = get_BNA_ROIs()
    key2short = {key: key.split(' ')[1] for key in BOLD_keys}
    BOLD_keys_short = [key2short[key] for key in BOLD_keys]

    return network2ROI, BOLD_keys_short


def lmer_stats(df, ROI_cols):
    from pymer4.models import Lmer

    # conn_sess_ERS = ['2', ] # '4', '5'
    # df = df[df['fp_idx'].isin(conn_sess_ERS)]
    # df = df[df['fp_idx'].isin(['1'])]
    #
    # n_sn = df['sn'].nunique()
    #
    # df_M = df.groupby('sn')['conn_score'].mean()
    # M_score = df_M.mean()
    # SD_score = df_M.std()
    # # print(np.sum(~np.isnan(df_M)))
    # # quit()
    # N_score = np.sum(~np.isnan(df_M))
    # SE_score = SD_score / np.sqrt(N_score)
    # t_score = M_score / SE_score
    #
    # print(f'Single FP: {M_score=:.3f}, {SD_score=:.3f}, {SE_score=:.3f}, '
    #       f'{t_score=:.3f}')
    #
    # cols = ['ROI', 'sn', 'conn_score', 'hit_hit', 'inc', 'vis_hit', 'con_hit',
    #         'inc_str', 'per_inc_str', 'fp_idx', 'BOLD_score'] + \
    #          ROI_cols
    # df = df[cols]
    # df.dropna(inplace=True)
    # df['vis_hit'] = df['vis_hit'].astype(int)
    # df['con_hit'] = df['con_hit'].astype(int)
    #
    #
    #
    # if len(df['fp_idx'].unique()) > 1:
    #     formula = 'conn_score ~ 1 + BOLD_score + (1|sn) + (1|fp_idx)'
    #     print('tteet')
    # else:
    #     formula = 'conn_score ~ 1 + BOLD_score + (1|sn)'
    #     print('toast')
    # # quit()
    #
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    #
    # print('-' * 120)
    # formula = ('conn_score ~ 1 + BOLD_score + ' +
    #            ' + '.join(ROI_cols) + '+  (1|sn)')# + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    #
    # formula = ('BOLD_score ~ 1 + ' +
    #            ' + '.join(ROI_cols) + '+  (1|sn)')# + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())

    # df['BOLD_score'] = df['conn_score']
    cols_keep = ['ROI_M', 'BOLD_score', 'conn_score', 'sn',
                 'fp_idx', 'hit_hit', 'con_hit', 'vis_hit']

    df['ROI_M'] = df[ROI_cols].mean(axis=1)
    df = df[cols_keep].dropna()

    mod = smf.ols(formula='BOLD_score ~ 1 + mem', data=df_)
    res = mod.fit()
    print(res.summary())





    mem_cols = ['hit_hit', 'con_hit', 'vis_hit']
    df[mem_cols] = df[mem_cols].astype(int)

    formula = ('BOLD_score ~ 1 + ROI_M + (1 + ROI_M |sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('conn_score ~ 1 + BOLD_score + ROI_M + '
               '(1 + BOLD_score + ROI_M | sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())
    quit()

    formula = ('hit_hit ~ 1 + BOLD_score*fp_idx +  (1 |sn)')# + (1|fp_idx)' ROI_M*fp_idx  +
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('con_hit ~ 1 + BOLD_score*fp_idx + (1 |sn)')# + (1|fp_idx)' + ROI_M*fp_idx
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('vis_hit ~ 1 + BOLD_score*fp_idx + (1 |sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    # print('-' * 120)
    # formula = 'conn_score ~ 1 + hit_hit + (1|sn) + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=True, summary=True)
    # print(model.summary())
    #
    # print('-' * 120)
    # formula = 'conn_score ~ 1 + con_hit + (1|sn) + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=True, summary=True)
    # print(model.summary())


def run_lmer_PFC_RSA():
    RSA = False
    semantic = False
    conn = 'euc'
    trial_similarity = 'corr'
    second_order = 'spear'
    four_tasks = '8'
    combine_regions = False
    split = False
    RDM_method = 'clever_std_complex_mean' # clever_std_complex_mean
    # RDM_method = 'within_nan'
    # RDM_method = 'clever_std'
    age = 'healthy'
    stdize_by_run = False

    # target_ROI = 'else_cortical'
    # target_ROI = 'perceptual'
    target_ROI = 'Ventral'

    ROI2network = {'Occipital': 1, 'Ventral': 1, 'Dorsal': 1,
                   'else_cortical': 2,
                   'PFC': 16, 'PFC_ACC': 14, 'FP': 14,
                   'perceptual': 17}
    do_networks = ROI2network[target_ROI]

    kwargs = {'RSA': RSA, 'semantic': semantic, 'do_networks': do_networks,
                'conn': conn, 'trial_similarity': trial_similarity,
                'second_order': second_order, 'four_tasks': four_tasks,
                'combine_regions': combine_regions, 'split': split,
                'RDM_method': RDM_method, 'age': age,
                'stdize_by_run': stdize_by_run
              }
    results_conn, results_bold_comb, results_bold_sep \
        = pickle_wrap(load_for_lmer, None, kwargs=kwargs,)

    # results_conn, results_bold_comb, results_bold_sep = (
    #     load_for_lmer(RSA=RSA, semantic=semantic, do_networks=do_networks,
    #                   conn=conn, trial_similarity=trial_similarity,
    #                   second_order=second_order, four_tasks=four_tasks,
    #                   combine_regions=combine_regions, split=split,
    #                   RDM_method=RDM_method,age=age,
    #                   stdize_by_run=stdize_by_run))

    df, ROI_cols = organize_df(results_conn, results_bold_sep,
                               results_bold_comb, do_networks, target_ROI)

    lmer_stats(df, ROI_cols)

def run_lmer_PFC_ISPC():
    do_networks = 17
    # target_ROI = 'else'

    df_bhv, _ = get_plain_df_sn()
    df_bhv = df_bhv.groupby('obj')['hit_hit'].mean()
    mem_scores = list(df_bhv.values) * 4


    results_conn, results_bold_comb, results_bold_sep = (
        load_for_lmer_ISPC(four_tasks='7', conn='euc', combine_regions=False,
                           split=False, do_networks=do_networks,
                           trial_similarity='corr',
                           age='healthy'))

    idx = 0
    results_conn['scores_by_ROI'] = results_conn['scores_by_ROI'][:, idx, :]
    results_bold_comb['scores_by_ROI'] \
        = results_bold_comb['scores_by_ROI'][:, idx, :]

    # networks = {'PFC': ['SFG', 'MFG', 'IFG', 'OrG'],
    #             'PFC_ACC': ['SFG', 'MFG', 'IFG', 'OrG', 'ACC'],
    #             'FP': ['MFG', 'IFG', 'IPL', 'SPL']}



    d = {'conn_score': np.reshape(results_conn['scores_by_ROI'], -1),
         'BOLD_score': np.reshape(results_bold_comb['scores_by_ROI'], -1),
         'fp_idx': list(range(4)) * 114,
         'img': list(range(114)) * 4,
         'mem': mem_scores}

    df = pd.DataFrame(d)
    df['fp_idx'] = df['fp_idx'].astype(str)
    # df = df.groupby('img').mean()
    # df['BOLD_score'] = stats.zscore(df['BOLD_score'])
    # df['conn_score'] = stats.zscore(df['conn_score'])

    for fp_idx in range(4):
        df_ = df[df['fp_idx'] == str(fp_idx)]
        df_['mem'] = stats.zscore(df_['mem'])
        mod = smf.ols(formula='BOLD_score ~ 1 + mem', data=df_)
        res = mod.fit()
        print(res.summary())
        # quit()
    quit()
    mod = smf.ols(formula='conn_score ~ 0 + mem * fp_idx', data=df)
    res = mod.fit()
    print(res.summary())

    # df = df.groupby('img')[['conn_score', 'BOLD_score']].mean()
    # mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    # res = mod.fit()
    # print(res.summary())
    quit()

    formula = 'conn_score ~ 1 + BOLD_score + (1|fp_idx)'
    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())

    # scores_conn = results_conn['scores_by_ROI']
    # print(f'{scores_conn.shape=}')
    # scores_conn = np.array(scores_conn)
    # # if not ISPC:
    # scores_sep = results_bold_sep['scores_by_ROI']
    # scores_sep = np.array(scores_sep)
    # scores_comb = results_bold_com['scores_by_ROI']


    # print(df)
    # print(results_conn['scores_by_ROI'].shape)
    quit()

    # scores_by_ROI = np.array(results['scores_by_ROI'])
    # scores_by_fp = np.transpose(scores_by_ROI, (0, 2, 1))

    # df, ROI_cols = organize_df(results_conn, results_bold_sep,
    #                            results_bold_comb, do_networks, target_ROI,
    #                            ISPC=True)
    #
    # lmer_stats(df, ROI_cols)

if __name__ == '__main__':
    # atlas = get_atlas(False, False)
    # print(atlas['tick_labels'])
    # quit()
    run_lmer_PFC_RSA()
    # run_lmer_PFC_ISPC()
