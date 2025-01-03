from colorama import Fore

from connRSA.old.conn_plot import pie_charts
from collections import defaultdict

from conn_utils import get_BNA_ROIs
from organize_bhv import get_trial_info

import pandas as pd
import numpy as np
from tqdm import tqdm

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


def roimap2np(ROI2ar, only_some=None):
    l = []
    for ROI, ar in ROI2ar.items():
        if only_some:
            for ROI_ in only_some:
                if ROI_ in ROI:
                    break
            else:
                continue
        l.append(ar)
    # print(np.array(l).shape)
    l = np.array(l)
    print(f'Data shape: {l.shape}')
    return l


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

    cols_keep = ['ROI_M', 'BOLD_score', 'conn_score', 'sn',
                 'fp_idx', 'hit_hit', 'con_hit', 'vis_hit']



    # plt.hist(df['ROI_M'])



    # df['ROI_M'] = df[ROI_cols].apply(rsum, axis=1)

    df['ROI_M'] = df[ROI_cols].mean(axis=1)





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
