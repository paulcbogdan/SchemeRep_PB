import os

import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import numpy as np

from HCP_gambling.preproc_gambling import get_df_events

def semi_do_bhv_df(sn, lr, cont_PE=.3):
    df_trials = get_df_events(sn, lr, cont_pe_by_event=True, cont_PE=cont_PE)
    run_num = 2 if lr == 'LR' else 1
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\GAMBLING_run{run_num}_TAB.txt'
    if not os.path.exists(fp):
        print(f'No file: {fp}')
        return None
    df = pd.read_csv(fp, delimiter='\t')
    df = df[df['Procedure[Trial]'] == r'GamblingTrialPROC'].reset_index()
    df['trial_within_block'] = list(range(8)) * 4

    n_nans = df['QuestionMark.RESP'].isna().sum()
    df.dropna(subset=['QuestionMark.RESP', 'QuestionMark.RT'],
              inplace=True)

    df['RESP_next'] = df['QuestionMark.RESP'].shift(-1)
    df['RESP_CHANGE'] = df['QuestionMark.RESP'] != df['RESP_next']

    df['RT'] = df['QuestionMark.RT']
    df['RT_next'] = df['RT'].shift(-1)
    df['PE'] = df_trials['trial_type']
    df['event'] = df_trials['event']

    return df


def get_gambling_behavior_df(sn, lr, key='RT_next', cont_PE=0.3):
    df_lr = semi_do_bhv_df(sn, 'LR', cont_PE=cont_PE)
    if df_lr is None:
        return None, None
    df_rl = semi_do_bhv_df(sn, 'RL', cont_PE=cont_PE)
    if df_rl is None:
        return None, None
    df = pd.concat([df_lr, df_rl], ignore_index=True)

    p_change = df['RESP_CHANGE'].sum() / df['RESP_CHANGE'].count()

    df['RT_slower'] = df['RT_next'] - df['RT']


    df = df[df['trial_within_block'] != 0] # drop first of block since they're all slower
    df = df[df['trial_within_block'] != 7] # drop first of next block

    df_win = df[df['event'] == 'win']
    df_loss = df[df['event'] == 'loss']
    # PE_ef = df_win[key].mean() - df_loss[key].mean()
    # print(df[['RT', 'RT_next', 'event', 'PE']])
    # quit()
    try:
        df_PE_win = df_win.groupby(['PE', 'trial_within_block'])[key].mean()
        # print(('high_PE', 3) in df_PE_win)
        # print(df_PE_win)
        for trial in range(1, 7):
            if ('high_PE', trial) not in df_PE_win.index:
                df_PE_win.loc[('low_PE', trial)] = np.nan
            if ('low_PE', trial) not in df_PE_win.index:
                df_PE_win.loc[('high_PE', trial)] = np.nan
        df_PE_win.dropna(inplace=True)
        df_PE_win = df_PE_win.groupby('PE').mean()

        df_PE_loss = df_loss.groupby(['PE', 'trial_within_block'])[key].mean()
        # print(df_PE_loss)
        # quit()
        for trial in range(1, 7):
            if ('high_PE', trial) not in df_PE_loss.index:
                df_PE_loss.loc[('low_PE', trial)] = np.nan
            if ('low_PE', trial) not in df_PE_loss.index:
                df_PE_loss.loc[('high_PE', trial)] = np.nan
        df_PE_loss.dropna(inplace=True)
        df_PE_loss = df_PE_loss.groupby('PE').mean()
        # df_PE_loss = df_loss.groupby('PE')[key].mean()

        # print(df_PE_win)

        PE_ef_win = df_PE_win['high_PE'] - df_PE_win['low_PE']
        PE_ef_loss = df_PE_loss['high_PE'] - df_PE_loss['low_PE']

    except KeyError:
        return None, None
    PE_ef = PE_ef_loss + PE_ef_win
    # print(f'{PE_ef_win=}')
    # print(f'{PE_ef_loss=}')
    # quit()
    # PE_ef = df['RT'].mean()
    print(f'{PE_ef=:.3f}')

    return p_change, PE_ef


def get_task_sns(reg_global=True, no_compcor=True):
    fns_task = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    if reg_global:
        fns_task = [fn for fn in fns_task if 'global' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'global' not in fn]
    if no_compcor:
        fns_task = [fn for fn in fns_task if 'nocc' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'nocc' not in fn]
    fns_task_LR = [fn for fn in fns_task if 'LR' in fn]
    fns_task_RL = [fn for fn in fns_task if 'RL' in fn]
    sns_task_LR = {fn.split('_')[0] for fn in fns_task_LR}
    sns_task_RL = {fn.split('_')[0] for fn in fns_task_RL}
    sns_task = sns_task_RL.intersection(sns_task_LR)
    return sns_task

def do_HCP_bhv_analysis():
    sns = get_task_sns(reg_global=True, no_compcor=True)
    efs_l = []
    M_m = []
    for sn in tqdm(sns, desc='looping through behavior sns'):
        sn_efs = []
        p_change, sn_efs = get_gambling_behavior_df(sn, None)
        if sn_efs is None:
            continue
        # for lr in ['LR', 'RL']:
        #     p_change, PE_ef = get_gambling_behavior_df(sn, lr)
        #     if PE_ef is None:
        #         continue
        #     sn_efs.append(PE_ef)
        efs_l.append(np.nanmean(sn_efs))
        M = np.nanmean(sn_efs)
        M_m.append(M)
        N = np.sum(~np.isnan(efs_l))
        t, p = stats.ttest_1samp(efs_l, 0, nan_policy='omit')
        GM = np.nanmean(M_m)
        print(f't[{N-1}] = {t=:.2f}, {p=:.4f} | {GM=:.3f}')

if __name__ == '__main__':
    do_HCP_bhv_analysis()
