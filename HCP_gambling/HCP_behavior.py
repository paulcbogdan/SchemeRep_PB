import os

import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import numpy as np

from HCP_gambling.preproc_gambling import get_df_events


def get_gambling_behavior_df(sn, lr, key='RT_next'):
    df_trials = get_df_events(sn, lr, cont_pe_by_event=True)
    run_num = 2 if lr == 'LR' else 1
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\GAMBLING_run{run_num}_TAB.txt'
    if not os.path.exists(fp):
        print(f'No file: {fp}')
        return None, None
    df = pd.read_csv(fp, delimiter='\t')
    df = df[df['Procedure[Trial]'] == r'GamblingTrialPROC'].reset_index()
    n_nans = df['QuestionMark.RESP'].isna().sum()
    df.dropna(subset=['QuestionMark.RESP', 'QuestionMark.RT'],
              inplace=True)

    df['RESP_next'] = df['QuestionMark.RESP'].shift(-1)
    df['RESP_CHANGE'] = df['QuestionMark.RESP'] != df['RESP_next']
    p_change = df['RESP_CHANGE'].sum() / df['RESP_CHANGE'].count()

    df['RT'] = df['QuestionMark.RT']
    df['RT_next'] = df['RT'].shift(-1)
    df['RT_slower'] = df['RT_next'] - df['RT']
    df['PE'] = df_trials['trial_type']
    df['event'] = df_trials['event']

    # df = df.groupby('PE')['RT_next'].mean()

    df_win = df[df['event'] == 'win']
    df_loss = df[df['event'] == 'loss']
    try:
        df_PE_win = df_win.groupby('PE')['RT_next'].mean()
        df_PE_loss = df_loss.groupby('PE')['RT_next'].mean()
        PE_ef_win = df_PE_win['high_PE'] - df_PE_win['low_PE']
        PE_ef_loss = df_PE_loss['high_PE'] - df_PE_loss['low_PE']
    except KeyError:
        return None, None
    PE_ef = PE_ef_win + PE_ef_loss
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
    for sn in tqdm(sns, desc='looping through behavior sns'):
        sn_efs = []
        for lr in ['LR', 'RL']:
            p_change, PE_ef = get_gambling_behavior_df(sn, lr)
            if PE_ef is None:
                continue
            sn_efs.append(PE_ef)
        efs_l.append(np.nanmean(sn_efs))
        N = np.sum(~np.isnan(efs_l))
        t, p = stats.ttest_1samp(efs_l, 0, nan_policy='omit')
        print(f't[{N-1}] = {t=:.2f}, {p=:.4f}')


            # if df is None:
            #     continue


if __name__ == '__main__':
    do_HCP_bhv_analysis()
