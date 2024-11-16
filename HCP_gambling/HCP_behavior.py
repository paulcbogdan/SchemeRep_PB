import os

import pandas as pd
import scipy.stats as stats
import numpy as np

from Study1B.preprocess_Study1B import get_df_PE

def semi_do_bhv_df(sn, lr, cont_PE=.3):
    try:
        df_trials = get_df_PE(sn, lr, learning_rate=cont_PE)
    except FileNotFoundError:
        print(f'No win.txt or loss.txt: {sn}, {lr}')
        return None
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


def get_gambling_behavior_good(sn, key='RT_next', cont_PE=0.3,
                               ctrl_trial_num_close=True):
    df_lr = semi_do_bhv_df(sn, 'LR', cont_PE=cont_PE)
    if df_lr is None:
        return None, None, None, None
    df_rl = semi_do_bhv_df(sn, 'RL', cont_PE=cont_PE)
    if df_rl is None:
        return None, None, None, None
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
        if ctrl_trial_num_close:
            df_PE_win = df_win.groupby(['PE', 'trial_within_block'])[key].mean()

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
        else:
            df_PE_win = df_win.groupby('PE')[key].mean()
            df_PE_loss = df_loss.groupby('PE')[key].mean()

        PE_ef_win = df_PE_win['high_PE'] - df_PE_win['low_PE']
        PE_ef_loss = df_PE_loss['high_PE'] - df_PE_loss['low_PE']

        high_PE_M = (df_PE_win['high_PE'] + df_PE_loss['high_PE']) / 2
        low_PE_M = (df_PE_win['low_PE'] + df_PE_loss['low_PE']) / 2

    except KeyError:
        return None, None, None, None
    PE_ef = PE_ef_loss

    return p_change, PE_ef, high_PE_M, low_PE_M



def do_HCP_bhv_analysis():
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')

    efs_l = []
    high_PE_l = []
    low_PE_l = []
    for sn in sns:
        p_change, sn_ef, high_PE, low_PE = (
            get_gambling_behavior_good(sn, ))
        if sn_ef is None:
            continue
        assert not np.isnan(sn_ef)
        efs_l.append(sn_ef)

        N = np.sum(~np.isnan(efs_l))
        t, p = stats.ttest_1samp(efs_l, 0)
        print(f't[{N-1}] = {t=:.2f}, {p=:.4f}')

        low_PE_l.append(low_PE)
        high_PE_l.append(high_PE)
        low_M = np.mean(low_PE_l)
        low_SD = np.std(low_PE_l)
        high_M = np.mean(high_PE_l)
        high_SD = np.std(high_PE_l)
        print(f'\t{high_M=:.2f} [{high_SD=:.2f}], {low_M=:.2f} [{low_SD=:.2f}]')
        print()

if __name__ == '__main__':
    do_HCP_bhv_analysis()
