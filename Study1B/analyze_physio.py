import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
from pkld import pkld
from tqdm import tqdm

from Study1B.analyze_plot_Fig3 import get_df_PE
from Study2B.analyze_Study2B import get_final_HCP_sns

# @pkld
def get_heart_rate(ppg_signal, downsample_rate=40):
    # 1. Bandpass filter to isolate heart rate range (0.5 - 3 Hz)
    fs = 400  # Sampling frequency
    lowcut, highcut = 0.5, 3.0
    b, a = signal.butter(3, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    filtered_signal = signal.filtfilt(b, a, ppg_signal)
    time = np.linspace(0, len(ppg_signal) / fs, len(ppg_signal))

    # 2. Detect peaks (heartbeats)
    peaks, _ = signal.find_peaks(filtered_signal, distance=fs // 2)  # Min 0.5 sec apart
    peak_times = time[peaks]

    # 3. Compute Instantaneous Heart Rate
    ibi = np.diff(peak_times)  # Time between peaks
    heart_rate = 60 / ibi  # Convert to bpm

    # 4. Interpolate to get HR at each time point
    hr_interp = interp.interp1d(peak_times[:-1], heart_rate, kind='linear', fill_value='extrapolate')
    hr_series = hr_interp(time)

    # 5. Downsample to 10 Hz
    # downsample_factor = fs // 10
    downsample_factor = downsample_rate
    time_downsampled = time[::downsample_factor]
    hr_downsampled = hr_series[::downsample_factor]
    return time_downsampled, hr_downsampled

@pkld
def get_physio(sn, lr):
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\tfMRI_GAMBLING_{lr}_Physio_log.txt'

    df = pd.read_csv(fp, sep='\t', names=['trigger', 'respitory', 'cardiac'])
    return df

@pkld
def get_physio_df(sn, lr='LR', downsample_rate=20,
                  wl='loss', plus=1):
    if wl == 'both':
        h_resp, l_resp, h_card, l_card = get_physio_df(sn, lr, downsample_rate, 'win')
        h_resp1, l_resp1, h_card1, l_card1 = get_physio_df(sn, lr, downsample_rate, 'loss')
        h_resp = (h_resp + h_resp1) / 2
        l_resp = (l_resp + l_resp1) / 2
        h_card = (h_card + h_card1) / 2
        l_card = (l_card + l_card1) / 2
        return h_resp, l_resp, h_card, l_card
    df = get_physio(sn, lr)
    # freq = 400
    df['time'] = df.index / 400

    time_downsampled, hr_downsampled = get_heart_rate(df['cardiac'],
                                                      downsample_rate=downsample_rate)
    time_downsampled, resp_downsampled = get_heart_rate(df['respitory'],
                                                        downsample_rate=downsample_rate)
    # print(time_downsampled)
    # quit()

    df = df.iloc[::downsample_rate]
    df['cardiac'] = hr_downsampled
    df['respitory'] = resp_downsampled

    df_rl, df_lr = get_df_PE(sn, 'both',
                             learning_rate=0.3,
                             drop_first=False,
                             reset_trial0=True)

    if lr == 'LR':
        df_trials = df_lr
    else:
        df_trials = df_rl

    cond2vals_respitory = {'high_PE': [], 'low_PE': []}
    cond2vals_cardiac = {'high_PE': [], 'low_PE': []}
    prev_card = None
    prev_resp = None
    # print(df_trials['event'])
    # quit()
    for i in range(len(df_trials)):
        if wl is not None:
            if df_trials['event'].iloc[i] != wl: continue
        onset = df_trials['onset'].iloc[i]
        # onset_baseline = onset - 3.5
        # onset += plus
        onset_next = onset + 3.5  # df_trials['onset'].iloc[i + 1]
        onset_next += plus
        # idx_baseline = int(onset_baseline * (400 / downsample_rate))
        idx_st = int(onset * (400 / downsample_rate))
        idx_end = int(onset_next * (400 / downsample_rate))
        cond = df_trials['trial_type'].iloc[i]
        if cond not in ['high_PE', 'low_PE']:
            continue
        card = df['cardiac'].iloc[idx_st:idx_end].mean()
        resp = df['respitory'].iloc[idx_st:idx_end].mean()
        # prev_card = df['cardiac'].iloc[idx_baseline:idx_st].mean()
        # prev_resp = df['respitory'].iloc[idx_baseline:idx_st].mean()
        # if prev_resp is None or prev_card is None:
        #     prev_resp = resp
        #     prev_card = card
        #     continue
        # card_dif = card - prev_card
        # resp_dif = resp - prev_resp
        # prev_resp = resp
        # prev_card = card

        cond2vals_cardiac[cond].append(card)
        cond2vals_respitory[cond].append(resp)

    h_resp = np.mean(cond2vals_respitory['high_PE'])
    l_resp = np.mean(cond2vals_respitory['low_PE'])
    h_card = np.mean(cond2vals_cardiac['high_PE'])
    l_card = np.mean(cond2vals_cardiac['low_PE'])
    return h_resp, l_resp, h_card, l_card

@pkld
def get_movement(sn, lr='LR'):
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\Movement_RelativeRMS.txt'
    movements = pd.read_csv(fp, sep='\t', names=['movement'])
    return movements

@pkld(overwrite=False)
def get_movement_df(sn, lr='LR', wl='loss', plus=1):
    if wl == 'both':
        h_movement, l_movement = get_movement_df(sn, lr, 'win')
        h_movement1, l_movement1 = get_movement_df(sn, lr, 'loss')
        h_movement = (h_movement + h_movement1) / 2
        l_movement = (l_movement + l_movement1) / 2
        return h_movement, l_movement
    movements = get_movement(sn, lr)

    df_rl, df_lr = get_df_PE(sn, 'both',
                             learning_rate=0.3,
                             drop_first=False,
                             reset_trial0=True)
    if lr == 'LR':
        df_trials = df_lr
    else:
        df_trials = df_rl
    cond2vals = {'high_PE': [], 'low_PE': []}
    for i in range(len(df_trials)):
        if wl is not None:
            if df_trials['event'].iloc[i] != wl: continue

        onset = df_trials['onset'].iloc[i]
        # onset += plus
        # onset_baseline = onset - 3.5
        onset_next = onset + 3.5 + plus # df_trials['onset'].iloc[i + 1]
        onset_idx = int(onset * 192 / 253 + .0001)
        onset_next_idx = int(onset_next * 192 / 253 + .0001)
        # print(f'{onset_idx}, {onset_next_idx}')
        cond = df_trials['trial_type'].iloc[i]
        if cond not in ['high_PE', 'low_PE']:
            continue
        cond2vals[cond].append(movements['movement'].iloc[onset_idx:onset_next_idx].mean())
    h_movement = np.mean(cond2vals['high_PE'])
    l_movement = np.mean(cond2vals['low_PE'])
    return h_movement, l_movement


if __name__ == '__main__':
    WL = 'both'
    PLUS = 0

    sns = get_final_HCP_sns()
    # get_movement_df(sns[0])

    # sns = sns[:100]
    cond2resp = {'high_PE': [], 'low_PE': []}
    cond2card = {'high_PE': [], 'low_PE': []}
    cond2move = {'high_PE': [], 'low_PE': []}
    sns_did = []
    for sn in tqdm(sns, desc='getting physio'):
        try:
            h_resp, l_resp, h_card, l_card = get_physio_df(sn, lr='LR', wl=WL,
                                                           plus=PLUS)
            h_resp1, l_resp1, h_card1, l_card1 = get_physio_df(sn, lr='RL', wl=WL,
                                                               plus=PLUS)
            h_resp = (h_resp + h_resp1) / 2
            l_resp = (l_resp + l_resp1) / 2
            h_card = (h_card + h_card1) / 2
            l_card = (l_card + l_card1) / 2
        except FileNotFoundError:
            print(f'File not found: {sn}')
            continue
        if np.isnan(h_resp) or np.isnan(l_resp) or np.isnan(h_card) or np.isnan(l_card):
            print(f'Nan: {sn}')
            continue
        h_move, l_move = get_movement_df(sn, lr='LR', wl=WL, plus=PLUS)
        h_move1, l_move1 = get_movement_df(sn, lr='RL', wl=WL, plus=PLUS)
        h_move = (h_move + h_move1) / 2
        assert not np.isnan(h_move) and not np.isnan(l_move), \
            f'{sn}: {h_move=}, {l_move=}'

        cond2resp['high_PE'].append(h_resp)
        cond2resp['low_PE'].append(l_resp)
        cond2card['high_PE'].append(h_card)
        cond2card['low_PE'].append(l_card)
        cond2move['high_PE'].append(h_move)
        cond2move['low_PE'].append(l_move)
        sns_did.append(sn)

    df = pd.DataFrame({'sn': sns_did,
                       'high_PE_resp': cond2resp['high_PE'],
                       'low_PE_resp': cond2resp['low_PE'],
                       'high_PE_card': cond2card['high_PE'],
                       'low_PE_card': cond2card['low_PE'],
                       'high_PE_move': cond2move['high_PE'],
                       'low_PE_move': cond2move['low_PE'],})
    print(f'{len(df)=}')

    cols = ['high_PE_resp', 'low_PE_resp', 'high_PE_card', 'low_PE_card',
            'high_PE_move', 'low_PE_move']
    p_higher_move = np.mean(df['high_PE_move'] > df['low_PE_move'])
    print(f'{p_higher_move=:.2%}')
    p_higher_heart = np.mean(df['high_PE_card'] > df['low_PE_card'])
    print(f'{p_higher_heart=:.2%}')
    p_higher_resp = np.mean(df['high_PE_resp'] > df['low_PE_resp'])
    print(f'{p_higher_resp=:.2%}')

    for col in cols:
        M = df[col].mean()
        SE = stats.sem(df[col])
        print(f'{col}: {M=:.3f} {SE=:.4f}')

    t_resp, p_resp = stats.ttest_rel(df['high_PE_resp'], df['low_PE_resp'])
    t_card, p_card = stats.ttest_rel(df['high_PE_card'], df['low_PE_card'])
    t_move, p_move = stats.ttest_rel(df['high_PE_move'], df['low_PE_move'])
    print(f'{t_resp=:.2f} {p_resp=:.4f}')
    print(f'{t_card=:.2f} {p_card=:.4f}')
    print(f'{t_move=:.2f} {p_move=:.4f}')
