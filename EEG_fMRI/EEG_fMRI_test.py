import os.path

from nilearn import image
from atlas_utils import get_atlas
from get_HCP_act import img_data2ar
from mne.io import read_raw_eeglab
from utils import pickle_wrap, stdize
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.io import loadmat

# warnings.filterwarnings('ignore', #category=RuntimeWarning,
#                         )

# message='were expanding outside')
# message='indicating data discontinuities'

ROOT_EEG_FMRI = fr'G:\EEG_fMRI'
# ROOT_EEG_FMRI = fr'C:\Users\Paul\Downloads'

def get_fMRI_score_sn(sn, sess='01', combine_regions=True):
    # root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess}'

    dir_func = fr'{root_sn}\func\sub-{sn}_ses-{sess}_task-rest_bold\func_preproc'
    fp_fMRI = fr'{dir_func}\func_pp_filter_sm0.mni152.3mm.nii.gz'
    if os.path.isfile(fp_fMRI):
        img = image.load_img(fp_fMRI)
    else:
        print(f'No file: {fp_fMRI=}')
        return None
    # img = pickle_wrap(image.load_img, kwargs={'img': fp_fMRI}) # 0.476 Hz
    data_fMRI = img.get_fdata()
    # print(data_fMRI.shape)
    # quit()
    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    key2idxs = {'ATL': [], 'MFG': [], 'IPL': [], 'LOC': []}
    for i, region in enumerate(atlas['ROI_regions']):
        for key, l in key2idxs.items():
            if key in region:
                l.append(i)

    ar_fMRI = img_data2ar(data_fMRI, atlas)
    ar_fMRI = stdize(ar_fMRI, axis=-1)
    # double check for GSR

    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]
    # pairs = [('ATL', 'MFG'), ('ATL', 'IPL'), ('ATL', 'LOC'), ]
    ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['MFG'])].mean(axis=(0, 1))
    ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))
    MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'],
                               key2idxs['IPL'])].mean(axis=(0, 1))
    IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))
    fluc = MFG_IPL + ATL_LOC - ATL_MFG - IPL_LOC
    return fluc

def get_event2true(dir_eeg, sn, fp_EEG):
    events = pd.read_csv(fr'{dir_eeg}\sub-{sn}_ses-01_task-rest_events.tsv',
                         sep='\t')

    mat = loadmat(fp_EEG)
    urevent = mat['urevent'][0]
    assert len(urevent[0]) == 7
    for event in urevent:
        if event[-2][0] == 'S  1':
            # t_st = event[0][0][0]
            frame_S1 = event[0][0][0]
            break
            # print(f'{t_st=}')
            # break
    else:
        raise ValueError
    # print(urevent)
    # quit()

    # max_onset = events['onset'].max()
    # total_duration = events.loc[events['value'] == 'boundary']['duration'].sum()
    # if not ('S  1' in events['value'].values):
    #     print('No S  1: BAD!')
    #     return None, None

    # print(f'{max_onset=:.3f}, {total_duration=:.3f}')
    # quit()

    events = events.iloc[1:]
    num2s = 0
    t_prev = 0
    start = False
    cnt = 0
    # S 1 = starting stimulus
    t_st = 0
    cnt_event = 0
    event2true = {}
    boundary_events = set()
    duration_so_far = 0
    last_cnt = 0
    for idx, event in events.iterrows():
        duration_so_far += event['duration']

        if not start and (250 * (duration_so_far + event['onset']) >
                          frame_S1 - 100):
            start = True
            # print(event['value'])
            # print(f'{event=}')
        # print(f'{duration_so_far=}')
        # if event['value'] == 'S  1':
        #     start = True

        if event['type'] == 'Response':
            num2s += 1
            t_prev = event['onset']
            event2true[cnt_event] = cnt
            cnt_event += 1

            cnt += 1
            last_cnt = cnt

        t_st += event['duration']

        if start and (event['value'] == 'boundary'):# and event['sample'] > 10:
            boundary_events.add(cnt_event)
            boundary_events.add(cnt_event + 1)
            # print(event)
            t_effect = event['onset'] - t_prev + event['duration']
            for i in range(int(t_effect / 2.1)):
                cnt += 1
    assert last_cnt == 288
    # print(f'{last_cnt=}')
    # quit()
    return event2true, boundary_events

def get_EEG_score_sn(sn, sess='01'):
    # root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'

    dir_eeg = fr'{root_sn}\eeg'
    fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-01_task-rest_eeg.set'


    # print(list(mat))
    # print(mat['urevent'])
    # quit()
    # print(mat['event'])
    # for event in mat['urevent'][0]:
    #     # print(event[-1])
    #     # print(len(event[-1][0]))
    #     print(event)
        # if len(event[-1][0]) == 0:
        #     break
        # quit()
    # print(f'{fp_EEG=}')
    # quit()

    if not os.path.isfile(fp_EEG):
        print(f'No file: {fp_EEG=}')
        return None
    raw = read_raw_eeglab(fp_EEG, preload=True, )
    event2true, boundary_events = get_event2true(dir_eeg, sn, fp_EEG)
    if event2true is None:
        return None

    events = mne.events_from_annotations(raw, verbose=False)
    events = events[0]
    data = raw.get_data()
    # print(f'{data.shape=}')

    events = events[events[:, 2] == 2]

    # print(raw.ch_names)
    # quit()
    picks = ['Fz', 'Cz', 'Pz',
             'F1', 'C1', 'P1',
             'F2', 'C2', 'P2']
    picks = [pick for pick in picks if pick in raw.ch_names]
    # picks = ['Fz', 'Pz'] # 'Cz',
    # picks = raw.ch_names[:5]
    # picks = 'all'
    # print(raw)
    data_eeg = raw.get_data(picks=picks)
    freqs = np.arange(1, 101, 1)
    tfr = mne.time_frequency.tfr_array_morlet(data_eeg[None, :, :],
                                              sfreq=250, freqs=freqs,
                                              output='power')
    # print('made TFR')
    tfr = tfr[0, ...]
    tfr = np.log(tfr)
    tfr = tfr.mean(axis=0) # avg Fz, Cz, Pz

    eeg_scores = np.full((288, len(freqs)), np.nan)
    for idx, event in enumerate(events):
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return None
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1*250)
        tfr_event = tfr[..., t_st:t_end].mean(axis=-1)
        eeg_scores[true_idx] = tfr_event

    return eeg_scores

def print_events(raw):
    events = mne.events_from_annotations(raw) # 250 Hz
    # print(events[1])
    events = events[0]
    # events = np.array(events).astype(float)
    # events[:, 0] /= 250

    events = np.array(events)
    events = events[np.isin(events[:, 2], [7])]
    # print(events)
    # quit()

    for i, event in enumerate(events[:-1]):
        next_event = events[i + 1]
        if next_event[2] == 7:
            next_event = events[i + 2]
            # print('blah')
        dif = next_event[0] - event[0]
        if dif != 525 and event[2] == 2:
            print(f'{event[0]} | {event[1]} | {event[2]} | {dif=} *** ')
        else:
            print(f'{event[0]} | {event[1]} | {event[2]} | {dif=}')
    # print(events)
    # print(events.shape)

    # There are only 280 sync events (event code = 2).
    #   I don't understand what the "sync on" (code = 6) means, given
    #  that it's not 2.1 s apart like the event code = 2 events are.
    #  The code = 6 events are 2 s apart and there are 300 of them.

def test_EEG_fMRI_sn(sn='06', sess='01'):
    fMRI_fluc = pickle_wrap(get_fMRI_score_sn, kwargs={'sn': sn,
                                                       'sess': sess},
                            easy_override=False, verbose=-1)
    if fMRI_fluc is None:
        return None
    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs={'sn': sn,
                                                     'sess': sess},
                           easy_override=True, verbose=-1)

    if EEG_fluc is None:
        return None

    ranges = {'delta': (1, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 100)}
    name2fluc = {}
    name2r = {}
    for name, rng in ranges.items():
        idxs = np.arange(*rng)
        fluc = EEG_fluc[:, idxs].mean(axis=1)
        name2fluc[name] = fluc
        r, p = stats.spearmanr(fluc, fMRI_fluc, nan_policy='omit')
        print(f'{name}: {r=:.3f}, {p=:.3f}')
        name2r[name] = r
    return name2r


if __name__ == '__main__':
    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    SNS = ['01']
    NAME2L = defaultdict(list)
    for SN in SNS:
        print(f'- ({SN}) -')
        name2r = test_EEG_fMRI_sn(SN)
        if name2r is None:
            continue
        print()
        for key, r in name2r.items():
            NAME2L[key].append(r)
            l = NAME2L[key]
            N = len(l)
            if N > 5:
                M = np.mean(l)
                SE = np.std(l) / np.sqrt(N)
                t = M / SE
                p = stats.t.sf(np.abs(t), len(l) - 1)
                print(f'{key} ({N=}): {M=:.3f} ({t=:.3f})')

    # for key, l in NAME2L.items():
    #     M = np.mean(l)
    #     SE = np.std(l) / np.sqrt(len(l))
    #     t = M / SE
    #     p = stats.t.sf(np.abs(t), len(l) - 1)
    #     print(f'{key}: {M=:.3f} ({p=:.3f})')



