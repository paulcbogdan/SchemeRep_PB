import os
import warnings
from time import sleep

import mne
import numpy as np
from mne.io import read_raw_eeglab
from scipy.io import loadmat

from Utils.pickle_wrap_funcs import pickle_wrap

ROOT_EEG_FMRI = fr'F:\EEG_fMRI'


def get_event2true(fp_EEG, num_TRs, excl_before=True):
    mat = loadmat(fp_EEG)
    R128_cnt = 0
    urevent2TR = {}
    for event in mat['urevent'][0]:
        if len(event[4][0]):
            urevent_num = event[4][0][0]

            assert len(event) in [7, 8], f'{len(event)=}'

            name = event[-2][0]
            if name == 'R128':
                urevent2TR[urevent_num] = R128_cnt
                R128_cnt += 1
    assert len(urevent2TR) == num_TRs, f'{len(urevent2TR)=} | {num_TRs=}'

    mat = loadmat(fp_EEG)
    event2true = {}
    good_events = set()
    boundary_events = set()
    R128_pos_cnt = 0
    for event in mat['event'][0]:
        if len(event[4][-1]) and event[-3][0] == 'R128':
            urevent_num = event[4][-1][0]
            event2true[R128_pos_cnt] = urevent2TR[urevent_num]
            good_events.add(event2true[R128_pos_cnt])

            R128_pos_cnt += 1

    for i in range(num_TRs):
        if (i + 1 not in good_events and
                (excl_before and (i - 1) in good_events)):
            boundary_events.add(i)
            boundary_events.add(i + 1)
    return event2true, boundary_events


def load_EEG(sn, sess, num_TRs, dir_eeg):
    fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-{sess}_eeg.set'

    if not os.path.isfile(fp_EEG):
        print(fr'Seemingly no file: {fp_EEG=}')
        sleep(1)
        if os.path.isfile(fp_EEG):
            print('\tFile found after 1 s pause')
        else:
            print('\tConfirmed no file')
            return None, None, None
    try:
        raw = read_raw_eeglab(fp_EEG, preload=True, )
    except OSError as e:
        warnings.warn(rf'OSError: {sn} ({sess}) {e=} | {fp_EEG=}')
        return None, None, None
    try:
        event2true, boundary_events = get_event2true(fp_EEG, num_TRs, )
    except AssertionError as e:
        print(f'{sn} | {e=}')
        return None, None, None
    if event2true is None:
        return None, None, None

    return raw, event2true, boundary_events


def get_EEG_score_sn(sn, num_TRs, sess='01', picks=None,
                     custom_freqs=None):
    if picks is None:
        picks = ['Fz', 'Cz', 'Pz',
                 'F1', 'C1', 'P1',
                 'F2', 'C2', 'P2']
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_eeg = fr'{root_sn}\eeg'

    kw = {'sn': sn, 'sess': sess, 'num_TRs': num_TRs, 'dir_eeg': dir_eeg, }
    raw, event2true, boundary_events = pickle_wrap(load_EEG, kwargs=kw,
                                                   easy_override=False)
    if raw is None:
        return None

    raw = raw.set_eeg_reference('average')

    events = mne.events_from_annotations(raw, verbose=False)
    events = events[0]
    events = events[events[:, 2] == 2]

    ds1 = 1

    if custom_freqs is not None:
        freqs = list(custom_freqs)
    else:
        freqs = np.linspace(0.5, 50, 100)

    picks_pruned = [pick for pick in picks if pick in raw.ch_names]
    data_eeg = raw.get_data(picks=picks_pruned)
    data_eeg = data_eeg[None, :, :]

    tfr = mne.time_frequency.tfr_array_morlet(data_eeg[..., ::ds1],
                                              sfreq=250 // ds1,
                                              freqs=freqs,
                                              output='power')

    tfr = tfr[0, ...]
    eeg_scores = np.full((len(picks_pruned), len(freqs), num_TRs,),
                         np.nan)

    for idx, event in enumerate(events):
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return None
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1 * (250 // ds1))

        tfr_event = tfr[..., t_st:t_end].mean(axis=-1)

        eeg_scores[:, :, true_idx] = tfr_event
    return eeg_scores
