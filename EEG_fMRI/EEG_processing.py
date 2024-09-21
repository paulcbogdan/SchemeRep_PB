import warnings
from time import sleep

import mne
import numpy as np

from mne.io import read_raw_eeglab
from scipy.io import loadmat

from utils import pickle_wrap

ROOT_EEG_FMRI = fr'F:\EEG_fMRI'


def get_event2true(fp_EEG, num_TRs, excl_before=False):
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
    # event = mat['event'][0]
    event2true = {}
    good_events = set()
    boundary_events = set()
    R128_pos_cnt = 0
    for event in mat['event'][0]:
        # print(f'{event[-3]} | {event=}')
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
    # print(len(good_events))
    # print(good_events)
    # print(len(boundary_events))
    # quit()
    return event2true, boundary_events


def load_EEG(sn, sess, num_TRs, dir_eeg, excl_before=False):
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
        event2true, boundary_events = get_event2true(fp_EEG, num_TRs,
                                                     excl_before=excl_before)
    except AssertionError as e:
        print(f'{sn} | {e=}')
        return None, None, None
    if event2true is None:
        return None, None, None

    return raw, event2true, boundary_events


def get_EEG_score_sn(sn, num_TRs, sess='01', picks=None,
                     avg_before=False, high_gamma=False,
                     super_slow=False, avg_ref=True,
                     double_speed=False, excl_before=True,
                     mastoid_ref=False, Fz_Pz_abs_dif=False,
                     fz_minus_pz=False, delay=False,
                     get_max=False, get_max_avg_before=False,
                     custom_freqs=None):
    assert not (high_gamma and super_slow)
    if picks is None:
        picks = ['Fz', 'Cz', 'Pz',
                 'F1', 'C1', 'P1',
                 'F2', 'C2', 'P2']
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_eeg = fr'{root_sn}\eeg'
    # fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-{sess}_task-rest_eeg.set'

    kw = {'sn': sn, 'sess': sess, 'num_TRs': num_TRs, 'dir_eeg': dir_eeg,
          'excl_before': excl_before}
    raw, event2true, boundary_events = pickle_wrap(load_EEG, kwargs=kw,
                                                   easy_override=False)

    if raw is None:
        return None

    if mastoid_ref:
        try:
            raw = raw.set_eeg_reference(['T7', 'T8'])
        except ValueError:
            try:
                raw = raw.set_eeg_reference(['T7',])
            except ValueError:
                try:
                    raw = raw.set_eeg_reference(['T8',])
                except ValueError:
                    pass
    elif avg_ref:
        raw = raw.set_eeg_reference('average')

    events = mne.events_from_annotations(raw, verbose=False)
    events = events[0]
    events = events[events[:, 2] == 2]


    ds1 = 1
    if Fz_Pz_abs_dif:
        picks_frontal = picks[0]
        Fz_pruned = [pick for pick in picks_frontal
                     if pick in raw.ch_names]
        picks_posterior = picks[1]
        Pz_pruned = [pick for pick in picks_posterior
                     if pick in raw.ch_names]

        data_Fz = raw.get_data(picks=Fz_pruned)
        data_Fz_M = np.mean(data_Fz, axis=0)
        data_Pz = raw.get_data(picks=Pz_pruned)
        data_Pz_M = np.mean(data_Pz, axis=0)

        tfr = np.abs(data_Fz_M - data_Pz_M)[None, None, :]
        eeg_scores = np.full((1, 1, num_TRs,),
                             np.nan)

    else:
        if custom_freqs is not None:
            freqs = list(custom_freqs)
        else:
            freqs = np.linspace(0.5, 50, 100)

        if fz_minus_pz:
            picks_fz = picks[0]
            picks_fz = [pick for pick in picks_fz if pick in raw.ch_names]
            picks_pz = picks[1]
            picks_pz = [pick for pick in picks_pz if pick in raw.ch_names]
            data_Fz_M = raw.get_data(picks=picks_fz)
            data_Fz_M = np.nanmean(data_Fz_M, axis=0)
            data_Pz_M = raw.get_data(picks=picks_pz)
            data_Pz_M = np.nanmean(data_Pz_M, axis=0)

            if delay:
                # data_Fz_M_padded = np.pad(data_Fz_M, pad_width=62, mode='constant',
                #                           constant_values=np.nan)
                # print(data_Pz_M.shape)
                # print(data_Pz_M.shape)

                data_Pz_M_padded = np.pad(data_Pz_M, pad_width=62, mode='constant',
                                          constant_values=data_Pz_M[-1])  # temporal resolution ~250 ms

                data_eeg = data_Fz_M - data_Pz_M_padded[124:]
                # print(data_eeg)
                # quit()
                data_eeg = data_eeg[None, None, :]
            else:
                data_eeg = data_Pz_M - data_Fz_M
                data_eeg = data_eeg[None, None, :]

            picks_pruned = ['dif']
        else:
            picks_pruned = [pick for pick in picks if pick in raw.ch_names]
            data_eeg = raw.get_data(picks=picks_pruned)
            if avg_before:
                data_eeg = np.nanmean(data_eeg, axis=0)[None, None, :]
            else:
                data_eeg = data_eeg[None, :, :]

        tfr = mne.time_frequency.tfr_array_morlet(data_eeg[..., ::ds1],
                                                  sfreq=250 // ds1,
                                                  freqs=freqs,
                                                  output='power')



        tfr = tfr[0, ...]
        eeg_scores = np.full((len(picks_pruned), len(freqs), num_TRs,),
                             np.nan)


    for idx, event in enumerate(events):
        # true_idx = event2true[idx]
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return None#, None
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1*(250 // ds1))
        if get_max:
            tfr_event = tfr[..., t_st:t_end].max(axis=-1)
        elif get_max_avg_before:
            tfr = np.mean(tfr, axis=0)[None, ...]
            tfr_event = tfr[..., t_st:t_end].mean(axis=-1)
        else:
            tfr_event = tfr[..., t_st:t_end].mean(axis=-1)

        eeg_scores[:, :, true_idx] = tfr_event
    return eeg_scores#, psd_full
