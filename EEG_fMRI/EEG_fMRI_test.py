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
from time import time

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
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'

    dir_func = fr'{root_sn}\func\sub-{sn}_ses-{sess}_bold\func_preproc'
    fp_fMRI = fr'{dir_func}\func_pp_filter_sm0.mni152.3mm.nii.gz'
    if os.path.isfile(fp_fMRI):
        img = image.load_img(fp_fMRI)
    else:
        print(f'No file: {fp_fMRI=}')
        return None
    # img = pickle_wrap(image.load_img, kwargs={'img': fp_fMRI}) # 0.476 Hz
    data_fMRI = img.get_fdata()

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

def get_event2true(fp_EEG, num_TRs):
    # events = pd.read_csv(fr'{dir_eeg}\sub-{sn}_ses-{sess}_task-rest_events.tsv',
    #                      sep='\t')
    # if 'task-rest' in sess:
    #     num_TRs = 288
    # elif 'task-inscapes' in sess:
    #     num_TRs = 293

    mat = loadmat(fp_EEG)
    # urevent = mat['urevent'][0]
    R128_cnt = 0
    urevent2TR = {}
    for event in mat['urevent'][0]:
        if len(event[4][0]):
            urevent_num = event[4][0][0]
            # print(event)
            # print(len(event))
            # print(event)
            assert len(event) in [7, 8], f'{len(event)=}'
            # if len(event) == 8:
            #     name = event[6][0]
            # else:
            # print(event)
            name = event[-2][0]
            # print(name, ':', len(event))
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
        if i + 1 not in good_events:
            boundary_events.add(i)
            boundary_events.add(i + 1)
    return event2true, boundary_events

def get_EEG_score_sn(sn, num_TRs, sess='01', picks=None):
    if picks is None:
        picks = ['Fz', 'Cz', 'Pz',
                 'F1', 'C1', 'P1',
                 'F2', 'C2', 'P2']
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_eeg = fr'{root_sn}\eeg'
    # fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-{sess}_task-rest_eeg.set'

    fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-{sess}_eeg.set'


    if not os.path.isfile(fp_EEG):
        print(f'No file: {fp_EEG=}')
        return None
    raw = read_raw_eeglab(fp_EEG, preload=True, )
    try:
        event2true, boundary_events = get_event2true(fp_EEG, num_TRs)
    except AssertionError as e:
        print(f'{sn} | {e=}')
        return None
    if event2true is None:
        return None

    events = mne.events_from_annotations(raw, verbose=False)
    events = events[0]
    data = raw.get_data()
    # print(f'{data.shape=}')

    events = events[events[:, 2] == 2]

    # print(raw.ch_names)
    # quit()

    picks = [pick for pick in picks if pick in raw.ch_names]
    # picks = ['Fz', 'Pz'] # 'Cz',
    # picks = raw.ch_names[:5]
    # print(raw.ch_names)
    # quit()
    # picks = 'all'
    # print(raw)
    data_eeg = raw.get_data(picks=picks)
    freqs = np.arange(1, 51, 1)
    # print(data_eeg.shape)
    # quit()
    ds1 = 2
    # ds2 = 10
    # t = time()
    tfr = mne.time_frequency.tfr_array_morlet(data_eeg[None, :, ::ds1],
                                              sfreq=250 // ds1, freqs=freqs,
                                              # decim=ds2,
                                              output='power')
    # print(f'{time() - t=:.5f}')
    # print('made TFR')

    tfr = tfr[0, ...]
    tfr = np.log(tfr)
    tfr -= tfr.mean(axis=-1, keepdims=True)

    # print(tfr)
    # quit()
    tfr = tfr.mean(axis=0) # avg Fz, Cz, Pz
    # print(tfr.shape)
    # plt.imshow(tfr[:, int(10*250/ds1):-int(5*250/ds1)], aspect='auto')
    # plt.show()
    # print(tfr)
    # print(tfr.shape)
    # quit()
    # if 'task-rest' in sess:
    #     num_TRs = 288
    # elif 'task-inscapes' in sess:
    #     num_TRs = 293

    eeg_scores = np.full((num_TRs, len(freqs)), np.nan)
    for idx, event in enumerate(events):
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return None
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1*(250 // ds1))
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
    # print(f'{fMRI_fluc.shape=}')
    # quit()
    num_TRs = fMRI_fluc.shape[-1]
    picks = ['Fz', 'Cz', 'Pz',
             'F1', 'C1', 'P1',
             'F2', 'C2', 'P2']
    picks = ['F1', 'Fz', 'F2',
             'FC1', 'FCz', 'FC2',]
    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs={'sn': sn,
                                                     'sess': sess,
                                                     'num_TRs': num_TRs,
                                                     'picks': picks},
                           easy_override=True, verbose=-1)

    if EEG_fluc is None:
        return None
    EEG_fluc = EEG_fluc[5:-5] # clip bad TFR from end
    fMRI_fluc = fMRI_fluc[5:-5] # clip bad TFR from end
    # plt.plot(fMRI_fluc)
    # plt.show()
    # quit()
    ranges = {'delta': (1, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 50)}
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
    SESSES = ['01_task-rest', '02_task-rest']
    SESS_INK = ['01_task-inscapes', '02_task-inscapes']
    SESS_MONKEY = ['01_task-monkey1', '02_task-monkey2']
    # SNS = ['06']
    # SNS = ['18']
    SESSES += SESS_INK
    # SESSES += SESS_MONKEY
    BAD_SNS = {('06', '02_task-rest'), ('12', '01_task-rest'),
               ('16', '02_task-rest'), ('18', '01_task-rest'),
               ()} #
    NAME2L = defaultdict(list)
    for SN in SNS:
        # if SN in BAD_SNS:
        #     continue
        for SESS in SESSES:
            if (SN, SESS) in BAD_SNS: continue
            print(f'- ({SN}; {SESS}) -')
            name2r = test_EEG_fMRI_sn(SN, SESS)
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



