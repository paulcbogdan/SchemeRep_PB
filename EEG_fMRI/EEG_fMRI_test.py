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

ROOT_EEG_FMRI = fr'G:\EEG_fMRI'
# ROOT_EEG_FMRI = fr'C:\Users\Paul\Downloads'

def get_fMRI_score_sn(sn, combine_regions=True):
    # root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'

    dir_func = fr'{root_sn}\func\sub-{sn}_ses-01_task-rest_bold\func_preproc'
    fp_fMRI = fr'{dir_func}\func_pp_filter_sm0.mni152.3mm.nii.gz'
    img = image.load_img(fp_fMRI)
    # img = pickle_wrap(image.load_img, kwargs={'img': fp_fMRI}) # 0.476 Hz
    print(img.shape)
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
    print(ar_fMRI.shape)
    # double check for GSR

    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]
    print(conn_fMRI.shape)
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

def get_event2true(dir_eeg, sn):
    events = pd.read_csv(fr'{dir_eeg}\sub-{sn}_ses-01_task-rest_events.tsv',
                         sep='\t')

    max_onset = events['onset'].max()
    total_duration = events.loc[events['value'] == 'boundary']['duration'].sum()
    if not ('S  1' in events['value'].values):
        print('No S  1: BAD!')
        return None

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
    for idx, event in events.iterrows():
        if event['value'] == 'S  1':
            start = True

        if event['type'] == 'Response':
            num2s += 1
            t_prev = event['onset']
            event2true[cnt_event] = cnt
            cnt_event += 1
            # if not start:
            #     print('Starting time:', t_st + event['onset'])
            #     quit()
            # start = True
            cnt += 1
            # print(f'LOL: {cnt}')
        # if event['type']
        t_st += event['duration']

        if start and (event['value'] == 'boundary'):# and event['sample'] > 10:
            boundary_events.add(cnt_event)
            boundary_events.add(cnt_event + 1)
            # print(event)
            t_effect = event['onset'] - t_prev + event['duration']
            for i in range(int(t_effect / 2.1)):
                cnt += 1
                # print(f'BOO: {cnt}')
        # elif (event['value'] == 'boundary'):
            # print('EHHHHHH:', event['duration'])

    return event2true, boundary_events

def get_EEG_score_sn(sn):
    # root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'

    dir_eeg = fr'{root_sn}\eeg'
    fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-01_task-rest_eeg.set'
    raw = read_raw_eeglab(fp_EEG, preload=True, )
    event2true, boundary_events = get_event2true(dir_eeg, sn)

    events = mne.events_from_annotations(raw)
    events = events[0]
    # raw, _ = mne.set_eeg_reference(raw)
    data = raw.get_data()
    print(f'{data.shape=}')

    # events = np.array(events)
    events = events[events[:, 2] == 2]
    # print(f'{events_test.shape=}')

    # picks = ['Fz', 'Pz'] # 'Cz',
    picks = raw.ch_names[:5]
    # print(raw)
    data_eeg = raw.get_data(picks=picks)
    freqs = np.arange(1, 101, 1)
    tfr = mne.time_frequency.tfr_array_morlet(data_eeg[None, :, :],
                                              sfreq=250, freqs=freqs,
                                              output='power')
    tfr = tfr[0, ...]
    tfr = np.log(tfr)
    tfr = tfr.mean(axis=0) # avg Fz, Cz, Pz

    # st_event = events[events[:, 2] == 2][0, :]
    eeg_scores = []
    # idx_st = st_event[0]
    # idx_end = idx_st + int(2.1*250)
    # while idx_end < tfr.shape[1]:
    #     tfr_event = tfr[..., idx_st:idx_end].mean(axis=-1)
    #     eeg_scores.append(tfr_event)
    #     idx_st = idx_end
    #     idx_end = idx_st + int(2.1*250)
    # eeg_scores = np.array(eeg_scores)
    # print(len(eeg_scores))
    # quit()
        # st_event = events[events[:, 0] > idx_end][0, :]
        # if st_event[0] > 60*250:
        #     break

    eeg_scores = np.full((288, len(freqs)), np.nan)
    for idx, event in enumerate(events):
        true_idx = event2true[idx]
        if true_idx in boundary_events:
            continue

        t_st = event[0]
        t_end = t_st + int(2.1*250)
        tfr_event = tfr[..., t_st:t_end].mean(axis=-1)
        eeg_scores[true_idx] = tfr_event
        # print(f'{idx}: {true_idx=}')
        # eeg_scores.append(tfr_event)
    # eeg_scores = np.array(eeg_scores)
    # quit()
    # plt.imshow(eeg_scores.T, aspect='auto')
    # plt.show()
    # quit()
    return eeg_scores

def print_events(raw):
    events = mne.events_from_annotations(raw) # 250 Hz
    print(events[1])
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

def test_EEG_fMRI_sn(sn='06'):
    fMRI_fluc = pickle_wrap(get_fMRI_score_sn, kwargs={'sn': sn},
                            easy_override=False)
    # print(f'{fMRI_fluc.shape=}')

    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs={'sn': sn},
                           easy_override=True)
    # print(f'{EEG_fluc.shape=}')
    # return
    # num_EEGs = EEG_fluc.shape[0]
    # fMRI_fluc = fMRI_fluc[2:num_EEGs]
    # EEG_fluc = EEG_fluc[2:num_EEGs]


    # EEG_fluc = EEG_fluc[2:280] # drop initially being low
    # quit()
    # print(EEG_fluc.shape)
    ranges = {'delta': (1, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 100)}
    name2fluc = {}
    for name, rng in ranges.items():
        idxs = np.arange(*rng)
        fluc = EEG_fluc[:, idxs].mean(axis=1)
        name2fluc[name] = fluc
        r, p = stats.spearmanr(fluc, fMRI_fluc, nan_policy='omit')
        print(f'{name}: {r=:.3f}, {p=:.3f}')
    #     plt.plot(fluc, label=name)
    # plt.legend()
    # plt.show()
    # print(name2fluc)





if __name__ == '__main__':
    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    SNS = ['01']
    for SN in SNS:
        test_EEG_fMRI_sn(SN)


