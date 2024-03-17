from nilearn import image
from atlas_utils import get_atlas
from get_HCP_act import img_data2ar
from mne.io import read_raw_eeglab
from utils import pickle_wrap, stdize
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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


def get_EEG_score_sn(sn):
    # root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-01'

    dir_eeg = fr'{root_sn}\eeg'
    fp_EEG = fr'{dir_eeg}\sub-{sn}_ses-01_task-rest_eeg.set'
    raw = read_raw_eeglab(fp_EEG, preload=True, )

    events = mne.events_from_annotations(raw)
    # print_events(raw)
    # print(events[1])
    # quit()
    events = events[0]
    # raw, _ = mne.set_eeg_reference(raw)
    data = raw.get_data()
    print(f'{data.shape=}')

    # events = np.array(events)
    # events = events[events[:, 2] == 2]

    picks = ['Fz', 'Pz'] # 'Cz',
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

    st_event = events[events[:, 2] == 2][0, :]
    print(st_event)
    quit()

    eeg_scores = []
    for event in events:
        t_st = event[0]
        t_end = t_st + int(2.1*250)
        tfr_event = tfr[..., t_st:t_end].mean(axis=-1)
        eeg_scores.append(tfr_event)
    eeg_scores = np.array(eeg_scores)
    return eeg_scores

def print_events(raw):
    events = mne.events_from_annotations(raw) # 250 Hz
    print(events[1])
    events = events[0]
    # events = np.array(events).astype(float)
    # events[:, 0] /= 250

    events = np.array(events)
    events = events[events[:, 2] == 2]

    for i, event in enumerate(events[:-1]):
        next_event = events[i+1]
        dif = next_event[0] - event[0]
        if dif != 525:
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
    print(f'{fMRI_fluc.shape=}')

    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs={'sn': sn},
                           easy_override=True)
    # print(f'{EEG_fluc.shape=}')
    # return
    num_EEGs = EEG_fluc.shape[0]
    fMRI_fluc = fMRI_fluc[2:num_EEGs]
    EEG_fluc = EEG_fluc[2:num_EEGs]


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
        r, p = stats.spearmanr(fluc, fMRI_fluc)
        print(f'{name}: {r=:.3f}, {p=:.3f}')
    #     plt.plot(fluc, label=name)
    # plt.legend()
    # plt.show()
    # print(name2fluc)





if __name__ == '__main__':
    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    for SN in SNS:
        test_EEG_fMRI_sn(SN)


