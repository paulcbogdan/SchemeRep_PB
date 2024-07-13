import os.path

from nilearn import image
from nilearn.glm.first_level import compute_regressor
from nilearn.image import high_variance_confounds
from scipy.interpolate import interpolate

from EEG_fMRI.plot_EEG_fMRI import plot_hz_corrs
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
from time import sleep
from scipy import signal
import pickle

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.io import loadmat

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

# warnings.filterwarnings('ignore', #category=RuntimeWarning,
#                         )

# message='were expanding outside')
# message='indicating data discontinuities'

#  suppress PerformanceWarning pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)# warnings.filterwarnings('ignore', category=pd.PerformanceWarning)

ROOT_EEG_FMRI = fr'G:\EEG_fMRI'

def get_fMRI_ar(sn, sess, combine_regions, clean=True):
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_func = fr'{root_sn}\func\sub-{sn}_ses-{sess}_bold\func_preproc'
    fp_fMRI = fr'{dir_func}\func_pp_filter_sm0.mni152.3mm.nii.gz'
    if os.path.isfile(fp_fMRI):
        img = image.load_img(fp_fMRI)
    else:
        print(fr'Seemingly no file: {fp_fMRI=}')
        sleep(1)
        if os.path.isfile(fp_fMRI):
            print('\tFile found after 1 s pause')
            img = image.load_img(fp_fMRI)
        else:
            print('\tConfirmed no file')
            return None, None

    if clean:
        df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2,))
        dir_nuisance = fr'{root_sn}\func\sub-{sn}_ses-{sess}_bold\func_nuisance'
        fp_motion = fr'{dir_nuisance}\mc_1-6.txt'
        df_motion = pd.read_csv(fp_motion, sep=' ', header=None,
                                names=[f'motion_{i}' for i in range(6)])
        df_confounds = pd.concat([df_compcor, df_motion], axis=1)
        img = image.clean_img(img, confounds=df_confounds, high_pass=1 / 128,
                              standardize=False, t_r=2.1)

    data_fMRI = img.get_fdata()

    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    key2idxs = {'ATL': [], 'MFG': [], 'IPL': [], 'LOC': []}
    for i, region in enumerate(atlas['ROI_regions']):
        for key, l in key2idxs.items():
            if key in region:
                l.append(i)

    ar_fMRI = img_data2ar(data_fMRI, atlas)
    ar_fMRI = stdize(ar_fMRI, axis=-1)
    return ar_fMRI, key2idxs

def get_fMRI_score_sn(sn, sess='01', combine_regions=False, clean=False,
                      many_ROI=True, abs_analysis=False, all_conn=True,
                      sanity=False):
    print({'sn': sn, 'sess': sess,
                                  'combine_regions': combine_regions,
                                  'clean': clean})

    ar_fMRI, key2idxs = pickle_wrap(get_fMRI_ar,
                          kwargs={'sn': sn, 'sess': sess,
                                  'combine_regions': combine_regions,
                                  'clean': clean},
                          easy_override=False, verbose=0,)

    if ar_fMRI is None:
        return None
    # print(ar_fMRI.shape)
    # conn = np.corrcoef(ar_fMRI)
    # plot_connectivity(conn, atlas=get_atlas(combine_regions=combine_regions))

    atlas = get_atlas(natview=True, combine_regions=combine_regions)


    # print(key2idxs['ATL'])
    # quit()

    ROI2idx = {ROI: [] for ROI in atlas['ROI_regions']}
    # region_l = []
    for i, region in enumerate(atlas['ROI_regions']):
        ROI2idx[region].append(i)
    region_l = list(ROI2idx.keys())
    #     'else': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS', 'SPL',
    #                      'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'Amyg', 'Hipp', 'Str',
    #                      'Tha', 'ACC', 'PCC'],

    if sanity:
        rnd0 = 2
        rnd1 = 8
        rnd2 = 12
        rnd3 = 20
        # rnd0 = np.random.randint(len(region_l))
        # rnd1 = np.random.randint(len(region_l))
        # rnd2 = np.random.randint(len(region_l))
        # rnd3 = np.random.randint(len(region_l))
        print(f'{region_l[rnd0]} | {region_l[rnd1]} | '
              f'{region_l[rnd2]} | {region_l[rnd3]}')
        key2idxs['MFG'] = ROI2idx[region_l[rnd0]]
        key2idxs['IPL'] = ROI2idx[region_l[rnd1]]
        key2idxs['LOC'] = ROI2idx[region_l[rnd2]]
        key2idxs['ATL'] = ROI2idx[region_l[rnd3]]
    elif many_ROI:
        key2idxs['MFG'] = ROI2idx['MFG'] + ROI2idx['IFG'] #
        key2idxs['IPL'] = ROI2idx['IPL'] #+ ROI2idx['SPL']
        key2idxs['LOC'] = ROI2idx['LOC'] + ROI2idx['sOcG'] + ROI2idx['EVC']# +
        key2idxs['ATL'] = ROI2idx['ATL']

    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]
    # print(conn_fMRI.shape)
    # quit()
    # pairs = [('ATL', 'MFG'), ('ATL', 'IPL'), ('ATL', 'LOC'), ]
    if all_conn:
        with open(f'cache/gen_idx2rank_test.pkl', 'rb') as f:
            gen_idx2rank = pickle.load(f)
        trils = np.tril_indices(246)
        trils = list(zip(*trils))
        print(len(trils))

        nan_cutoff = 21736
        low_cutoff = nan_cutoff // 10
        high_cutoff = nan_cutoff - low_cutoff - 1

        low_conns = []
        for idx in range(nan_cutoff):
            rank = gen_idx2rank[idx]
            # print(f'{rank=}')
            if rank > low_cutoff:
                continue
            i, j = trils[idx]
            low_conns.append(conn_fMRI[i, j])
        low_conns = np.array(low_conns)
        high_conns = []
        for idx in range(nan_cutoff): # high_cutoff,
            rank = gen_idx2rank[idx]
            if rank < high_cutoff:
                continue
            i, j = trils[idx]
            high_conns.append(conn_fMRI[i, j])
        high_conns = np.array(high_conns)
        low_conns = stdize(low_conns, axis=-1, rankdata=False)
        high_conns = stdize(high_conns, axis=-1, rankdata=False)
        rnd_low = np.arange(low_conns.shape[0])
        np.random.shuffle(rnd_low)
        low0 = low_conns[rnd_low[::2]]
        low1 = low_conns[rnd_low[1::2]]
        rnd_high = np.arange(high_conns.shape[0])
        np.random.shuffle(rnd_high)
        high0 = high_conns[rnd_high[::2]]
        high1 = high_conns[rnd_high[1::2]]




        same_low = (np.nanmean(low0, axis=0) *
                    np.nanmean(low1, axis=0))
        same_high = (np.nanmean(high0, axis=0) *
                     np.nanmean(high1, axis=0))
        dif_01 = (np.nanmean(low0, axis=0) *
                  np.nanmean(high1, axis=0))
        dif_00 = (np.nanmean(low0, axis=0) *
                  np.nanmean(high0, axis=0))
        dif_10 = (np.nanmean(low1, axis=0) *
                  np.nanmean(high0, axis=0))
        dif_11 = (np.nanmean(low1, axis=0) *
                  np.nanmean(high1, axis=0))



        fluc = (same_low + same_high -
                (dif_01 + dif_10 + dif_00 + dif_11) / 2)

        dif_abs = np.abs(np.nanmean(high_conns, axis=0) -
                         np.nanmean(low_conns, axis=0))

        fluc = dif_abs
    # elif sanity:
    #     ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'],
    #                                key2idxs['MFG'])].mean(axis=(0, 1))
    #     ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'],
    #                                key2idxs['LOC'])].mean(axis=(0, 1))
    #     MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'],
    #                                key2idxs['IPL'])].mean(axis=(0, 1))
    #     IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'],
    #                                key2idxs['LOC'])].mean(axis=(0, 1))
    #     ATL_MFG = stdize(ATL_MFG, axis=-1, rankdata=False)
    #     ATL_LOC = stdize(ATL_LOC, axis=-1, rankdata=False)
    #     MFG_IPL = stdize(MFG_IPL, axis=-1, rankdata=False)
    #     IPL_LOC = stdize(IPL_LOC, axis=-1, rankdata=False)
    #     # fluc = MFG_IPL + ATL_LOC - ATL_MFG + IPL_LOC
    #     fluc = np.abs(MFG_IPL + ATL_LOC + ATL_MFG + IPL_LOC) ** 2

    elif abs_analysis:
        ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'],
                                   key2idxs['MFG'])].mean(axis=(0, 1))
        ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'],
                                   key2idxs['LOC'])].mean(axis=(0, 1))
        MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'],
                                   key2idxs['IPL'])].mean(axis=(0, 1))
        IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'],
                                   key2idxs['LOC'])].mean(axis=(0, 1))
        ATL_MFG = stdize(ATL_MFG, axis=-1, rankdata=False)
        ATL_LOC = stdize(ATL_LOC, axis=-1, rankdata=False)
        MFG_IPL = stdize(MFG_IPL, axis=-1, rankdata=False)
        IPL_LOC = stdize(IPL_LOC, axis=-1, rankdata=False)
        # fluc = MFG_IPL + ATL_LOC - ATL_MFG + IPL_LOC
        fluc = np.abs(MFG_IPL + ATL_LOC - ATL_MFG - IPL_LOC) #** 2
        # fluc = np.abs(MFG_IPL + ATL_LOC + ATL_MFG + IPL_LOC) ** 2

    else:
        ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'], key2idxs['MFG'])]
        ATL_MFG = np.reshape(ATL_MFG, (-1, ATL_MFG.shape[-1]))
        ATL_MFG = np.nanmean(ATL_MFG, axis=0)[None]

        ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'], key2idxs['LOC'])]
        ATL_LOC = np.reshape(ATL_LOC, (-1, ATL_LOC.shape[-1]))
        ATL_LOC = np.nanmean(ATL_LOC, axis=0)[None]

        MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'], key2idxs['IPL'])]
        MFG_IPL = np.reshape(MFG_IPL, (-1, MFG_IPL.shape[-1]))
        MFG_IPL = np.nanmean(MFG_IPL, axis=0)[None]

        IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'], key2idxs['LOC'])]
        IPL_LOC = np.reshape(IPL_LOC, (-1, IPL_LOC.shape[-1]))
        IPL_LOC = np.nanmean(IPL_LOC, axis=0)[None]

        # DV = np.concatenate([IPL_LOC, ATL_MFG], axis=0)
        # print(DV.shape)
        # quit()
        # DV = stdize(DV, axis=-1, rankdata=True)
        # AP = np.concatenate([ATL_LOC, MFG_IPL], axis=0)
        # AP = stdize(AP, axis=-1, rankdata=True)
        # print(DV.shape)
        # print(AP.shape)
        # DV_AP = DV[None, ...] * AP[:, None, ...]

        IPL_LOC = stdize(IPL_LOC, axis=-1, rankdata=False)
        ATL_LOC = stdize(ATL_LOC, axis=-1, rankdata=False)
        MFG_IPL = stdize(MFG_IPL, axis=-1, rankdata=False)
        ATL_MFG = stdize(ATL_MFG, axis=-1, rankdata=False)

        DD_VV = IPL_LOC * ATL_MFG
        DV_DV = ATL_LOC * MFG_IPL
        LD = IPL_LOC * MFG_IPL
        LV = IPL_LOC * ATL_LOC
        RD = ATL_MFG * MFG_IPL
        RV = ATL_MFG * ATL_LOC
        fluc = DD_VV + DV_DV - (LD + LV + RD + RV) / 2
        # fluc = LD + RV

        fluc = fluc[0]


        # print(DV_AP.shape)
        # plt.hist(DV_AP.flatten(), bins=100)
        # plt.show()
        # quit()
        # fluc = DV_AP.mean(axis=(0, 1))
        # print(fluc)

        # print(f'{np.mean(fluc)=:.9f}')




    return fluc

def get_event2true(fp_EEG, num_TRs, excl_before=False):
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
                     avg_before=True, high_gamma=False,
                     super_slow=False, avg_ref=True,
                     double_speed=False,
                     excl_before=True,
                     mastoid_ref=False,
                     Fz_Pz_abs_dif=False):
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
        # print(f'{data_Fz.shape=}')
        data_Fz_M = np.mean(data_Fz, axis=0)
        data_Pz = raw.get_data(picks=Pz_pruned)
        data_Pz_M = np.mean(data_Pz, axis=0)

        tfr = np.abs(data_Fz_M - data_Pz_M)[None, None, :]
        eeg_scores = np.full((1, 1, num_TRs,),
                             np.nan)

    else:
        picks_pruned = [pick for pick in picks if pick in raw.ch_names]
        data_eeg = raw.get_data(picks=picks_pruned)
        if high_gamma:
            freqs = np.arange(1, 101)
        elif super_slow:
            freqs = np.linspace(0.1, 1.0, 10)
        elif double_speed:
            freqs = np.linspace(0.5, 50, 100)
        else:
            freqs = np.arange(1, 51)

        if avg_before:
            data_eeg = np.nanmean(data_eeg, axis=0)[None, None, :]
        else:
            data_eeg = data_eeg[None, :, :]

        tfr = mne.time_frequency.tfr_array_morlet(data_eeg[..., ::ds1],
                                                  sfreq=250 // ds1, freqs=freqs,
                                                  output='power')
        # print(f'{tfr.shape=}')

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
        tfr_event = tfr[..., t_st:t_end].mean(axis=-1)
        eeg_scores[:, :, true_idx] = tfr_event
    # print(eeg_scores.shape)
    # quit()
    return eeg_scores#, psd_full

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

def get_hrf():
    onset, amplitude, duration = 0.0, 1.0, 0.1
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    # time_length = 21
    frame_times = np.arange(20) * 2.1
    # print(frame_times)
    # quit()
    signal, _labels = compute_regressor(
        exp_condition,
        'spm',
        frame_times,
        con_id="main",
        oversampling=50,
        # min
    )
    # print(signal.shape)
    # print(signal)
    # quit()
    return signal[:, 0]

ELECTRODES_KEPT = []

M_AUTOCORR = []

# TODO: TEST AVERAGE BEFORE!!
def test_EEG_fMRI_sn(sn='06', sess='01', avg_before=False,
                     high_gamma=False, many_ROI=True,
                     super_slow=False, avg_ref=True,
                     abs_analysis=True, all_conn=False,
                     double_speed=True, Fz_Pz_abs_dif=False):
    # changed to remove SFGG
    # dt_max = datetime(2024, day=18, month=3, hour=9) if many_ROI else None

    fMRI_fluc = pickle_wrap(get_fMRI_score_sn,
                            kwargs={'sn': sn, 'sess': sess,
                                    'many_ROI': many_ROI,
                                    'combine_regions': False,
                                    'abs_analysis': abs_analysis,
                                    'all_conn': all_conn,
                                    'clean': True,
                                    'sanity': False},
                            easy_override=False, verbose=-1,)

    if fMRI_fluc is None:
        print(f'None fMRI fluc ({sn}; {sess}) !')
        return None, None

    num_TRs = fMRI_fluc.shape[-1]

    if Fz_Pz_abs_dif:
        picks = [['F1', 'Fz', 'F2', 'F3', 'F4',
                 'FC1', 'FCz', 'FC2', 'FC3', 'FC4'],
                 [ 'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
                 'P1', 'Pz', 'P2', 'P3', 'P4']]
    else:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4',
                 'FC1', 'FCz', 'FC2', 'FC3', 'FC4',
                 'C1', 'Cz', 'C2', 'C3', 'C4',
                 'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
                 'P1', 'Pz', 'P2', 'P3', 'P4']


    EEG_fluc = pickle_wrap(get_EEG_score_sn,
                           kwargs={'sn': sn, 'sess': sess,
                                   'num_TRs': num_TRs,
                                   'picks': picks,
                                   'avg_before': avg_before,
                                   'high_gamma': high_gamma,
                                   'super_slow': super_slow,
                                   'avg_ref': avg_ref,
                                   'mastoid_ref': False,
                                   'double_speed': double_speed,
                                   'excl_before': True,
                                   'Fz_Pz_abs_dif': Fz_Pz_abs_dif,},
                           easy_override=False, verbose=-1)

    if EEG_fluc is None:
        print(f'BAD EEG!! ({sn}; {sess})')
        return None, None
    if Fz_Pz_abs_dif:
        EEG_fluc = np.repeat(EEG_fluc, 100, axis=1)
    fMRI_fluc = fMRI_fluc[4:]


    # num_nans = np.isnan(EEG_fluc[0, 4]).sum()

    EEG_fluc = np.nanmean(EEG_fluc, axis=0) # (freq, TR)

    EEG_fluc = EEG_fluc.T # (TR, freq)
    # print(EEG_fluc.shape)
    # quit()
    EEG_fluc = EEG_fluc[4:, :] # drop edge artifact
    EEG_fluc = conv(EEG_fluc)
    # print(EEG_fluc.shape)
    # quit()
    # quit()
    # plt.imshow(EEG_fluc)
    # plt.show()
    # quit()
    # print(EEG_fluc_test)
    # print(EEG_fluc.shape)
    # quit()
    # plt.imshow(EEG_fluc)
    # plt.show()
    # import scipy.ndimage as ndimage
    # HRF = get_hrf()
    # for i in range(EEG_fluc.shape[1]):
    #     nans, x = np.isnan(EEG_fluc[:, i]), lambda z: z.nonzero()[0]
    #     EEG_fluc[nans, i] = np.interp(x(nans), x(~nans), EEG_fluc[~nans, i])
    #
    # EEG_fluc = ndimage.convolve1d(EEG_fluc, HRF, mode='nearest',
    #                               origin=-HRF.shape[0] // 2, axis=0)
    # plt.imshow(EEG_fluc)
    # plt.show()
    # quit()
    # print('-'*100)
    # print(EEG_fluc)
    # quit()
    # if sn == '04':
    #     theta_fluc = np.nanmean(EEG_fluc[:, 60:], axis=1)
    #     print(theta_fluc)
    #     quit()
    # print(EEG_fluc.shape)
    # quit()
    # if sn == '04':
    #     print(EEG_fluc)
    #     quit()
    # EEG_fluc = conv(EEG_fluc)
    # EEG_fluc = EEG_fluc.T
    # print(EEG_fluc)
    # quit()
    # print(EEG_fluc.shape)
    # quit()
    # print(np.nanmean(EEG_fluc[:, 60:], axis=1))
    # quit()

    if Fz_Pz_abs_dif:
        ranges = {'Fz_Pz_abs_dif': (0, 1)}
    else:
        ranges = {'delta': (1, 4),
                  'theta': (4, 8),
                  'alpha': (8, 13),
                  'beta': (13, 30),
                  'gamma': (30, 50.5),
                  }
        if double_speed:
            for key, tup in ranges.items():
                ranges[key] = (int(tup[0]*2), int(tup[1]*2))

        if high_gamma:
            ranges['high_gamma'] = (50, 100)

        if double_speed:
            ranges_ = {f'r{hz}': (hz, hz + 1) for hz in range(1, 101)}
        else:
            ranges_ = {f'r{hz}': (hz, hz + 1) for hz in range(1, 51)}
        ranges.update(ranges_)

    name2fluc = {}
    name2r = {}
    df = pd.DataFrame({'fMRI_fluc': fMRI_fluc})
    df['sn'] = sn
    df['sess'] = sess

    for name, rng in ranges.items():
        idxs = np.arange(*rng)
        idxs -= 1
        if name == 'gamma':
            range_fluc = EEG_fluc[:, idxs[0]:].mean(axis=1)
        else:
            range_fluc = EEG_fluc[:, idxs].mean(axis=1)
        df[name] = range_fluc
        df_ = df.dropna()

        # print(range_fluc)
        # df_[[name, 'fMRI_fluc']] = (
        #     stats.zscore(df_[[name, 'fMRI_fluc']], axis=0))

        # if sn == '04' and name == 'gamma':
        #     # print(EEG_fluc)
        #     # print(df_[name])
        #     # print(ran)
        #     # print(df[name])
        #     # print(idxs[0])
        #     # range_fluc = EEG_fluc[:, idxs[0]:].mean(axis=1)
        #     # print(range_fluc)
        #     # print(df_)
        #
        #     print(df_[[name, 'fMRI_fluc']])
        #     for idx, row in df_.iterrows():
        #         print(f'{idx}', round(row['fMRI_fluc'], 2), row[name])
        #     r, p = stats.pearsonr(df_[name], df_['fMRI_fluc'])
        #     print(f'{r=}')
        #     quit()

        r, p = stats.spearmanr(df_[name], df_['fMRI_fluc'])
        # print(f'{r=:.2f}')

        name2fluc[name] = range_fluc

        r = np.arctanh(r)
        if 'r' != name[0]:
            print(f'{name}: {r=:.3f}, {p=:.3f}')
            # print('TOAST')

        name2r[name] = r
    # quit()
    # if sn == '04':
    #     quit()
    return name2r, df#, psd


def plot_regrs(df, key='theta'):
    cm = plt.get_cmap('viridis')
    n_sn = len(df['sn'].unique())
    colors = [cm(i / n_sn) for i in range(n_sn)]
    colors = np.array(colors)
    plt.rcParams.update({'font.size': 14})

    l = []
    for i, (sn, df_sn) in enumerate(df.groupby('sn')):
        df_sn_ = df_sn[['fMRI_fluc', key]].dropna()
        df_sn_[key] = stats.zscore(df_sn_[key])
        r, p = stats.spearmanr(df_sn_['fMRI_fluc'], df_sn_[key])
        print(f'final: {r=}')
        l.append(r)
        # df_sn_['fMRI_fluc'] = stats.zscore(df_sn_['fMRI_fluc'])
        m, b = np.polyfit(df_sn_['fMRI_fluc'], df_sn_[key], 1)
        # fluc_min = df_sn_['fMRI_fluc'].min()
        fluc_min = 0
        fluc_max = 2
        # fluc_max = df_sn_['fMRI_fluc'].max()
        vmin = -0.04
        vmax = 0.1
        dif = vmax - vmin
        frac = (r - vmin)/dif
        # print(frac)
        frac = min(0.95, frac)
        c = cm(frac)
        plt.plot([fluc_min, fluc_max],
                 [m*fluc_min + b, m*fluc_max + b], linewidth=1.5,
                 color=c)
    t, p = stats.ttest_1samp(l, 0)
    print(f'Final ({key}): {t=:.3f}')
    plt.axhline(y=0., color='k', linestyle='-')
    plt.xlim(0, 2)
    plt.gca().spines[['bottom', 'right', 'top']].set_visible(False)
    plt.xlabel('fMRI fluctuation (z-score)', labelpad=10)
    plt.ylabel('Power (z-score)', labelpad=4)
    plt.title('Theta [4-8 Hz]')
    plt.tight_layout()
    plt.show()
    quit()

def conv(EEG_fluc):
    if len(EEG_fluc.shape) == 1:
        EEG_fluc = EEG_fluc[:, None]
        undo = True

    else:
        undo = False

    # print(EEG_fluc)
    import scipy.ndimage as ndimage
    HRF = get_hrf()
    for i in range(EEG_fluc.shape[1]):
        nans, x = np.isnan(EEG_fluc[:, i]), lambda z: z.nonzero()[0]
        EEG_fluc[nans, i] = np.interp(x(nans), x(~nans), EEG_fluc[~nans, i])

    EEG_fluc = ndimage.convolve1d(EEG_fluc, HRF, mode='nearest',
                                  origin=-HRF.shape[0] // 2, axis=0)
    if undo:
        return EEG_fluc[:, 0]
    else:
        return EEG_fluc

    # EEG_fluc = EEG_fluc.T # (TR, freq)
    # HRF = get_hrf()
    # nans, x = np.isnan(EEG_fluc), lambda z: z.nonzero()[0]
    # EEG_fluc[nans] = np.interp(x(nans), x(~nans), EEG_fluc[~nans])
    #
    # EEG_fluc = ndimage.convolve1d(EEG_fluc, HRF, mode='nearest',
    #                               origin=-HRF.shape[0] // 2, axis=0)
    # return EEG_fluc.T


if __name__ == '__main__':
    # ar = get_fMRI_ar('05', '01_task-rest', False,
    #                  clean=True)
    # print(ar.shape)
    # quit()

    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    # SNS = ['13']
    SESSES = ['01_task-rest', '02_task-rest']
    SESS_INK = ['01_task-inscapes', '02_task-inscapes'] # shape things

    SESS_OTHER = ['01_task-checker', # Designed to induce visual effects
                  '01_task-dme_run-01', '01_task-dme_run-02', # dispicable me
                  '01_task-monkey1_run-01', '01_task-monkey1_run-02', # movie
                  # '01_task-peer', # used for E-T calibration
                  '01_task-tp_run-01', '01_task-tp_run-02' # "The present"
                  ]
    # SESS_OTHER = ['01_task-monkey1_run-01', '01_task-monkey1_run-02']
    SESS_OTHER += [f'02' + sess[2:] for sess in SESS_OTHER]

    # SNS = ['06']
    # SNS = ['18']
    SESSES += SESS_INK
    # SESSES = SESS_OTHER # I'm not sure if im even processsing the task properylll

    # SESSES += SESS_OTHER
    # SESSES += SESS_MONKEY
    # SESSES = ['01_task-dme_run-01']
    # SNS = ['22']
    BAD_SNS = {('06', '02_task-rest'), ('12', '01_task-rest'),
               ('16', '02_task-rest'), ('18', '01_task-rest'),
               ('06', '02_task-inscapes'), ('07', '01_task_inscapes'),
               ('09', '01_task-inscapes'), ('12', '01_task-inscapes'),
               ('15', '02_task-inscapes'), ('18', '01_task_inscapes')} # Bad EEG alignment
    NAME2L = defaultdict(list)
    NAME2SN2L = defaultdict(lambda: defaultdict(list))
    PSDs = []
    dfs_l = []
    for SN in SNS[14:]:
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            print(f'- ({SN}; {SESS}) -')
            name2r, df = test_EEG_fMRI_sn(SN, SESS)
            # print(f'{PSD.shape=}')
            if name2r is None:
                continue
            dfs_l.append(df)

            for key, r in name2r.items():
                # NAME2L[key].append(r)
                NAME2SN2L[SN][key].append(r)

            # print(np.array(PSDs).shape)

            NAME2L = defaultdict(list)
            simple2l = defaultdict(list)
            for SN, d in NAME2SN2L.items():
                # print(f'left: {SN}')
                for key, l in d.items():
                    NAME2L[key].append(np.mean(l))
                    simple2l[key].extend(l)
                    # print(f'{len(NAME2L[key])=}')
                # NAME2L[]

            for key, l in NAME2L.items():
                N = len(l)
                if N > 5:
                    M = np.mean(l)
                    SD = np.std(l, ddof=1)
                    SE = SD / np.sqrt(N)
                    t = M / SE
                    d = M / SD
                    p = stats.t.sf(np.abs(t), len(l) - 1) * 2
                    M_low = M - 1.96 * SE
                    M_high = M + 1.96 * SE


                    res = stats.wilcoxon(l)
                    # print(f'{res=}')
                    p_wilcox = res.pvalue * 2

                    M_above = np.mean([m > 0 for m in l])
                    # print(f'test: {key}')
                    if key[0] == 'r': continue
                    # if 'r1' in NAME2L:
                    #     continue
                    print(f'{key} ({N=}): {M=:.3f} [{M_low:.3f}, {M_high:.3f}] '
                          f'({t=:.3f} | {d=:.3f}), {p=:.1e}, {p_wilcox=:.1e} | '
                          f'{M_above:.1%}')
            # continue
            if 'r50' not in NAME2L:
                continue
            if N == 21 and j == len(SESSES) - 1:
                # print('done')
                if 'r1' in NAME2L:
                    plot_hz_corrs(NAME2L)
                    # plot_hz_corrs(simple2l)
                    plot_hz_corrs(NAME2L, effect_size=True, plot_se=True)
                else:
                    plot_regrs(pd.concat(dfs_l))


