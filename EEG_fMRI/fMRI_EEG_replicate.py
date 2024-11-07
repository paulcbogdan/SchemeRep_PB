import os.path

from nilearn.glm.first_level import compute_regressor

from Study3.fMRI_simultaneous_funcs import get_fMRI_ar
from atlas_utils import get_atlas
from mne.io import read_raw_eeglab

from old_Apr6.fluctuations import partial_corr_df
from utils import pickle_wrap
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd
import warnings
from collections import defaultdict
from time import sleep

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.io import loadmat

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

# warnings.filterwarnings('ignore', #category=RuntimeWarning,
#                         )

# message='were expanding outside')
# message='indicating data discontinuities'

ROOT_EEG_FMRI = fr'G:\EEG_fMRI'
# ROOT_EEG_FMRI = fr'C:\Users\Paul\Downtfr_array_morletloads'



def get_fluc_score_sn(sn, sess='01', combine_regions=True, clean=False,
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


    key2idxs['MFG'] = ROI2idx['MFG'] + ROI2idx['IFG'] #
    key2idxs['IPL'] = ROI2idx['IPL'] #+ ROI2idx['SPL']
    key2idxs['LOC'] = ROI2idx['LOC'] + ROI2idx['sOcG'] + ROI2idx['EVC']# +
    key2idxs['ATL'] = ROI2idx['ATL']

    all_regions = atlas['ROI_regions']
    all_regions = [region for region in all_regions if region not in
                   ['MFG', 'IFG', 'IPL', 'LOC', 'sOcG', 'EVC', 'ATL',
                    'Hipp', 'Str', 'Amyg', 'Tha']]
    key2idxs['else'] = []
    for region in all_regions:
        key2idxs['else'] += ROI2idx[region]


    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]

    ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['MFG'])].mean(axis=(0, 1))
    ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))
    MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'],
                               key2idxs['IPL'])].mean(axis=(0, 1))
    IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))

    LOC_else = conn_fMRI[np.ix_(key2idxs['LOC'],
                                key2idxs['else'])].mean(axis=(0, 1))
    MFG_else = conn_fMRI[np.ix_(key2idxs['MFG'],
                                key2idxs['else'])].mean(axis=(0, 1))
    IPL_else = conn_fMRI[np.ix_(key2idxs['IPL'],
                                key2idxs['else'])].mean(axis=(0, 1))
    ATL_else = conn_fMRI[np.ix_(key2idxs['ATL'],
                                key2idxs['else'])].mean(axis=(0, 1))


    # ATL_MFG = stdize(ATL_MFG, axis=-1, rankdata=False)
    # ATL_LOC = stdize(ATL_LOC, axis=-1, rankdata=False)
    # MFG_IPL = stdize(MFG_IPL, axis=-1, rankdata=False)
    # IPL_LOC = stdize(IPL_LOC, axis=-1, rankdata=False)

    dd_vv = ATL_LOC + MFG_IPL
    dv_dv = IPL_LOC + ATL_MFG
    df = pd.DataFrame({'dd_vv': dd_vv, 'dv_dv': dv_dv,
                       'pd_no': IPL_else, 'ad_no': MFG_else,
                       'av_no': ATL_else, 'pv_no': LOC_else})
    networks = ['dd_vv', 'dv_dv']
    ar = partial_corr_df(df, networks,
                         cov=['pd_no', 'ad_no', 'av_no', 'pv_no'])
    r = ar[0, 1]
    return r

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

    if avg_ref:
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
        print(f'{data_Fz.shape=}')
        data_Fz_M = np.mean(data_Fz, axis=0)
        data_Pz = raw.get_data(picks=Pz_pruned)
        data_Pz_M = np.mean(data_Pz, axis=0)

        tfr = np.abs(data_Fz_M - data_Pz_M)[None, None, :]
        eeg_scores = np.full((1, 1, num_TRs,),
                             np.nan)
        # print(f'{eeg_scores=}')
        # # print(f'{tfr.shape=}')
        # quit()
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
    # print(f'{tfr.shape=}')
    # quit()


    # print(f'')

    # print(tfr)
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

def test_EEG_fMRI_sn(sn='06', sess='01', avg_before=False,
                     high_gamma=False, many_ROI=True,
                     super_slow=False, avg_ref=False,
                     abs_analysis=True, all_conn=False,
                     double_speed=True,
                     Fz_Pz_abs_dif=False):
    # changed to remove SFGG
    # dt_max = datetime(2024, day=18, month=3, hour=9) if many_ROI else None

    fMRI_fluc = pickle_wrap(get_fluc_score_sn,
                            kwargs={'sn': sn, 'sess': sess,
                                    'many_ROI': many_ROI,
                                    'combine_regions': False,
                                    'abs_analysis': abs_analysis,
                                    'all_conn': all_conn,
                                    'clean': True,
                                    'sanity': False},
                            easy_override=True, verbose=-1, )
    return fMRI_fluc
    # print(f'{fMRI_fluc.shape=}')


if __name__ == '__main__':

    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    SESSES = ['01_task-rest', '02_task-rest']
    SESS_INK = ['01_task-inscapes', '02_task-inscapes']

    SESS_OTHER = ['01_task-checker',
                  '01_task-dme_run-01', '01_task-dme_run-02',
                  '01_task-monkey1_run-01', '01_task-monkey1_run-02',
                  '01_task-tp_run-01', '01_task-tp_run-02' # "The present"
                  ]
    SESS_OTHER += [f'02' + sess[2:] for sess in SESS_OTHER]

    SESSES += SESS_INK

    BAD_SNS = {('06', '02_task-rest'), ('12', '01_task-rest'),
               ('16', '02_task-rest'), ('18', '01_task-rest'),
               ('06', '02_task-inscapes'), ('07', '01_task_inscapes'),
               ('09', '01_task-inscapes'), ('12', '01_task-inscapes'),
               ('15', '02_task-inscapes'), ('18', '01_task_inscapes')} # Bad EEG alignment
    NAME2L = defaultdict(list)
    NAME2SN2L = defaultdict(lambda: defaultdict(list))
    PSDs = []
    dfs_l = []
    scores_all = []
    for SN in SNS:
        scores_sn = []
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            print(f'- ({SN}; {SESS}) -')
            score = test_EEG_fMRI_sn(SN, SESS)
            scores_sn.append(score)
            # print(f'{score=}')
        scores_all.append(np.nanmean(scores_sn))
        t, p = stats.ttest_1samp(scores_all, 0)
        if np.isnan(t):
            print('NAN')
            continue
        print(f'one-sample: t[{np.sum(~np.isnan(scores_all))}] = '
              f'{t:.2f}, {p=:.4f}')
        M_r = np.mean(scores_all)
        print(f'\t{M_r=:.3f}')