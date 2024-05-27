import os.path

from nilearn import image
from nilearn.glm.first_level import compute_regressor
from nilearn.image import high_variance_confounds
from scipy.interpolate import interpolate

from EEG_fMRI.EEG_fMRI_test import load_EEG, get_fMRI_ar, get_hrf
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
import pickle

import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

ROOT_EEG_FMRI = fr'G:\EEG_fMRI'

def get_fMRI_AP_VD(sn, sess='01', combine_regions=True, clean=False,
                   many_ROI=True, abs_analysis=False, all_conn=True):
    # print({'sn': sn, 'sess': sess,
    #                               'combine_regions': combine_regions,
    #                               'clean': clean})
    ar_fMRI, key2idxs = pickle_wrap(get_fMRI_ar,
                          kwargs={'sn': sn, 'sess': sess,
                                  'combine_regions': combine_regions,
                                  'clean': clean},
                          easy_override=False, verbose=0,)


    if ar_fMRI is None:
        return None

    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    ROI2idx = {ROI: [] for ROI in atlas['ROI_regions']}
    for i, region in enumerate(atlas['ROI_regions']):
        ROI2idx[region].append(i)

    if many_ROI:
        key2idxs['MFG'] = ROI2idx['MFG'] + ROI2idx['IFG'] #
        key2idxs['IPL'] = ROI2idx['IPL'] #+ ROI2idx['SPL']
        key2idxs['LOC'] = ROI2idx['LOC'] + ROI2idx['sOcG'] + ROI2idx['EVC']# +
        key2idxs['ATL'] = ROI2idx['ATL']

    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]

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

    return ATL_MFG, ATL_LOC, MFG_IPL, IPL_LOC
    # print(ATL_MFG.shape)
    # quit()

def get_ERP_sn(sn, num_TRs, sess='01', picks=None, avg_before=True,
               high_gamma=False, super_slow=False, avg_ref=True,
               double_speed=False, excl_before=True, Fz_Pz_abs_dif=False):
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
        return None, None, None

    if avg_ref:
        raw = raw.set_eeg_reference('average')

    events = mne.events_from_annotations(raw, verbose=False)
    events = events[0]
    events = events[events[:, 2] == 2]

    ds1 = 1
    picks_frontal = picks[0]
    Fz_pruned = [pick for pick in picks_frontal
                 if pick in raw.ch_names]
    picks_posterior = picks[1]
    Pz_pruned = [pick for pick in picks_posterior
                 if pick in raw.ch_names]

    data_Fz = raw.get_data(picks=Fz_pruned)
    # print(f'{data_Fz.shape=}')
    # quit()
    data_Fz_M = np.mean(data_Fz, axis=0)
    data_Pz = raw.get_data(picks=Pz_pruned)
    data_Pz_M = np.mean(data_Pz, axis=0)

    # tfr = np.abs(data_Fz_M - data_Pz_M)[None, None, :]
    # eeg_scores = np.full((1, 1, num_TRs,),
    #                      np.nan)

    abs_M = np.abs(data_Fz_M - data_Pz_M)
    # print(f'{abs_M=}')
    # quit()

    Fz_scores = np.full(num_TRs, np.nan)
    Pz_scores = np.full(num_TRs, np.nan)
    abs_scores = np.full(num_TRs, np.nan)

    for idx, event in enumerate(events):
        # true_idx = event2true[idx]
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return None, None, None
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1*(250 // ds1))
        Fz_scores[true_idx] = np.mean(data_Fz_M[t_st:t_end])
        Pz_scores[true_idx] = np.mean(data_Pz_M[t_st:t_end])
        abs_scores[true_idx] = np.mean(abs_M[t_st:t_end])



    return Fz_scores, Pz_scores, abs_scores

def conv(EEG_fluc):
    EEG_fluc = EEG_fluc.T # (TR, freq)
    import scipy.ndimage as ndimage
    HRF = get_hrf()
    # for i in range(EEG_fluc.shape[1]):
    nans, x = np.isnan(EEG_fluc), lambda z: z.nonzero()[0]
    EEG_fluc[nans] = np.interp(x(nans), x(~nans), EEG_fluc[~nans])

    EEG_fluc = ndimage.convolve1d(EEG_fluc, HRF, mode='nearest',
                                  origin=-HRF.shape[0] // 2, axis=0)
    return EEG_fluc

def test_ERP_fMRI_sn(sn='06', sess='01', avg_before=False,
                     high_gamma=False, many_ROI=True,
                     super_slow=False, avg_ref=False,
                     abs_analysis=True, all_conn=False,
                     double_speed=True,
                     Fz_Pz_abs_dif=False):
    ATL_MFG, ATL_LOC, MFG_IPL, IPL_LOC = pickle_wrap(
        get_fMRI_AP_VD, kwargs={'sn': sn, 'sess': sess,
                                'many_ROI': many_ROI,
                                'combine_regions': False,
                                'abs_analysis': abs_analysis,
                                'all_conn': all_conn,
                                'clean': True},
                            easy_override=False, verbose=-1,)



    FC_fluc = np.abs(IPL_LOC - MFG_IPL - ATL_LOC + ATL_MFG)
    # FC_fluc = np.abs(IPL_LOC + MFG_IPL - ATL_LOC - ATL_MFG)


    AP_bias = ATL_LOC + MFG_IPL - IPL_LOC - ATL_MFG
    AP = ATL_LOC + MFG_IPL
    # VD_bias = ATL_LOC + ATL_MFG - IPL_LOC - MFG_IPL
    VD = IPL_LOC + ATL_MFG

    num_TRs = ATL_MFG.shape[-1]
    picks = [['F1', 'Fz', 'F2', 'F3', 'F4',
              'FC1', 'FCz', 'FC2', 'FC3', 'FC4'],
             ['CP1', 'CPz', 'CP2', 'CP3', 'CP4',
              'P1', 'Pz', 'P2', 'P3', 'P4']]

    Fz_scores, Pz_scores, abs_scores = pickle_wrap(get_ERP_sn,
                           kwargs={'sn': sn, 'sess': sess,
                                   'num_TRs': num_TRs,
                                   'picks': picks,
                                   'avg_before': avg_before,
                                   'high_gamma': high_gamma,
                                   'super_slow': super_slow,
                                   'avg_ref': avg_ref,
                                   'double_speed': double_speed,
                                   'excl_before': True,
                                   'Fz_Pz_abs_dif': Fz_Pz_abs_dif,},
                           easy_override=False, verbose=-1)
    if Fz_scores is None:
        return None
    # Fz_scores = conv(Fz_scores)
    # Pz_scores = conv(Pz_scores)
    abs_scores = conv(abs_scores)
    dif_scores = Fz_scores - Pz_scores

    name2r = {}
    names_conn = ['ATL_MFG', 'ATL_LOC', 'MFG_IPL', 'IPL_LOC',
                  'fluc',
                  'AP_bias', 'AP',  'VD']
    names_eeg = ['Fz', 'Pz', 'abs', 'dif']
    for name_conn, conn in zip(names_conn, [ATL_MFG, ATL_LOC, MFG_IPL, IPL_LOC,
                                            FC_fluc,
                                            AP_bias, AP,  VD]):
        for name_eeg, eeg in zip(names_eeg,
                                 [Fz_scores, Pz_scores, abs_scores, dif_scores]):
            df = pd.DataFrame({name_conn: conn, name_eeg: eeg})
            df.dropna(inplace=True)
            r, p = stats.spearmanr(df[name_conn], df[name_eeg])
            name2r[(name_conn, name_eeg)] = r
    return name2r

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
    for SN in SNS:
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            print(f'- ({SN}; {SESS}) -')
            name2r = test_ERP_fMRI_sn(SN, SESS)
            if name2r is None:
                continue
            # dfs_l.append(df)

            for key, r in name2r.items():
                # NAME2L[key].append(r)
                NAME2SN2L[SN][key].append(r)

            # print(np.array(PSDs).shape)

            NAME2L = defaultdict(list)
            for SN, d in NAME2SN2L.items():
                # print(f'left: {SN}')
                for key, l in d.items():
                    NAME2L[key].append(np.mean(l))
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

                    print(f'{key} ({N=}): {M=:.3f} [{M_low:.3f}, {M_high:.3f}] '
                          f'({t=:.3f} | {d=:.3f}), {p=:.1e}, {p_wilcox=:.1e} | '
                          f'{M_above:.1%}')
            # continue
            # if 'r50' not in NAME2L:
            #     continue
            # if N == 21 and j == len(SESSES) - 1:
            #     # print('done')
            #     if 'r1' in NAME2L:
            #         plot_hz_corrs(NAME2L)
                # else:
                #     plot_regrs(pd.concat(dfs_l))
