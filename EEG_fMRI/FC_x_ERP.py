from Study3.run_analysis_Study3 import conv
from Study3.fMRI_simultaneous_funcs import get_fMRI_ar
from Study3.EEG_funcs import load_EEG, get_EEG_score_sn
from atlas_utils import get_atlas

from utils import pickle_wrap, stdize
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd
from collections import defaultdict

import os
try:
    import statsmodels.formula.api as smf
except ModuleNotFoundError:
    pass


ROOT_EEG_FMRI = fr'G:\EEG_fMRI'

def get_fMRI_AP_VD(sn, sess='01', combine_regions=True, clean=False,
                   many_ROI=True):

    ar_fMRI, key2idxs = pickle_wrap(get_fMRI_ar,
                          kwargs={'sn': sn, 'sess': sess,
                                  'combine_regions': combine_regions,
                                  'clean': clean},
                          easy_override=False, verbose=0,)


    if ar_fMRI is None:
        return None, None, None, None

    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    ROI2idx = {ROI: [] for ROI in atlas['ROI_regions']}
    for i, region in enumerate(atlas['ROI_regions']):
        ROI2idx[region].append(i)

    if many_ROI:
        key2idxs['MFG'] = ROI2idx['MFG'] + ROI2idx['IFG']
        key2idxs['IPL'] = ROI2idx['IPL']
        key2idxs['LOC'] = ROI2idx['LOC'] + ROI2idx['sOcG'] + ROI2idx['EVC']
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

def get_ERP_sn(sn, num_TRs, sess='01', picks=None, avg_before=True,
               high_gamma=False, super_slow=False, avg_ref=True,
               double_speed=False, excl_before=True, Fz_Pz_abs_dif=False,
               mastoid_ref=False):
    assert not (high_gamma and super_slow)
    if picks is None:
        picks = ['Fz', 'Cz', 'Pz',
                 'F1', 'C1', 'P1',
                 'F2', 'C2', 'P2']
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_eeg = fr'{root_sn}\eeg'

    kw = {'sn': sn, 'sess': sess, 'num_TRs': num_TRs, 'dir_eeg': dir_eeg,
          }
    raw, event2true, boundary_events = pickle_wrap(load_EEG, kwargs=kw,
                                                   easy_override=False)

    if raw is None:
        return [None] * 13

    if mastoid_ref:
        raw = raw.set_eeg_reference(['T7', 'T8'])
    elif avg_ref:
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

    data_Fz_M = np.mean(data_Fz, axis=0)
    data_Pz = raw.get_data(picks=Pz_pruned)
    data_Pz_M = np.mean(data_Pz, axis=0)

    # tfr = np.abs(data_Fz_M - data_Pz_M)[None, None, :]
    # eeg_scores = np.full((1, 1, num_TRs,),
    #                      np.nan)

    data_Fz_M_padded = np.pad(data_Fz_M, pad_width=62, mode='constant',
                              constant_values=np.nan)
    data_Pz_M_padded = np.pad(data_Pz_M, pad_width=62, mode='constant',
                              constant_values=np.nan) # temporal resolution ~250 ms

    abs_M_forward = np.abs(data_Fz_M - data_Pz_M_padded[:-124]) # confused fwd and bkwd
    abs_M_backward = np.abs(data_Fz_M - data_Pz_M_padded[124:])
    # dif_M_forward = data_Fz_M - data_Pz_M_padded[:-124]
    # dif_M_backward = data_Fz_M - data_Pz_M_padded[124:]
    # abs_M_forward = dif_M_forward
    # abs_M_backward = dif_M_backward
    sum_M_forward = np.abs(data_Fz_M + data_Pz_M_padded[:-124])
    sum_M_backward = np.abs(data_Fz_M + data_Pz_M_padded[124:])

    # Fz_forward = data_Fz_M - data_Fz_M_padded[:-124]
    # Pz_forward = data_Pz_M - data_Pz_M_padded[:-124]

    # print(data_Pz_M_forward.shape)
    abs_M = np.abs(data_Fz_M - data_Pz_M)
    Fz_Pz_sum = np.abs(data_Fz_M + data_Pz_M)

    Fz_Pz_dif = data_Fz_M - data_Pz_M

    Fz_scores = np.full(num_TRs, np.nan)
    Pz_scores = np.full(num_TRs, np.nan)
    abs_scores = np.full(num_TRs, np.nan)
    abs_scores_fwd = np.full(num_TRs, np.nan)
    abs_scores_bwd = np.full(num_TRs, np.nan)
    Fz_Pz_dif_scores = np.full(num_TRs, np.nan)
    Fz_Pz_sum_scores = np.full(num_TRs, np.nan)
    sum_scores_fwd = np.full(num_TRs, np.nan)
    sum_scores_bwd = np.full(num_TRs, np.nan)
    peak2peak = np.full(num_TRs, np.nan)
    peak2peak_rev = np.full(num_TRs, np.nan)
    min_Fz = np.full(num_TRs, np.nan)
    max_Pz = np.full(num_TRs, np.nan)

    for idx, event in enumerate(events):
        # true_idx = event2true[idx]
        try:
            true_idx = event2true[idx]
        except KeyError as e:
            print(f'{e=}')
            return [None] * 13
        if true_idx in boundary_events:
            continue
        t_st = event[0]
        t_end = t_st + int(2.1*(250 // ds1))
        Fz_scores[true_idx] = np.mean(data_Fz_M[t_st:t_end])
        Pz_scores[true_idx] = np.mean(data_Pz_M[t_st:t_end])
        abs_scores[true_idx] = np.mean(abs_M[t_st:t_end])
        abs_scores_fwd[true_idx] = np.mean(abs_M_forward[t_st:t_end])
        abs_scores_bwd[true_idx] = np.mean(abs_M_backward[t_st:t_end])
        Fz_Pz_dif_scores[true_idx] = np.mean(Fz_Pz_dif[t_st:t_end])
        Fz_Pz_sum_scores[true_idx] = np.mean(Fz_Pz_sum[t_st:t_end])
        sum_scores_fwd[true_idx] = np.mean(sum_M_forward[t_st:t_end])
        sum_scores_bwd[true_idx] = np.mean(sum_M_backward[t_st:t_end])

        Fz_low = np.min(data_Fz_M[t_st:t_end])
        Fz_high = np.max(data_Fz_M[t_st:t_end])
        Pz_high = np.max(data_Pz_M[t_st:t_end])
        Pz_low = np.min(data_Pz_M[t_st:t_end])
        peak2peak[true_idx] = Pz_high - Fz_low
        peak2peak_rev[true_idx] = Fz_high - Pz_low
        min_Fz[true_idx] = Fz_low
        max_Pz[true_idx] = Pz_high


        # peak2peak[true_idx] = Fz_high
        # peak2peak_rev[true_idx] = Pz_high

        # peak2peak_rev[true_idx] = Fz_peak - Pz_peak

    return (Fz_scores, Pz_scores, abs_scores, abs_scores_fwd, abs_scores_bwd,
            Fz_Pz_sum_scores, Fz_Pz_dif_scores, sum_scores_fwd, sum_scores_bwd,
            peak2peak, peak2peak_rev, min_Fz, max_Pz)


AUTOCORR = []

def test_ERP_fMRI_sn(sn='06', sess='01', avg_before=False,
                     high_gamma=False, many_ROI=True,
                     super_slow=False, avg_ref=False,
                     abs_analysis=True, all_conn=False,
                     double_speed=True, Fz_Pz_abs_dif=False,
                     override=True, fz_minus_pz=False):

    ATL_MFG, ATL_LOC, MFG_IPL, IPL_LOC = pickle_wrap(
        get_fMRI_AP_VD, kwargs={'sn': sn, 'sess': sess,
                                'many_ROI': many_ROI,
                                'combine_regions': False,
                                'abs_analysis': abs_analysis,
                                'all_conn': all_conn,
                                'clean': True},
                            easy_override=False, verbose=-1,)
    if ATL_MFG is None:
        print('No ATL_MFG!')
        return None



    num_TRs = ATL_MFG.shape[-1]
    picks = [['F1', 'Fz', 'F2', 'F3', 'F4',
              'FC1', 'FCz', 'FC2', 'FC3', 'FC4'
              ],
             ['CP1', 'CPz', 'CP2', 'CP3', 'CP4',
              'P1', 'Pz', 'P2', 'P3', 'P4'
              ]]

    (Fz_scores, Pz_scores, abs_scores, abs_scores_fwd, abs_scores_bwd,
     Fz_Pz_sum_scores, Fz_Pz_dif, sum_scores_fwd, sum_scores_bwd,
     peak2peak, peak2peak_rev, min_Fz, max_Pz) = (
        pickle_wrap(get_ERP_sn,
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
                           easy_override=False, verbose=-1))

    if Fz_scores is None:
        print('No Fz!')
        return None

    # picks = ['F1', 'Fz', 'F2', 'F3', 'F4',
    #          'FC1', 'FCz', 'FC2', 'FC3', 'FC4',
    #          'C1', 'Cz', 'C2', 'C3', 'C4',
    #          'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
    #          'P1', 'Pz', 'P2', 'P3', 'P4']

    if Fz_Pz_abs_dif or fz_minus_pz:
        picks = [['F1', 'Fz', 'F2', 'F3', 'F4',
                  'FC1', 'FCz', 'FC2', 'FC3', 'FC4'
                  ],
                 ['CP1', 'CPz', 'CP2', 'CP3', 'CP4',
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
                                   'fz_minus_pz': fz_minus_pz,
                                   'delay': False,
                                   'double_speed': double_speed,
                                   'excl_before': True,
                                   'Fz_Pz_abs_dif': Fz_Pz_abs_dif, },
                           easy_override=False, verbose=-1)

    global AUTOCORR
    df_ = pd.DataFrame({'a': peak2peak[:-1], 'b': peak2peak[1:]})
    r, p = stats.spearmanr(df_['a'], df_['b'], nan_policy='omit')
    AUTOCORR.append(r)
    M_autocorr = np.nanmean(AUTOCORR)

    EEG_fluc = np.nanmean(EEG_fluc, axis=0)
    # print(EEG_fluc.shape)

    EEG_fluc = EEG_fluc#.T # (TR, freq)
    EEG_fluc = EEG_fluc[:, 4:]


    EEG_fluc = conv(EEG_fluc.T)
    # theta_fluc = EEG_fluc

    # EEG_fluc = conv(EEG_fluc.T)
    theta_fluc = np.nanmean(EEG_fluc[:, :11], axis=1)
    gamma_fluc = np.nanmean(EEG_fluc[:, 59:], axis=1)


    IPL_LOC = stats.zscore(IPL_LOC, nan_policy='omit')
    ATL_LOC = stats.zscore(ATL_LOC, nan_policy='omit')
    MFG_IPL = stats.zscore(MFG_IPL, nan_policy='omit')
    ATL_MFG = stats.zscore(ATL_MFG, nan_policy='omit')

    ATL_MFG = ATL_MFG[4:]
    ATL_LOC = ATL_LOC[4:]
    MFG_IPL = MFG_IPL[4:]
    IPL_LOC = IPL_LOC[4:]

    FC_fluc = np.abs(IPL_LOC - MFG_IPL - ATL_LOC + ATL_MFG)
    AP_bias = ATL_LOC + MFG_IPL - IPL_LOC - ATL_MFG
    AP = ATL_LOC + MFG_IPL
    VD = IPL_LOC + ATL_MFG
    test_abs = np.abs(ATL_MFG - ATL_LOC)

    min_Fz = min_Fz[4:]
    # print(Pz_scores)
    # print(np.diff(Pz_scores))
    # quit()
    # max_Pz = np.array([np.nan] + list(np.diff(Pz_scores)[4:]))
    # print(max_Pz)
    # quit()
    max_Pz = max_Pz[4:]
    # print(Pz_scores[:3])
    # print(np.diff(Pz_scores[:3]))
    # print(Pz_scores[1] - Pz_scores[0])
    # quit()


    Fz_scores = Fz_scores[4:]
    Pz_scores = Pz_scores[4:]
    abs_scores = abs_scores[4:]
    abs_scores_fwd = abs_scores_fwd[4:]
    abs_scores_bwd = abs_scores_bwd[4:]
    Fz_Pz_sum_scores = Fz_Pz_sum_scores[4:]
    Fz_Pz_dif = Fz_Pz_dif[4:]
    sum_scores_fwd = sum_scores_fwd[4:]
    sum_scores_bwd = sum_scores_bwd[4:]
    peak2peak = peak2peak[4:]
    peak2peak_rev = peak2peak_rev[4:]

    # print(f'{M_autocorr=:.3f}')
    # plt.plot(min_Fz)
    # plt.show()
    # quit()

    Fz_scores = conv(Fz_scores)
    Pz_scores = conv(Pz_scores)
    abs_scores = conv(abs_scores)
    sum_scores = conv(Fz_Pz_sum_scores)
    abs_scores_fwd = conv(abs_scores_fwd)
    abs_scores_bwd = conv(abs_scores_bwd)
    sum_scores_fwd = conv(sum_scores_fwd)
    sum_scores_bwd = conv(sum_scores_bwd)
    peak2peak = conv(peak2peak)
    peak2peak_rev = conv(peak2peak_rev)
    min_Fz = conv(min_Fz)
    max_Pz = conv(max_Pz)



    # abs_scores = np.abs(Fz_scores - Pz_scores)
    # dif_scores = conv(dif_scores)
    #

    Pz_abs = np.abs(Pz_scores)
    Fz_abs = np.abs(Fz_scores)
    dif_scores = conv(Fz_Pz_dif)
    # print(dif_scores.shape)
    # quit()
    # plt.plot(dif_scores)
    # plt.show()
    # quit()

    name2r = {}
    names_conn = ['ATL_MFG', 'ATL_LOC', 'MFG_IPL', 'IPL_LOC',
                  'fluc',  'AP_bias', 'AP',  'VD']
    names_eeg = ['Fz', 'Pz', 'abs', 'abs_fwd', 'abs_bwd', 'dif']

    # names_conn = ['AP_bias']
    names_conn = ['theta']#, 'fluc', 'VD', 'AP']
    names_eeg = ['Pz_abs', 'Fz_abs', 'abs']

    name2conn = {'ATL_MFG': ATL_MFG, 'ATL_LOC': ATL_LOC,
                 'MFG_IPL': MFG_IPL, 'IPL_LOC': IPL_LOC,
                 'fluc': FC_fluc, 'AP_bias': AP_bias,
                 'AP': AP, 'VD': VD, 'TEST': test_abs,
                 'theta': theta_fluc}
    name2eeg = {'Fz': Fz_scores, 'Pz': Pz_scores,
                'Fz_pz': Fz_scores * Pz_scores,
                'abs': abs_scores, 'abs_fwd': abs_scores_fwd,
                'abs_bwd': abs_scores_bwd, #'dif': dif_scores,
                'abs_Fz': Fz_Pz_sum_scores, 'dif': Fz_Pz_dif,
                'Fz_abs': Fz_abs, 'Pz_abs': Pz_abs
                }

    for name_conn in names_conn:
        conn = name2conn[name_conn]
        for name_eeg in names_eeg:
            if override:
                df = pd.DataFrame({#'abs_scores_fwd': abs_scores_fwd,
                                   # 'sum_scores': max_Pz,
                                   # 'abs_scores': min_Fz,
                                   'peak2peak_rev': peak2peak_rev,
                                   'peak2peak': peak2peak,
                                   'max_Pz': max_Pz,
                                   'min_Fz': min_Fz,
                                   'theta': theta_fluc,
                                   'gamma': gamma_fluc,
                                   'conn': FC_fluc,
                                   'AP': AP, 'VD': VD, 'AP_bias': AP_bias})
                df.dropna(inplace=True)
                # df = stats.zscore(df, axis=0)
                for col in df.columns:
                    df[col] = stdize(df[col], rankdata=True)
                    # df[col] = stats.zscore(df[col])

                mod = smf.ols(formula='conn ~ 1 + peak2peak + peak2peak_rev',
                              data=df)

                # df['trial'] = [str(i) for i in range(len(df))]
                # df_fwd = df.copy()
                # df_fwd['dir'] = 0.5
                # df_bwd = df.copy()
                # df_bwd['peak2peak'] = df_bwd['peak2peak_rev']
                # df_bwd['dir'] = -0.5
                # df_new = pd.concat([df_fwd, df_bwd]).reset_index()
                # # df_new['peak2peak'].fillna(df_new['peak2peak_rev'], inplace=True)
                # df_new['peak2peak'] = stats.zscore(df_new['peak2peak'],
                #                                    nan_policy='omit')
                #
                # mod = smf.ols(formula='peak2peak ~ 1 + theta*dir',
                #               data=df_new)
                res = mod.fit()

                # print(res.summary())
                # quit()

                r = res.params['peak2peak']
                if name_eeg == 'Pz_abs':
                    r = res.params['peak2peak_rev']
            else:
                eeg = name2eeg[name_eeg]
                df = pd.DataFrame({name_conn: conn, name_eeg: eeg})
                df.dropna(inplace=True)
                r, p = stats.spearmanr(df[name_conn], df[name_eeg])

            name2r[(name_conn, name_eeg)] = r
    return name2r

def get_sess_setup():
    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    SESSES = ['01_task-rest', '02_task-rest',
              '01_task-inscapes', '02_task-inscapes'] # shape things

    SESS_OTHER = ['01_task-checker', # Designed to induce visual effects
                  '01_task-dme_run-01', '01_task-dme_run-02', # dispicable me
                  '01_task-monkey1_run-01', '01_task-monkey1_run-02', # movie
                  # '01_task-peer', # used for E-T calibration
                  '01_task-tp_run-01', '01_task-tp_run-02' # "The present"
                  ]
    SESS_OTHER += [f'02' + sess[2:] for sess in SESS_OTHER]
    # SESSES += SESS_OTHER
    # SESSES = SESS_OTHER


    BAD_SNS = {('06', '02_task-rest'), ('12', '01_task-rest'),
               ('16', '02_task-rest'), ('18', '01_task-rest'),
               ('06', '02_task-inscapes'), ('07', '01_task_inscapes'),
               ('09', '01_task-inscapes'), ('12', '01_task-inscapes'),
               ('15', '02_task-inscapes'), ('18', '01_task_inscapes')} # Bad EEG alignment
    NAME2L = defaultdict(list)
    NAME2SN2L = defaultdict(lambda: defaultdict(list))
    return SNS, SESSES, BAD_SNS, NAME2L, NAME2SN2L

if __name__ == '__main__':
    SNS, SESSES, BAD_SNS, NAME2L, NAME2SN2L = get_sess_setup()
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
                          f'({t=:.3f} | {d=:.3f}), {p=:.2e}, {p_wilcox=:.2e} | '
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
