import os.path

from nilearn.glm.first_level import compute_regressor

from EEG_fMRI.EEG_processing import get_EEG_score_sn
from EEG_fMRI.fMRI_simul_processing import get_fMRI_score_sn
from EEG_fMRI.plot_EEG_fMRI import plot_hz_corrs
# from get_HCP_act import img_data2ar

from utils import pickle_wrap
import numpy as np
import scipy.stats as stats
import pandas as pd
import warnings
from collections import defaultdict
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore', category=RuntimeWarning)

os.chdir(r'C:\PycharmProjects\SchemeRep')

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)# warnings.filterwarnings('ignore', category=pd.PerformanceWarning)

def get_hrf(tr=2.1):
    onset, amplitude, duration = 0.0, 1.0, 0.1
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    frame_times = np.arange(20) * tr
    signal, _labels = compute_regressor(
        exp_condition,
        'spm',
        frame_times,
        con_id="main",
        oversampling=50,
        # min
    )
    return signal[:, 0]



def test_EEG_fMRI_sn(sn='06', sess='01', double_speed=True,
                     get_max=False, get_max_avg_before=True,
                     log_freqs=False, just_frontal=True, alt_v=0):

    assert not (get_max and get_max_avg_before), 'Only have one true'

    fMRI_fluc, alt1_signed, alt2_abs_sum, alt3_sum, alt4_sum_abs = pickle_wrap(
        get_fMRI_score_sn, kwargs={'sn': sn, 'sess': sess, 'combine_regions': False,
                                   'clean': True, 'mask': True, 'nofilter': False,
                                   'n_compcor': 5, 'lateral': False},
                            easy_override=True, verbose=-1, )
    # r1, _ = stats.spearmanr(fMRI_fluc, alt1_signed)
    # print(f'Sanity 1: {r1=:.3f}')
    # r2, _ = stats.spearmanr(fMRI_fluc, alt2_abs_sum)
    # print(f'Sanity 2: {r2=:.3f}')
    # r3, _ = stats.spearmanr(fMRI_fluc, alt3_sum_abs)
    # print(f'Sanity 3: {r3=:.3f}')
    # return None, None

    if fMRI_fluc is None:
        print(f'None fMRI fluc ({sn}; {sess}) !')
        return None, None

    num_TRs = fMRI_fluc.shape[-1]

    if just_frontal:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4']
    else:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4',
                 'FC1', 'FCz', 'FC2', 'FC3', 'FC4',
                 'C1', 'Cz', 'C2', 'C3', 'C4',
                 'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
                 'P1', 'Pz', 'P2', 'P3', 'P4']

    if log_freqs:
        custom_freqs = np.logspace(-1, 1.7, 100, base=10)
    else:
        custom_freqs = np.linspace(0.5, 50, 100)

    custom_freqs = tuple(custom_freqs)

    # kw2 = {'sn': sn, 'sess': sess,
    #                                'num_TRs': num_TRs,
    #                                'picks': picks,
    #                                'avg_before': False,
    #                                'high_gamma': False,
    #                                'super_slow': False,
    #                                'avg_ref': True,
    #                                'mastoid_ref': False,
    #                                'fz_minus_pz': False,
    #                                'delay': False,
    #                                'double_speed': double_speed,
    #                                'excl_before': True,
    #                                'Fz_Pz_abs_dif': False,
    #                                'get_max': get_max,
    #                                'get_max_avg_before': get_max_avg_before,
    #                                'custom_freqs': custom_freqs}

    kw = {'sn': sn, 'sess': sess, 'avg_before': False,
          'num_TRs': num_TRs, 'picks': picks, 'avg_ref': True,
          'mastoid_ref': False, 'delay': False,
          'double_speed': double_speed, 'excl_before': True,
          'get_max': get_max, 'get_max_avg_before': get_max_avg_before,
          'custom_freqs': custom_freqs}
    # for key, val in kw2.items():
    #     if key in kw:
    #         assert kw[key] == val

    EEG_fluc = pickle_wrap(get_EEG_score_sn,
                           kwargs=kw,
                           easy_override=False, verbose=-1)
    # print(EEG_fluc)
    # quit()

    if EEG_fluc is None:
        print(f'BAD EEG!! ({sn}; {sess})')
        return None, None

    fMRI_fluc = fMRI_fluc[6:]
    if len(alt3_sum) > len(fMRI_fluc):
        alt1_signed = alt1_signed[6:]
        alt2_abs_sum = alt2_abs_sum[6:]
        alt3_sum = alt3_sum[6:]
        alt4_sum_abs = alt4_sum_abs[6:]

    # num_nans = np.isnan(EEG_fluc[0, 4]).sum()

    EEG_fluc = np.nanmean(EEG_fluc, axis=0) # (freq, TR)
    EEG_fluc = EEG_fluc.T # (TR, freq)

    EEG_fluc = EEG_fluc[6:, :] # drop edge artifact
    EEG_fluc = conv(EEG_fluc)

    if custom_freqs is not None:
        ranges = {}
        for idx, freq in enumerate(custom_freqs):
            ranges[f'r{freq:.1f}'] = (idx, idx + 1)
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

        if double_speed:
            ranges_ = {f'r{hz}': (hz, hz + 1) for hz in range(1, 101)}
        else:
            ranges_ = {f'r{hz}': (hz, hz + 1) for hz in range(1, 51)}
        ranges.update(ranges_)

    ranges = {}
    for idx, freq in enumerate(custom_freqs):
        ranges[f'r{freq:.1f}'] = (idx, idx + 1)
    ranges_bins = {'delta': (1, 4), 'theta': (4, 8),
                   'alpha': (8, 13), 'beta': (13, 30),
                   'gamma': (30, 50.5),}
    custom_freqs = np.array(custom_freqs)
    for key, tup in ranges_bins.items():
        low = np.argmin(np.abs(custom_freqs - tup[0]))
        high = np.argmin(np.abs(custom_freqs - tup[1])) + 1
        ranges[key] = (low, high)
    #     print(f'{key}: {low=}, {high=}')
    # quit()
        # for freq in custom_freqs:
        # ranges[key] = (int(tup[0] * 2), int(tup[1] * 2))
    # print(ranges)
    # quit()
    # ranges.update(ranges_bins)

    name2fluc = {}
    name2r = {}


    for name, rng in ranges.items():
        df = pd.DataFrame({'fMRI_fluc': fMRI_fluc})
        df['sn'] = sn
        df['sess'] = sess

        idxs = np.arange(*rng)
        # print(name)
        # print(rng)
        # print(idxs)
        idxs -= 1
        # print(idxs)
        # if name == 'gamma':
        #     range_fluc = EEG_fluc[:, idxs[0]:].mean(axis=1)
        # else:
        range_fluc = EEG_fluc[:, idxs].mean(axis=1)
        df[name] = range_fluc
        df['focus'] = df[name]
        df['ctrl'] = np.nanmean(stats.zscore(EEG_fluc, axis=1), axis=1)

        df['focus'] = stats.zscore(df['focus'])
        df['fMRI_fluc'] = stats.zscore(df['fMRI_fluc'])
        if alt_v > 0:
            if alt_v == 1:
                df['ctrl'] = stats.zscore(alt1_signed)
            elif alt_v == 2:
                df['ctrl'] = stats.zscore(alt2_abs_sum)
            elif alt_v == 3:
                df['ctrl'] = stats.zscore(alt3_sum)
            elif alt_v == 4:
                df['ctrl'] = stats.zscore(alt4_sum_abs)
        df_ = df.dropna()
        if len(df_) < 1:
            name2r[name] = np.nan
            continue

        if alt_v == 0:
            r, p = stats.spearmanr(df_[name], df_['fMRI_fluc'])
        else:
            res = smf.ols(f'focus ~ fMRI_fluc + ctrl', data=df_).fit()
            r = res.params['fMRI_fluc']
        name2r[name] = r

        name2fluc[name] = range_fluc

        r = np.arctanh(r)
        if 'r' != name[0]:
            print(f'{name}: {r=:.3f}, {p=:.3f}')
        name2r[name] = r

    # l = np.nanmean([r for r in name2r.values()])
    # for name, r in name2r.items():
    #     name2r[name] = r - l
    return name2r, df#, psd



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

def get_sess_names(only_rs=True):
    SESSES = ['01_task-rest', '02_task-rest']
    SESS_INK = ['01_task-inscapes', '02_task-inscapes']  # shape things
    if only_rs:
        return SESSES + SESS_INK
    SESS_OTHER = ['01_task-checker', # Designed to induce visual effects
                  '01_task-dme_run-01', '01_task-dme_run-02', # dispicable me
                  '01_task-monkey1_run-01', '01_task-monkey1_run-02', # movie
                  '01_task-tp_run-01', '01_task-tp_run-02' # "The present"
                  ]
    SESS_OTHER += [f'02' + sess[2:] for sess in SESS_OTHER]
    return SESSES + SESS_INK + SESS_OTHER


def do_EEG_fMRI_test():
    SNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22']
    SESSES = get_sess_names()

    # A fair number of bad/missing datasets.
    #   I double-checked to make sure I downloaded this right.
    #   They just weren't in the data. This is mentioned in the Methods.
    BAD_SNS = {('06', '02_task-rest'), ('12', '01_task-rest'),
               ('16', '02_task-rest'), ('18', '01_task-rest'),
               ('06', '02_task-inscapes'), ('07', '01_task_inscapes'),
               ('09', '01_task-inscapes'), ('12', '01_task-inscapes'),
               ('15', '02_task-inscapes'), ('18', '01_task_inscapes'),
               } # Bad EEG alignment

    NAME2SN2L = defaultdict(lambda: defaultdict(list))
    dfs_l = []
    for SN in SNS:
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

                    try:
                        res = stats.wilcoxon(l)
                        p_wilcox = res.pvalue * 2
                        M_above = np.mean([m > 0 for m in l])
                        # print(f'{key}: {l=}')
                    except ValueError:
                        p_wilcox = np.nan
                        M_above = np.nan

                    print(f'{key} ({N=}): {M=:.3f} [{M_low:.3f}, {M_high:.3f}] '
                          f'({t=:.3f} | {d=:.3f}), {p=:.1e}, {p_wilcox=:.1e} | '
                          f'{M_above:.1%}')
            # continue
            if 'r50.0' not in NAME2L:
                continue
            if N == 21 and j == len(SESSES) - 1:
                print('PLOTTTT')
                plot_hz_corrs(NAME2L)
                plot_hz_corrs(NAME2L, effect_size=True, plot_se=True)


if __name__ == '__main__':
    do_EEG_fMRI_test()


