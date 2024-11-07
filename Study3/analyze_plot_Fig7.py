import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

from nilearn.glm.first_level import compute_regressor

from Study3.EEG_simultaneous_funcs import get_EEG_score_sn
from Study3.fMRI_simultaneous_funcs import get_fMRI_score_sn
from Study3.plot_fMRI_EEG_funcs import plot_hz_corrs

from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
import scipy.stats as stats
import pandas as pd
import warnings
from collections import defaultdict
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore', category=RuntimeWarning)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


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
    )
    return signal[:, 0]


def test_EEG_fMRI_sn(sn='06', sess='01', double_speed=True,
                     log_freqs=False, just_frontal=True,
                     ctrl_SuppMat=False):
    fMRI_fluc, alt1_signed, alt2_abs_sum, alt3_sum = pickle_wrap(
        get_fMRI_score_sn, kwargs={'sn': sn, 'sess': sess, },
        easy_override=True, verbose=-1, )

    if fMRI_fluc is None:
        print(f'None fMRI fluc ({sn}; {sess}) !')
        return None, None

    num_TRs = fMRI_fluc.shape[-1]

    if just_frontal:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4', ]
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

    kw = {'sn': sn, 'sess': sess,
          # 'avg_before': False,
          'num_TRs': num_TRs, 'picks': picks,
          # 'avg_ref': True,
          # 'mastoid_ref': False, 'delay': False,
          # 'double_speed': double_speed, 'excl_before': True,
          'custom_freqs': custom_freqs}

    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs=kw,
                           easy_override=False, verbose=-1)

    if EEG_fluc is None:
        print(f'BAD EEG!! ({sn}; {sess})')
        return None, None

    fMRI_fluc = fMRI_fluc[6:]
    if len(alt3_sum) > len(fMRI_fluc):
        alt1_signed = alt1_signed[6:]
        alt2_abs_sum = alt2_abs_sum[6:]
        alt3_sum = alt3_sum[6:]

    EEG_fluc = np.nanmean(EEG_fluc, axis=0)  # (freq, TR)
    EEG_fluc = EEG_fluc.T  # (TR, freq)

    EEG_fluc = EEG_fluc[6:, :]  # drop edge artifact
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
                ranges[key] = (int(tup[0] * 2), int(tup[1] * 2))

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
                   'gamma': (30, 50.5), }
    custom_freqs = np.array(custom_freqs)
    for key, tup in ranges_bins.items():
        low = np.argmin(np.abs(custom_freqs - tup[0]))
        high = np.argmin(np.abs(custom_freqs - tup[1])) + 1
        ranges[key] = (low, high)

    name2fluc = {}
    name2r = {}

    for name, rng in ranges.items():
        df = pd.DataFrame({'fMRI_fluc': fMRI_fluc})
        df['sn'] = sn
        df['sess'] = sess

        idxs = np.arange(*rng)

        idxs -= 1

        range_fluc = EEG_fluc[:, idxs].mean(axis=1)
        df[name] = range_fluc
        df['focus'] = df[name]
        df['ctrl'] = np.nanmean(stats.zscore(EEG_fluc, axis=1), axis=1)

        df['focus'] = stats.zscore(df['focus'])
        df['fMRI_fluc'] = stats.zscore(df['fMRI_fluc'])
        if ctrl_SuppMat:
            df['ctrl_one'] = stats.zscore(alt1_signed)
            df['ctrl_two'] = stats.zscore(alt2_abs_sum)
            df['ctrl_three'] = stats.zscore(alt3_sum)

        df_ = df.dropna()
        if len(df_) < 1:
            name2r[name] = np.nan
            continue

        if ctrl_SuppMat:
            res = smf.ols(f'focus ~ fMRI_fluc + ctrl_one + ctrl_two + ctrl_three',
                          data=df_).fit()
            r = res.params['fMRI_fluc']
        else:
            r, p = stats.spearmanr(df_[name], df_['fMRI_fluc'])

        name2r[name] = r

        name2fluc[name] = range_fluc

        r = np.arctanh(r)
        if 'r' != name[0]:
            print(f'{name}: {r=:.3f}')
        name2r[name] = r

    return name2r, df


def conv(EEG_fluc):
    if len(EEG_fluc.shape) == 1:
        EEG_fluc = EEG_fluc[:, None]
        undo = True

    else:
        undo = False

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
    SESS_OTHER = ['01_task-checker',  # Designed to induce visual effects
                  '01_task-dme_run-01', '01_task-dme_run-02',  # dispicable me
                  '01_task-monkey1_run-01', '01_task-monkey1_run-02',  # movie
                  '01_task-tp_run-01', '01_task-tp_run-02'  # "The present"
                  ]
    SESS_OTHER += [f'02' + sess[2:] for sess in SESS_OTHER]
    return SESSES + SESS_INK + SESS_OTHER


def do_EEG_fMRI_test(just_frontal=True):
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
               }  # Bad EEG alignment

    NAME2SN2L = defaultdict(lambda: defaultdict(list))
    dfs_l = []
    # SNS = SNS[-10:]
    for SN in SNS:
        for j, SESS in enumerate(SESSES):
            if SN in ['01', '02', '03', '09', '18'] and '02' in SESS: continue
            if (SN, SESS) in BAD_SNS: continue
            print(f'- ({SN}; {SESS}) -')
            name2r, df = test_EEG_fMRI_sn(SN, SESS, just_frontal=just_frontal)
            if name2r is None:
                continue
            dfs_l.append(df)

            for key, r in name2r.items():
                NAME2SN2L[SN][key].append(r)

            NAME2L = defaultdict(list)
            simple2l = defaultdict(list)
            for SN, d in NAME2SN2L.items():
                for key, l in d.items():
                    NAME2L[key].append(np.mean(l))
                    simple2l[key].extend(l)

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
                    except ValueError:
                        p_wilcox = np.nan
                        M_above = np.nan

                    print(f'{key} ({N=}): {M=:.3f} [{M_low:.3f}, {M_high:.3f}] '
                          f'({t=:.3f} | {d=:.3f}), {p=:.1e}, {p_wilcox=:.1e} | '
                          f'{M_above:.1%}')

            if 'r50.0' not in NAME2L:
                continue
            if N == 21 and j == len(SESSES) - 1:
                plot_hz_corrs(NAME2L, effect_size=True, plot_se=True,
                              just_frontal=just_frontal)


if __name__ == '__main__':
    do_EEG_fMRI_test()
