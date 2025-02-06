import matplotlib.pyplot as plt
from spacy.tokens.doc import defaultdict

from org_sns import get_sns
from organize_bhv import get_trial_info
from scipy import stats
import numpy as np

def analyze_rt_x_inc_neu():
    age2sn = get_sns('obj7_fMRI', sh=False)
    sns = age2sn['healthy']
    rs = []
    inc2per2rt = defaultdict(lambda: defaultdict(list))
    for sn in sns:
        df_sn = get_trial_info(sn, ret=False)
        for (inc, per), df_ip in df_sn.groupby(['inc', 'per_inc']):
            rt_cond = np.nanmean(df_ip['inc_rt'])
            inc2per2rt[int(inc)][int(per)].append(rt_cond)
        df_sn = df_sn[df_sn['inc'] == 2]
        rts = df_sn['inc_rt']
        per_inc = df_sn['per_inc']
        r, p = stats.spearmanr(rts, per_inc, nan_policy='omit')
        rs.append(r)
        print(f'{sn} | {r=:.3f}')
    rs = np.array(rs)
    N = np.sum(~np.isnan(rs))
    M = np.nanmean(rs)
    SE = np.nanstd(rs, ddof=1) / np.sqrt(N)
    t, p = stats.ttest_1samp(rs, 0, nan_policy='omit')
    print(f'{M=:.3f}, {SE=:.3f}, t[{N - 1}] = {t:.2f}, {p=:.3f}')

    for inc, per2rt in inc2per2rt.items():
        for per, rts in per2rt.items():
            rts = np.array(rts)
            N = np.sum(~np.isnan(rts))
            M = np.nanmean(rts)
            SE = np.nanstd(rts, ddof=1) / np.sqrt(N)
            print(f'({inc}, {per}) {M=:.3f}, {SE=:.3f}')

def simple_rt_comparison(target='inc'):
    age2sn = get_sns('obj7_fMRI', sh=False)
    sns = age2sn['healthy']
    inc2rt = {1: [], 2: [], 3: [], 4: []}
    for sn in sns:
        df_sn = get_trial_info(sn, ret=False)
        if target == 'inc':
            df_sn = df_sn[((df_sn['inc'] <= 2) & (df_sn['per_inc'] == 1)) |
                          ((df_sn['inc'] >= 2) & (df_sn['per_inc'] == 4)) |
                          ((df_sn['inc'] == 2))]
        df_grp = df_sn.groupby(target)['inc_rt'].mean()
        if target == 'per_inc': assert len(df_grp) == 4
        for i, val in df_grp.to_dict().items():
            inc2rt[int(i)].append(val)

    for i, l in inc2rt.items():
        print(f'{i}: {len(l)=}')

    keys = [i for i, l in inc2rt.items() if len(l) > 0]
    if len(keys) == 3:
        t12, p12 = stats.ttest_rel(inc2rt[1], inc2rt[2], nan_policy='omit')
        d = t12 / np.sqrt(len(inc2rt[1]))
        print(f'Inc vs. Neu: {t12=:.3f}, {p12=:.3f}, {d=:.3f}')
        t23, p23 = stats.ttest_rel(inc2rt[2], inc2rt[3], nan_policy='omit')
        d = t23 / np.sqrt(len(inc2rt[1]))
        print(f'Neu vs. Con: {t23=:.3f}, {p23=:.3f}, {d=:.3f}')
        t13, p13 = stats.ttest_rel(inc2rt[1], inc2rt[3], nan_policy='omit')
        d = t13 / np.sqrt(len(inc2rt[1]))
        print(f'Inc vs. Con: {t13=:.3f}, {p13=:.3f}, {d=:.3f}')
    else:
        for i in keys:
            for j in keys:
                if i >= j:
                    continue
                i_M = np.nanmean(inc2rt[i])
                j_M = np.nanmean(inc2rt[j])
                t, p = stats.ttest_rel(inc2rt[i], inc2rt[j], nan_policy='omit')
                d = t / np.sqrt(len(inc2rt[i]))
                print(f'{i} ({i_M:.3f}) vs. {j} ({j_M:.3f}): '
                      f'{t=:.3f}, {p=:.3f} | {d=:.3f}')

if __name__ == '__main__':
    simple_rt_comparison()
    # analyze_rt_x_inc_neu()