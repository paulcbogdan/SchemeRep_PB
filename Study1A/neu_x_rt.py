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
        # print(rts)
        # print(per_inc)
        r, p = stats.spearmanr(rts, per_inc, nan_policy='omit')
        rs.append(r)
        # plt.scatter(per_inc, rts, alpha=.5)
        # plt.title(f'{sn}: {r=:.3f}')
        # plt.ylim(0, 3)
        # plt.show()
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

if __name__ == '__main__':
    analyze_rt_x_inc_neu()