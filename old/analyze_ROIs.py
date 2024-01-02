import pickle

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests

from atlas_utils import get_atlas
from old.plot_gen import my_plot_surf
from utils import get_RSA_fn, make_title_str

# from connsearch.report.plots import plot_ROI_scores
import scipy.stats as stats

def do_lmer(d, key, ROI, hits_only=True):

    IRAFs = d['IRAFs_ROI'][key][ROI]
    # print(IRAFs.shape)
    n_trials = IRAFs.shape[-1]
    sns = np.repeat(np.array(d['sns'])[:, None], n_trials, axis=1)
    incs = d['bhv']['inc']

    hit_hit = d['bhv']['hit_hit']
    # run = d['bhv']['run']
    IRAFs = np.reshape(IRAFs, -1)
    sns = np.reshape(sns, -1)
    incs = np.reshape(incs, -1)
    hit_hit = np.reshape(hit_hit, -1)
    # run = np.reshape(run, -1)
    incs = list(map(lambda x: 'i' if x == 1 else 'n' if x == 2 else 'c', incs))
    # sns = [sn for sn in sns if sn != '102']
    d = {'IRAF': IRAFs, 'sn': sns, 'inc': incs, 'hit_hit': hit_hit}
    # for key, l in d.items():
    #     print(f'{key}, {len(l)}')

    df = pd.DataFrame(d)

    # bad_sns = {'119', '115'}
    # df = df[df['sn'].isin(bad_sns) == False]
    # df = df[~((df['sn'] == '132') & (df['run'] == 3))]

    if hits_only:
        df = df[df['hit_hit'] > 0]  # doing "> 0" causes NaN -> False

    # Load = 3.3 s, first run = 1.7 s, rest runs = 0.32 s
    from pymer4.models import Lmer
    formula = f'IRAF ~ 1 + (1 | sn)'
    df.dropna(subset=['IRAF', 'sn'], inplace=True)
    # df = df[['IRAF', 'sn']]
    # print(df)
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    summary = model.coefs
    t = summary['T-stat'].loc['(Intercept)']
    p = summary['P-val'].loc['(Intercept)']
    return t, p

def do_ttest(d, key, ROI, wilcox=False):
    if wilcox:
        res = stats.wilcoxon(d['z'][key][ROI], alternative='greater')
        N = np.sum(~np.isnan(d['z'][key][ROI]))
        p = res.pvalue
        t = stats.t.ppf(1 - p, N - 1)
        if t < -4:
            t = -4
        M0 = np.nanmean(d['z'][key][ROI] > 0.)
        return M0, t, p, 0
    else:
        M0 = np.nanmean(d['z'][key][ROI])
        SD = np.nanstd(d['z'][key][ROI])

        # M0 = np.nanmean(d['rxr'][key][ROI])
        # SD = np.nanstd(d['rxr'][key][ROI])

        N = np.sum(~np.isnan(d['z'][key][ROI]))
        SE = SD / np.sqrt(N)
        t = M0 / SE
        p = stats.t.sf(np.abs(t), N - 1) * 2
        return M0, t, p, N

def setup_colors(atlas):
    cmap = plt.get_cmap('turbo')
    cmap_scrambled = cmap(np.linspace(0, 1, len(atlas['ticks'])))
    idxs = list(np.arange(len(cmap_scrambled)))
    idxs_ = idxs.copy()
    idxs_[::3] = idxs[:9]
    idxs_[1::3] = idxs[9:18]
    idxs_[2::3] = idxs[18:]
    cmap_scrambled = cmap_scrambled[idxs_]
    cmap_scrambled[:, :3] /= 1.5
    region2color = dict(zip(atlas['tick_labels'], cmap_scrambled))
    return region2color

def do_pb_ROI_plot(ps, ts, colors, atlas, title, region2color):
    font = {'size': 14}
    matplotlib.rc('font', **font)
    alpha = .10
    sigs, p_corr, alpha_sidak, alpha_bon = multipletests(ps, alpha=alpha,
                                                         method='fdr_bh')
    ts = [min(t, 5) for t in ts]
    if np.min(p_corr) < alpha:
        p_corr_ = p_corr.copy()
        p_corr_[p_corr_ > alpha] = 0
        narrowest_cutoff = np.argmax(p_corr_)
        t_cutoff = ts[narrowest_cutoff]
        plt.plot([0, len(ts)], [t_cutoff, t_cutoff], 'k--', linewidth=1,
                 zorder=-10)

    plt.scatter(np.array(atlas['ROI_nums']) - 1, ts, color=colors, s=25)
    for roi_num, M in zip(atlas['ROI_nums'], ts):
        roi_num -= 1
        plt.plot([roi_num, roi_num], [0, M], color=colors[roi_num], zorder=1)

    max_val = np.nanmax(ts)*1.05
    min_val = min(0, np.nanmin(ts)*1.05)
    plt.ylim([min_val, max_val])

    plt.plot([min_val, len(ts)], [0, 0], color='k', zorder=-1, linewidth=1)
    plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=90, fontsize=12)
    plt.ylabel('t-value', labelpad=5)
    plt.title(title, fontsize=11.5)

    for i in range(len(atlas['ticks'])):
        plt.gca().get_xticklabels()[i].set_color(
            region2color[atlas['tick_labels'][i]])
    plt.show()

def prune_bad_sns(d, drop_ret=False):
    # d['sns'] = d['sns'][1:] '138',
    if drop_ret:
        bad_sns = {'116', '125', '133', '213', '215'}
    else:
        bad_sns = {'104', '109', '115', '119'}
    sns_bool = np.array([sn not in bad_sns for sn in d['sns']])
    for key, d_sub in d['IRAFs_ROI'].items():
        d_sub_z = d['z'][key]
        for ROI, ar in d_sub.items():
            d_sub[ROI] = ar[sns_bool]
            d_sub_z[ROI] = d_sub_z[ROI][sns_bool]
    for roi, d_sub in d['activity'].items():
        d['activity'][roi] = d_sub[sns_bool]
    for col in d['bhv']:
        d['bhv'][col] = d['bhv'][col][sns_bool]
    d['sns'] = d['sns'][sns_bool]

    return d

def analyze_ROIs(age=2, early=True, semantic=False, inc=None,
                 bilateral=False, combine_regions=True,
                 vec_prod=False, PCA_obj=True,
                 org_by_region=False, rxr=False,
                 run_lmer=False,
                 DNN_layer=2, fp_fMRI_col='obj4_fMRI',
                 verbose=True, fp=None, require_all_sns=True,
                 req_all_N=False, key='dif_abs_'):
    if fp is None:
        fn = get_RSA_fn(inc=inc, age=age, semantic=semantic,
                        DNN_layer=DNN_layer,
                         fp_fMRI_col=fp_fMRI_col, PCA_obj=PCA_obj,
                         bilateral=bilateral, combine_regions=combine_regions,
                         vec_prod=vec_prod, org_by_region=org_by_region,
                         )
        fp = fr'cache/RSA/{fn}.pkl'

    with open(fp, 'rb') as file:
        d = pickle.load(file)
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      combine_bilateral=bilateral or org_by_region)
    region2color = setup_colors(atlas)
    colors = []
    ts = []
    ps = []
    d['sns'] = list(d['sns'])
    # d['sns'].remove('138')
    num_sns = len(d['sns'])
    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        if run_lmer:
            t, p = do_lmer(d, key, ROI, hits_only=True)
            M0 = 0
            N = 0
        else:
            M0, t, p, N = do_ttest(d, key, ROI, wilcox=False)
        if req_all_N and N != num_sns:
            continue
        ts.append(t)
        ps.append(p)
        color = region2color[region]
        colors.append(color)
        if verbose: print(f'{ROI}, {M0=:.3f}, {t=:.3f}, {p=:.3f}, {N=}')
    if verbose:
        title_short = make_title_str('', key, age, early, semantic, short=True)
        my_plot_surf(np.array(ts), atlas, title_short)
        title = make_title_str('', key, age, early, semantic, inc)
        do_pb_ROI_plot(ps, ts, colors, atlas, title, region2color)
    return ts, num_sns

if __name__ == '__main__':
    # Test connectivity within region between ROIs as nodes
    analyze_ROIs()

    # analyze_ROIs(early=True, semantic=False, cin=None,
    #              bilateral=False, combine_regions=False, vec_prod=True,
    #              org_by_region=True, rxr=False)
