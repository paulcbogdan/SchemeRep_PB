import pickle

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import utils
from atlas_utils import get_atlas
from plot_gen import my_plot_surf
from utils import get_cache_RSA_fp, make_title_str, prune_to_only_hits

from connsearch.report.plots import plot_ROI_scores
from copy import deepcopy
from time import time
import scipy.stats as stats

def do_lmer(d, key, ROI, hits_only=True):

    IRAFs = d['IRAFs_ROI'][key][ROI]
    n_trials = IRAFs.shape[-1]
    sns = np.repeat(np.array(d['sns'])[:, None], n_trials, axis=1)
    incs = d['bhv']['inc']
    # print(list(d['bhv']))
    # for key in d['bhv']:
    #     print(key, ':', pd.isna(d['bhv'][key]).sum())
    # quit()
    hit_hit = d['bhv']['hit_hit']
    run = d['bhv']['run']
    IRAFs = np.reshape(IRAFs, -1)
    sns = np.reshape(sns, -1)
    incs = np.reshape(incs, -1)
    hit_hit = np.reshape(hit_hit, -1)
    run = np.reshape(run, -1)
    incs = map(lambda x: 'i' if x == 1 else 'n' if x == 2 else 'c', incs)
    df = pd.DataFrame({'IRAF': IRAFs, 'sn': sns, 'inc': incs,
                       'hit_hit': hit_hit, 'run': run})

    # bad_sns = {'119', '115'}
    # df = df[df['sn'].isin(bad_sns) == False]
    # df = df[~((df['sn'] == '132') & (df['run'] == 3))]

    if hits_only:
        df = df[df['hit_hit'] > 0]  # doing "> 0" causes NaN -> False

    # Load = 3.3 s, first run = 1.7 s, rest runs = 0.32 s
    from pymer4.models import Lmer
    formula = f'IRAF ~ 1 + (1 | sn)'
    df.dropna(subset=['IRAF'], inplace=True)
    model = Lmer(formula, data=df)
    model.fit(REML=False, verbose=False, summary=False)
    summary = model.coefs
    # print(summary.round(3))
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
        return t, p
    else:
        M0 = np.nanmean(d['z'][key][ROI])
        SD = np.nanstd(d['z'][key][ROI])
        N = np.sum(~np.isnan(d['z'][key][ROI]))
        SE = SD / np.sqrt(N)
        t = M0 / SE
        p = stats.t.sf(np.abs(t), N - 1) * 2
        return t, p

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
    alpha = .10
    sigs, p_corr, alpha_sidak, alpha_bon = multipletests(ps, alpha=alpha,
                                                         method='fdr_bh')
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


def analyze_ROIs(age=1, early=True, semantic=False, inc=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False, run_lmer=True):
    font = {'size': 14}
    matplotlib.rc('font', **font)

    fp1 = get_cache_RSA_fp(inc=inc, age=age, semantic=semantic, DNN_layer=2,
                           fp_fMRI_col='obj_fMRI',
                           bilateral=bilateral, combine_regions=combine_regions,
                           vec_prod=vec_prod, org_by_region=org_by_region,
                           rxr=rxr)
    # fp1 = r'C:\PycharmProjects_C\SchemeRep\cache\RSA\good_data_backups\YA_early.pkl'

    with open(fp1, 'rb') as file:
        d = pickle.load(file)
    # print(d['sns'])
    # print(d['z']['obj'].keys())
    # print(d['z']['obj']['SFG_L'].shape)
    # quit()
    key = 'dif_abs'
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    region2color = setup_colors(atlas)
    colors = []
    ts = []
    ps = []
    for ROI, region in tqdm(zip(atlas['ROIs'], atlas['ROI_regions'])):
        # try:
        if run_lmer:
            t, p = do_lmer(d, key, ROI, hits_only=True)
        else:
            t, p = do_ttest(d, key, ROI)
        # except KeyError:
        #     continue
        ts.append(t)
        ps.append(p)
        color = region2color[region]
        colors.append(color)
        print(f'{ROI}, {t=:.3f}, {p=:.3f}')
    title_short = make_title_str('', key, age, early, semantic, short=True)
    my_plot_surf(np.array(ts), atlas, title_short)
    title = make_title_str('', key, age, early, semantic, inc)
    do_pb_ROI_plot(ps, ts, colors, atlas, title, region2color)


if __name__ == '__main__':
    # Test connectivity within region between ROIs as nodes
    analyze_ROIs()

    # analyze_ROIs(early=True, semantic=False, cin=None,
    #              bilateral=False, combine_regions=False, vec_prod=True,
    #              org_by_region=True, rxr=False)
