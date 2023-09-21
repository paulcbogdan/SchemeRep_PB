import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as stats
from statsmodels.stats.multitest import multipletests

import utils
from ROIs import get_atlas
from utils import get_cache_RSA_fp, make_title_str, prune_to_only_hits

from connsearch.report.plots import plot_ROI_scores
from copy import deepcopy

def analyze_ROIs(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False):
    font = {'size': 14}
    matplotlib.rc('font', **font)
    cmap = plt.get_cmap('turbo')

    fp1 = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                           bilateral=bilateral, combine_regions=combine_regions,
                           vec_prod=vec_prod, org_by_region=org_by_region,
                           rxr=rxr)
    print(f'Attempted fp: {fp1}')
    with open(fp1, 'rb') as file:
        d = pickle.load(file)

    key = 'prod'

    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    colors = cmap(np.linspace(0, 1, len(atlas['ticks'])))
    idxs = list(np.arange(len(colors)))
    idxs_ = idxs.copy()
    idxs_[::3] = idxs[:9]
    idxs_[1::3] = idxs[9:18]
    idxs_[2::3] = idxs[18:]
    colors = colors[idxs_]
    colors[:, :3] /= 1.5
    # print(colors)
    # quit()
    region2color = dict(zip(atlas['tick_labels'], colors))
    # print(region2color)
    # quit()
    # print(tick_labels)
    # quit()
    Ms = []
    colors = []
    ps = []
    # if org_by_region:
    #     atlas['ROIs'] = atlas['ROI_regions'] = atlas['tick_labels']
    d_hit = deepcopy(d)
    d_miss = deepcopy(d)
    # prune_to_only_hits(d_hit, key1, misses=False)
    # prune_to_only_hits(d_miss, key1, misses=True)
    # prune_to_only_hits(d1, key1, misses=False)
    # for ROI in atlas['ROIs']:
    #     d_hit['z'][key1][ROI] = np.nanmean(d_hit['IRAFs_ROI'][key1][ROI], axis=1)
    #     d_miss['z'][key1][ROI] = np.nanmean(d_miss['IRAFs_ROI'][key1][ROI], axis=1)
    #     d1['z'][key1][ROI] = d_hit['z'][key1][ROI] - d_miss['z'][key1][ROI]
        # d1['z'][key1][ROI] = np.nanmean(d1['IRAFs_ROI'][key1][ROI], axis=1)

    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        # ROI_num, ROI_str = ROI.split(' ')
        # region = ROI_str.split('_')[0]

        # d['z'][key][ROI] = utils.regress_out_multi([d['z']['obj_abs'][ROI]],
        #                                             d['z'][key][ROI])

        color = region2color[region]
        colors.append(color)
        M0 = np.nanmean(d['z'][key][ROI])
        SD = np.nanstd(d['z'][key][ROI])
        N = len(d['z'][key][ROI])
        SE = SD / np.sqrt(N)
        t = M0 / SE
        p = stats.t.sf(np.abs(t), N-1)*2
        ps.append(p)
        Ms.append(t)
        print(f'{ROI}, {t=}')
    if max(Ms) > 1.6:
        plot_ROI_scores(Ms, atlas['coords'], show=True, fp_out='meh.png',
                    vmin=1.6)

    alpha = .10

    sigs, p_corr, alpha_sidak, alpha_bon = multipletests(ps, alpha=alpha,
                                                         method='fdr_bh')
    if np.min(p_corr) < alpha:
        p_corr_ = p_corr.copy()
        # print(p_corr)
        p_corr_[p_corr_ > alpha] = 0
        narrowest_cutoff = np.argmax(p_corr_)
        # print(narrowest_cutoff)
        # print(p_corr[narrowest_cutoff])
        t_cutoff = Ms[narrowest_cutoff]
        plt.plot([0, len(Ms)], [t_cutoff, t_cutoff], 'k--', linewidth=1,
                 zorder=-10)

    plt.scatter(np.array(atlas['ROI_nums']) - 1, Ms, color=colors, s=25)
    min_val = np.nanmin(Ms)*1.05
    for roi_num, M in zip(atlas['ROI_nums'], Ms):
        roi_num -= 1
        # if M > 0:
        plt.plot([roi_num, roi_num], [0, M], color=colors[roi_num], zorder=1)
        # else:
        # plt.plot([roi_num, roi_num], [0, M], color=colors[roi_num], zorder=1)

    max_val = np.nanmax(Ms)*1.05
    plt.ylim([0, max_val])
    # for i in range(len(tick_lows)):
    #     plt.plot([tick_lows[i], tick_lows[i]], [min_val, max_val], 'k--',
    #              zorder=-10)
    plt.plot([0, len(Ms)], [0, 0], color='k', zorder=-1, linewidth=1)
    plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=90, fontsize=12,
               )
    plt.ylabel('t-value', labelpad=5)
    title_str = make_title_str('', key, age, early, semantic, cin)
    plt.title(title_str, fontsize=11.5)

    # plt.gca().tick_params(axis='x', colors=colors)

    for i in range(len(atlas['ticks'])):
        # print(tick_labels[i])
        plt.gca().get_xticklabels()[i].set_color(
            region2color[atlas['tick_labels'][i]])
    plt.show()


if __name__ == '__main__':
    # Test connectivity within region between ROIs as nodes
    analyze_ROIs(early=True, semantic=True, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False)

    # analyze_ROIs(early=True, semantic=False, cin=None,
    #              bilateral=False, combine_regions=False, vec_prod=True,
    #              org_by_region=True, rxr=False)
