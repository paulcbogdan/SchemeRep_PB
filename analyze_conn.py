from pickle_wrap import pickle_wrap

import numpy as np

from basic_fCon import get_FC
from plotting import plot_connectivity
from utils import regress_out, make_title_str
from ROIs import get_BN_atlas, get_atlas

import matplotlib.pyplot as plt
import scipy.stats as stats

import pickle
import matplotlib
from statsmodels.stats.multitest import multipletests

def plot_test():
    age = 1
    cin = None
    semantic = False
    early_late = True
    fp_out = get_cache_RSA_fp(cin, age, semantic, early_late)
    with open(fp_out, 'rb') as file:
        d = pickle.load(file)

    d_IRAF_conn = d['IRAF_conn']
    mat0 = d_IRAF_conn['obj']
    mat1 = d_IRAF_conn['scn']
    mat2 = d_IRAF_conn['dif']
    mat3 = d_IRAF_conn['dif_']

    fig, axs = plt.subplots(1, 4, figsize=(24, 7))
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    plot_connectivity(mat0, ticks, tick_labels, tick_lows, title='obj',
                      ax=axs[0])
    plot_connectivity(mat1, ticks, tick_labels, tick_lows, title='scene',
                      ax=axs[1])
    plot_connectivity(mat2, ticks, tick_labels, tick_lows, title='dif',
                      ax=axs[2])
    plot_connectivity(mat3, ticks, tick_labels, tick_lows, title='dif reg',
                      ax=axs[3])
    plt.tight_layout()
    plt.show()

def get_age_str(age):
    return 'healthy' if age == 'healthy' else 'YA' if age == 1 else 'OA'

def get_cin_str(cin):
    return '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'

def get_cache_RSA_fp(cin, age, semantic, early, pre_str='', rxr=False,
                     combine_regions=False, bilateral=False, vec_prod=False,
                     org_by_region=False):
    age_str = 'healthy' if age == 'healthy' else \
        'YA' if age == 1 else 'OA'
    cin_str = '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'
    sem_str = '_sem' if semantic else ''
    el_str = '' if semantic else '_early' if early else '_late'
    rxr_str = '_rxr' if rxr else ''
    combine_str = '_comb' if combine_regions else ''
    bilat_str = '_bil' if bilateral else ''
    vecprod_str = '_vecprod' if vec_prod else ''
    by_region_str = '_byR' if org_by_region else ''
    fp_out = fr'cache/RSA/{pre_str}{age_str}{cin_str}{sem_str}{el_str}' \
             fr'{combine_str}{bilat_str}{vecprod_str}{by_region_str}{rxr_str}.pkl'
    print(f'{fp_out=}')
    return fp_out

def regress_out_normal_connectivity(mat, age, cin):
    mat = np.array(mat)
    age_str = get_age_str(age)
    cin_str = get_cin_str(cin)
    fp_FC = fr'cache/fCon_{age_str}{cin_str}.pkl'
    FC = pickle_wrap(fp_FC, lambda: get_FC(age, cin),
                     easy_override=False)
    n_ROIs = FC.shape[1]
    for i in range(n_ROIs):
        for j in range(n_ROIs):
            if i == j:
                continue
            print(mat[:, i, j])
            plt.scatter(FC[:, i, j], mat[:, i, j])
            plt.show()
            mat[:, i, j] = regress_out(FC[:, i, j], mat[:, i, j])
            print(mat[:, i, j])
            quit()
    return mat


def analyze_ROIs(age=2, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False):
    font = {'size': 14}
    matplotlib.rc('font', **font)
    cmap = plt.get_cmap('turbo')

    fp1 = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                           bilateral=bilateral, combine_regions=combine_regions,
                           vec_prod=vec_prod, org_by_region=org_by_region,
                           rxr=rxr)
    with open(fp1, 'rb') as file:
        d1 = pickle.load(file)
    key0 = 'z'
    key1 = 'obj'

    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    colors = cmap(np.linspace(0, 1, len(atlas['ticks'])))
    idxs = list(np.arange(len(colors)))
    idxs_ = idxs.copy()

    idxs_[::2] = idxs[:len(idxs_)//2 + 1]
    idxs_[1::2] = idxs[len(idxs_)//2 + 1:]
    # random.shuffle(idxs)
    colors = colors[idxs_]
    colors[:, :3] /= 1.3
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

    # prune_to_only_hits(d1, key1)
    # for ROI in atlas['ROIs']:
    #     d1['z'][key1][ROI] = np.nanmean(d1['IRAFs_ROI'][key1][ROI], axis=1)
    #     print(d1['z'][key1][ROI])

    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        # ROI_num, ROI_str = ROI.split(' ')
        # region = ROI_str.split('_')[0]
        color = region2color[region]
        colors.append(color)
        M0 = np.nanmean(d1[key0][key1][ROI])
        SD = np.nanstd(d1[key0][key1][ROI])
        N = len(d1[key0][key1][ROI])
        SE = SD / np.sqrt(N)
        t = M0 / SE
        p = stats.t.sf(np.abs(t), N-1)*2
        ps.append(p)
        Ms.append(t)
        print(f'{ROI}, {t=}')
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
        plt.plot([0, len(Ms)], [t_cutoff, t_cutoff], 'k--', linewidth=1)

    plt.scatter(np.array(atlas['ROI_nums']) - 1, Ms, color=colors, s=10)
    min_val = np.nanmin(Ms)
    max_val = np.nanmax(Ms)
    plt.ylim([min_val*1.02, max_val*1.02])
    # for i in range(len(tick_lows)):
    #     plt.plot([tick_lows[i], tick_lows[i]], [min_val, max_val], 'k--',
    #              zorder=-10)
    plt.plot([0, len(Ms)], [0, 0], color='k', zorder=-1, linewidth=1)
    plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=90, fontsize=10)
    plt.ylabel('t-value')
    title_str = make_title_str('', key1, age, early,
                               semantic, cin)
    plt.title(title_str, fontsize=11.5)

    # plt.gca().tick_params(axis='x', colors=colors)

    for i in range(len(atlas['ticks'])):
        # print(tick_labels[i])
        plt.gca().get_xticklabels()[i].set_color(
            region2color[atlas['tick_labels'][i]])
    plt.show()
    quit()

def CIN_compare(age=1, early=False, semantic=False):
    fp1 = get_cache_RSA_fp(cin=1, age=age, semantic=semantic, early=early)
    fp2 = get_cache_RSA_fp(cin=2, age=age, semantic=semantic, early=early)
    with open(fp1, 'rb') as file:
        d1 = pickle.load(file)
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)

    # mat = regress_out_normal_connectivity(d1['IRAF_conn']['obj'], age, 1)

    key0 = 'z'
    key1 = 'dif_'
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_BN_atlas()
    # plot_connectivity(mat, ticks, tick_labels, tick_lows, title='obj')
    # quit()

    # for key2 in ['obj', 'scn', 'dif', 'dif_']:
    for ROI in atlas['ROIs']:
        M0 = np.mean(d1[key0][key1][ROI])
        M1 = np.mean(d2[key0][key1][ROI])
        l0 = np.array(d1[key0][key1][ROI])
        l1 = np.array(d2[key0][key1][ROI])
        t, p = stats.ttest_ind(l0, l1)
        print(f'{ROI}: {M0:.3f} {t=:.3f} ')

def replace_w_nan_if_needed(vals):
    clean = []
    for x in vals:
        if x.shape == vals[0].shape:
            clean.append(x)
        else:
            shape_nan = (vals[0].shape[0] - x.shape[0], vals[0].shape[1])
            fill_nan = np.full(shape_nan, np.nan)
            x = np.concatenate([x, fill_nan])
            clean.append(x)
            # print(f'{x.shape=}')
            # quit()
            # clean.append(np.full(vals[0].shape, np.nan))
    return np.array(clean)

def prune_to_only_hits(d, key, misses=False):
    d['bhv']['hit_bool'] = np.nan_to_num(d['bhv']['hit_bool'], True).astype(bool)
    for ROI in d['IRAFs_ROI'][key]:
        if d['IRAFs_ROI'][key][ROI].shape[0] != 33:
            mask = np.full(d['IRAFs_ROI'][key][ROI].shape, False)
            print('BAH')
        else:
            if misses:
                mask = ~d['bhv']['hit_bool']
            else:
                mask = d['bhv']['hit_bool']
        d['IRAFs_ROI'][key][ROI][~mask] = np.nan
        d['activity'][ROI] = np.array(d['activity'][ROI])
        d['activity'][ROI][~mask] = np.nan

def test_IRAF_x_activity(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False):
    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                           bilateral=bilateral, combine_regions=combine_regions,
                           vec_prod=vec_prod, org_by_region=org_by_region,
                           rxr=rxr)
    with open(fp, 'rb') as file:
        d = pickle.load(file)



    # print(np.array(d['bhv']['hit_bool']).shape)
    # print(d['IRAFs_ROI']['dif']['SFG_L'].shape)
    # test = np.full((33, 114), True)
    # d['bhv']['hit_bool'] = np.nan_to_num(d['bhv']['hit_bool'], True).astype(bool)
    # print(d['bhv']['hit_bool'])
    # test = d['bhv']['hit_bool']
    # d['IRAFs_ROI']['dif']['SFG_L'][~test] = np.nan
    # print( d['IRAFs_ROI']['dif']['SFG_L'])
    # quit()

    #
    #
    # quit()

    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    key = 'dif'
    prune_to_only_hits(d, key)



    IRAFs = [np.array(d['IRAFs_ROI'][key][roi0]) for roi0 in atlas['ROIs']]
    # IRAFs = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]

    IRAFs = replace_w_nan_if_needed(IRAFs)
    # print(f'{IRAFs.shape=}')
    # quit()
    # IRAFs = np.array(list(filter(lambda x: x.shape[0] == 33, IRAFs)))
    # activity = [np.array(d['IRAFs_ROI']['scn'][roi1]) for roi1 in atlas['ROIs']]
    activity = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]
    activity = replace_w_nan_if_needed(activity)
    # print(activity.shape)
    # print(IRAFs.shape)
    # quit()
    # activity = np.array(list(filter(lambda x: x.shape[0] == 33, activity)))
    r_Ms, r_SDs, t = bulk_correlate(IRAFs, activity, nans=True)

    title = make_title_str('Activity x IRAF', key, age, early, semantic, cin)
    # title = 'YA. Activity x Difference-IRAF, 1st-layer DNN.'
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=title, no_avg=True,
                      cbar_label='t-value')

def bulk_correlate(vals0, vals1, nans=False):

    vals0 = np.expand_dims(vals0, axis=1)
    vals1 = np.expand_dims(vals1, axis=0)

    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    vals0_M = m(vals0, axis=-1)

    vals0_SD = s(vals0, axis=-1)
    vals0_ = (vals0 - vals0_M[:, :, :, None]) / vals0_SD[:, :, :, None]
    vals1_M = m(vals1, axis=-1)
    vals1_SD = s(vals1, axis=-1)
    vals1_ = (vals1 - vals1_M[:, :, :, None]) / vals1_SD[:, :, :, None]

    rs = vals0_ * vals1_
    rs = m(rs, axis=-1)
    r_Ms = np.nanmean(rs, axis=-1)
    r_Ms[np.diag_indices_from(r_Ms)] = np.nan
    r_SDs = np.nanstd(rs, axis=-1)
    r_SDs[np.diag_indices_from(r_Ms)] = np.nan
    t = r_Ms / r_SDs * np.sqrt(rs.shape[-1]) # fix to account for different # nans per edge
    print(f'{t.shape=}')
    return r_Ms, r_SDs, t
    # print(rs.shape)
    # quit()
    # pass

def test_triple_z(age=1, early=True, semantic=False,
                         combine_regions=True, bilateral=False):
    fp = get_cache_RSA_fp(cin=None, age=age, semantic=semantic, early=early,
                          combine_regions=combine_regions, bilateral=bilateral)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)
    data = d['triple_z']['obj']
    data = np.array(data)
    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    t = M / SD * np.sqrt(len(data))
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')


def test_rxr(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=True, rxr=True):
    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                           bilateral=bilateral, combine_regions=combine_regions,
                           vec_prod=vec_prod, org_by_region=org_by_region,
                           rxr=rxr)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    data = d['triple_z']['obj']
    data = np.array(data)
    # for val in data[:, 41, 45]:
    #     print(val > 0)

    # print(data[:, 41, 45])
    # quit()

    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    # t = M
    t = M / SD * np.sqrt(len(data))

    # t[41, 45] = 100
    # print(t[41, 43])
    # quit()
    # t = np.mean(data > 0, axis=0)
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')


    # print(np.array(data).shape)
    quit()


if __name__ == '__main__':
    # test_triple_z()
    # CIN_compare()
    analyze_ROIs()
    # test_rxr()
    # test_IRAF_x_activity()