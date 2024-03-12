import os

from tqdm import tqdm

from HCP import load_HCP_act
from analyze_rs import load_act_conn, load_resting_data
from atlas_utils import get_atlas
from old.analyze_ROIs import setup_colors
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from utils import pickle_wrap, stdize
from vendor_lmers import get_dfs_conn_trials
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from time import time
from functools import partial
from pathlib import Path
from pprint import pprint
from colorama import Fore


def shuffle_rows(sn_roi_act):
    new_ar = np.full(sn_roi_act.shape, np.nan)
    sn_roi_act_sq = np.nanmean(sn_roi_act, axis=1)
    num_ea = sn_roi_act.shape[-1] // 3
    for j in range(sn_roi_act_sq.shape[0]):
        cnt_l = [0, 1, 2] * num_ea
        random.shuffle(cnt_l)
        for i in range(sn_roi_act_sq.shape[-1]):
            cnt = cnt_l[i]
            new_ar[j, cnt, :, i] = sn_roi_act_sq[j, :, i]
    return new_ar

def act2conn(sn_roi_act, flatten=True, conn_euc=False):
    # print(f'ac2conn: {sn_roi_act.shape}')
    t_st = time()
    sn_roi_act0 = sn_roi_act[..., :, None, :]
    sn_roi_act1 = sn_roi_act[..., None, :, :]
    # print('a')
    if conn_euc:
        sn_roi_conn = np.abs(sn_roi_act0 - sn_roi_act1)
    else:
        sn_roi_conn = sn_roi_act0 * sn_roi_act1
    if flatten:
        # print('b')
        trils = np.tril_indices(sn_roi_act.shape[-2], -1)
        sn_roi_conn = sn_roi_conn[..., trils[0], trils[1], :]
        # print(f'{sn_roi_conn.shape} | time needed: {time() - t_st:.2f} s')
    return sn_roi_conn

def plot_ranks(idx2rank, atlas, n_roi=246, do_conn=True, split=.5):
    from nichord import convert_matrix, get_idx_to_label
    from nichord.combine import plot_and_combine
    if do_conn:
        print(f'{idx2rank=}')
        trils = np.tril_indices(n_roi, -1)
        ar2rank = np.full((n_roi, n_roi), np.nan)
        ar2rank[trils] = idx2rank
        plot_connectivity(ar2rank, atlas=atlas, cbar_label='rank')
        return
        dir_out = fr'result_pics/nichord_rank'
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        edges, weights = convert_matrix(ar2rank)

        edges_neg = [edge for edge, weight in zip(edges, weights) if weight < 0]
        weights_neg = [weight for weight in weights if weight < 0]

        edges_pos = [edge for edge, weight in zip(edges, weights) if weight > 0]
        weights_pos = [weight for weight in weights if weight > 0]

        coords = atlas['coords']
        idx_to_label = pickle_wrap(get_idx_to_label,
                                   kwargs={'coords': coords})

        fn_out = f'pos_{split}.png'
        plot_and_combine(dir_out, fn_out, idx_to_label, edges_pos,
                         edge_weights=weights_pos, coords=coords)
        fn_out = f'neg_{split}.png'
        plot_and_combine(dir_out, fn_out, idx_to_label, edges_neg,
                         edge_weights=weights_neg, coords=coords)
        quit()
    else:
        region2color = setup_colors(atlas)
        colors = []
        for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
            color = region2color[region]
            colors.append(color)
        for idx, rank in enumerate(idx2rank):
            plt.plot([idx, idx], [0, rank],
                     color=colors[idx], zorder=1)

        plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=90,
                   fontsize=12)
        plt.ylabel('Ranking', labelpad=5)
        plt.show()

def load_task_rs_data(combine_regions, fp='vis7_fMRI', only_cortical=False,
                      do_hit_hit=True, HCP=False, light=False, medium=False,
                      trad=False):
    # sn_roi_act, _, df_sns_l, sns  = \
        # pickle_wrap(get_dfs_conn_trials,
        #             kwargs={'fp': fp, 'single': False,
        #                     'squeeze': False,
        #                     'combine_regions': combine_regions},
        #             easy_override=False, RAM_cache=True)
        #
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions
              }
    if do_hit_hit:
        kwargs['key'] = 'hit_hit'
        kwargs['key_vals'] = (False, 0.5, True)

    _, _, _, sn_roi_act, df_sns_l = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=0, cache_dir='cache',
                    RAM_cache=True)

    sns = [df['sn'].iloc[0] for df in df_sns_l]
    good_i = [i for i, sn in enumerate(sns) if sn != '133'] # bad rs
    sn_roi_act = sn_roi_act[good_i, :, :]
    sns = [sn for sn in sns if sn != '133'] # bad rs



    n_roi = sn_roi_act.shape[-2]
    atlas = get_atlas(combine_regions=combine_regions)
    if HCP:
        hcp_kwargs = {'N': 30, 'combine_regions': combine_regions}
        sn_roi_rs, sns_ = pickle_wrap(load_HCP_act, kwargs=hcp_kwargs,
                                      easy_override=False, verbose=0,
                                      RAM_cache=True)
        sns = sns_
    else:
        # if light:
        #     f = partial(load_resting_data, combine_regions=combine_regions,
        #                 sns_key=fp, light=True)
        # else:
        f = partial(load_resting_data, combine_regions=combine_regions,
                    sns_key=fp, light=light, medium=medium, trad=trad)
        sn_roi_rs, sns_, _ = load_act_conn(False, f=f,
                                                    easy_override=False,
                                           RAM_cache=True)
        assert sn_roi_act.shape[0] == sn_roi_rs.shape[0], ('Unequal # subjects '
                                                           f'{sn_roi_act.shape=},'
                                                           f'{sn_roi_rs.shape=} '
                                                           f'| {sns=}\n{sns_=}')
        assert sn_roi_act.shape[-2] == sn_roi_rs.shape[-2], (f'Unequal # ROIs: '
                                                             f'{sn_roi_act.shape=},'
                                                             f' {sn_roi_rs.shape=}')
        assert sns == sns_, print(f'{sns=}\n{sns_=}')
        # quit()

        bad_sns = {'116', '117', '110', '130', '232'}
        # bad_sns = {'116'}
        bad_i = [i for i, sn in enumerate(sns) if sn in bad_sns]
        sns = [sn for sn in sns if sn not in bad_sns]
        sn_roi_act = np.delete(sn_roi_act, bad_i, axis=0)
        sn_roi_rs = np.delete(sn_roi_rs, bad_i, axis=0)

    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        sn_roi_act[..., bad_j, :] = np.nan
        sn_roi_rs[..., bad_j, :] = np.nan
    # print(f'{sn_roi_act.shape=}')
    # print(f'{sn_roi_rs.shape=}')
    # quit()

    return sn_roi_act, sn_roi_rs, atlas, n_roi, sns

def link_activity(do_generic=True, do_conn=True, verbose=0,
                  shuffle=False, combine_regions=True,
                  split=.3, do_plot=False, fp_task='obj7_fMRI',
                  do_hit_hit=False, shuffle_ss=False, HCP=False,
                  light=False, medium=False, conn_euc=True, trad=True):
    sn_roi_act, sn_roi_rs, atlas, n_roi, sns = \
        load_task_rs_data(combine_regions, only_cortical=True,
                          fp=fp_task, do_hit_hit=do_hit_hit,
                          HCP=HCP, light=light, medium=medium,
                          trad=trad)

    if do_conn:
        sn_roi_act = stdize(sn_roi_act, nans=True, axis=-1)
        sn_roi_act = act2conn(sn_roi_act, conn_euc=conn_euc)
        sn_roi_rs = stdize(sn_roi_rs, nans=True, axis=-1)
        sn_roi_rs = act2conn(sn_roi_rs, conn_euc=conn_euc)

    if shuffle and not shuffle_ss:
        sn_roi_act = shuffle_rows(sn_roi_act)

    sn_inc_roi_Ms = np.nanmean(sn_roi_act, axis=-1)
    sn_roi_difs = sn_inc_roi_Ms[:, 2, :] - sn_inc_roi_Ms[:, 0, :]

    generic_difs = np.nanmean(sn_roi_difs, axis=0)
    generic_Ns = np.sum(~np.isnan(sn_roi_difs), axis=0)
    generic_SDs = np.nanstd(sn_roi_difs, axis=0)
    generic_SEs = generic_SDs / np.sqrt(generic_Ns) # TODO: toggle vs. just dif
    generic_ts = generic_difs / generic_SEs

    n_nans = np.sum(np.isnan(generic_ts))
    n_elements = len(generic_ts)
    nan_cutoff = n_elements - n_nans
    gen_rank2idx = generic_ts.argsort()

    gen_idx2rank = np.argsort(gen_rank2idx).astype(float)

    gen_idx2rank[gen_idx2rank >= nan_cutoff] = np.nan

    low_cutoff = int(nan_cutoff * split)
    high_cutoff = int(nan_cutoff * (1 - split))
    # print(f'{low_cutoff=}')
    # print(f'{high_cutoff=}')

    if do_plot:
        gen_idx2rank[(gen_idx2rank > low_cutoff) &
                     (gen_idx2rank < high_cutoff)] = np.nan
        temp = generic_ts

        temp[np.isnan(gen_idx2rank)] = np.nan

        plot_ranks(temp, atlas, n_roi=n_roi, do_conn=do_conn, split=split)

        idx_to_cnt_con = np.zeros(len(temp))
        idx_to_cnt_inc = np.zeros(len(temp))
        for sn_i in range(sn_roi_difs.shape[0]): # , position=0, leave=False)

            sn_effs = sn_roi_difs[sn_i, :]
            n_nans_sn = np.sum(np.isnan(sn_effs))
            rank2idx = sn_effs.argsort()

            inc_rois = rank2idx[:low_cutoff]
            con_rois = rank2idx[high_cutoff:-n_nans_sn]
            for x in inc_rois:
                idx_to_cnt_inc[x] += 1
            for x in con_rois:
                idx_to_cnt_con[x] += 1

        idx_to_cnt_con[idx_to_cnt_con == 0] = np.nan
        idx_to_cnt_inc[idx_to_cnt_inc == 0] = np.nan
        plot_ranks(idx_to_cnt_con,
                   atlas, n_roi=n_roi, do_conn=do_conn, split=split)
        plot_ranks(idx_to_cnt_inc,
                   atlas, n_roi=n_roi, do_conn=do_conn, split=split)


    corrs = []
    if verbose: print('Onto looping')
    i2con_rois = []
    i2inc_rois = []
    for sn_i in range(sn_roi_difs.shape[0]): # , position=0, leave=False)
        if do_generic:
            rank2idx = gen_rank2idx
            n_nans_sn = n_nans
            nan_idxs = np.argwhere(np.isnan(generic_ts))[:, 0]
        else:
            sn_effs = sn_roi_difs[sn_i, :]
            n_nans_sn = np.sum(np.isnan(sn_effs))
            rank2idx = sn_effs.argsort()
            nan_idxs = np.argwhere(np.isnan(sn_effs))[:, 0]

        inc_rois = rank2idx[:low_cutoff]
        con_rois = rank2idx[high_cutoff:-n_nans_sn]

        assert np.all([(j not in inc_rois) for j in nan_idxs]), \
            f'Bad in inc_rois: {inc_rois=}'
        assert np.all([(j not in con_rois) for j in nan_idxs]), \
            f'Bad in con_rois: {inc_rois=}'

        i2con_rois.append(con_rois)
        i2inc_rois.append(inc_rois)
        # print(f'{sn_i} | {list(inc_rois)[:5]=}')


    i2con_rois = np.array(i2con_rois)
    i2inc_rois = np.array(i2inc_rois)

    if shuffle_ss:
        permuter = np.random.permutation(i2con_rois.shape[0])
        i2con_rois = i2con_rois[permuter]
        i2inc_rois = i2inc_rois[permuter]
        # for i in range(i2con_rois.shape[0]):
        #     print(f'{i}: {list(i2con_rois[i])[:5]}')
        # np.random.shuffle(i2con_rois)
        # # for i in range(i2con_rois.shape[0]):
        # #     print(f'-{i}: {list(i2con_rois[i])[:5]}')
        # # quit()
        # np.random.shuffle(i2inc_rois)

    for sn_i in range(sn_roi_rs.shape[0]):
        inc_rois, con_rois = i2inc_rois[sn_i], i2con_rois[sn_i]
        # print(f'{sn_i} | {list(inc_rois)[:5]=}')
        # quit()
        # print(sn_roi_rs[sn_i, ])
        # quit()
        rs_inc = sn_roi_rs[sn_i, inc_rois, :]
        rs_inc = np.nanmean(rs_inc, axis=0) # TODO: toggle to nan and exclude?
        # print(rs_inc)
        # quit()

        rs_con = sn_roi_rs[sn_i, con_rois, :]
        rs_con = np.nanmean(rs_con, axis=0)
        # print(f'{con_rois=}')
        # print(f'{rs_con=}')
        # print(f'{con_rois=}')
        # quit()

        n_con_nans = np.sum(np.isnan(rs_con))
        n_inc_nans = np.sum(np.isnan(rs_inc))
        if (n_con_nans or n_inc_nans) and not HCP:
            assert n_con_nans == n_inc_nans == 206, \
                f'{n_con_nans=} {n_inc_nans=}'
            print(f'skip: {sns[sn_i]}')
        #
        # print(f'{rs_inc=}')
        # print(f'{np.nanmax(rs_inc)=}')
        # print(f'{np.nanmin(rs_inc)=}')
        # print(f'{np.nanmax(rs_con)=}')
        # print(f'{np.nanmin(rs_con)=}')


        corr, _ = stats.pearsonr(rs_inc, rs_con)
        corr = np.arctanh(corr)
        corrs.append(corr)

    corrs = np.array(corrs)
    M_corr = np.mean(corrs)
    generic_str = 'Generic' if do_generic else 'Subject-specific'
    if shuffle_ss:
        print(f'{generic_str} ({split}) | Shuffle SS: {M_corr=:.3f}')
    elif shuffle:
        print(f'{generic_str} ({split}) | Shuffle: {M_corr=:.3f}')
    else:
        print(f'{generic_str} ({split}): '
              f'{Fore.LIGHTYELLOW_EX}{M_corr=:.3f}{Fore.RESET}')
    return M_corr

def permutation_test_(n_sim=100, **kwargs):
    M_corrs = []
    for _ in range(n_sim):
        M_corr = link_activity(**kwargs, do_plot=False, shuffle=True)
        M_corrs.append(M_corr)
    return M_corrs


def permutation_test(**kwargs):
    from connsearch import print_list_stats
    if kwargs['do_generic']:
        M_corrs = pickle_wrap(permutation_test_, kwargs=kwargs,
                              RAM_cache=False, verbose=-1,
                              easy_override=False)
        M_corrs = np.array(M_corrs)
        print('--*--')
        # pprint(kwargs)
        # print('--*--')
        print_list_stats(M_corrs)
        print('--*--')

    if not kwargs['do_generic']:
        kwargs['shuffle_ss'] = True
        M_corrs = pickle_wrap(permutation_test_, kwargs=kwargs,
                              RAM_cache=False, verbose=-1,
                              easy_override=False)
        print_list_stats(M_corrs)
        print('--*--')


def plot_by_split(do_conn=True, do_generic=True, combine_regions=True,
                  do_hit_hit=False, fp_task='obj7_fMRI', HCP=False,
                  light=False, medium=False, conn_euc=False, trad=True):
    kwargs = {
              'do_conn': do_conn,
              'combine_regions': combine_regions,
              'do_generic': do_generic,
              'do_hit_hit': do_hit_hit,
              'fp_task': fp_task,
              'HCP': HCP,
              'light': light,
              'medium': medium,
              'conn_euc': conn_euc,
              'trad': trad
              }
    assert do_conn or not combine_regions
    for split in [.3, .5, .4, .2, .1, .05]: # . [0.3]:# .5, .4,   .2, .1, .05 .3,
        kwargs['split'] = split
        print('--*--')
        pprint(kwargs)
        print('--*--')

        M_corr = link_activity(**kwargs, do_plot=False, shuffle=False)
        # quit()
        #                       do_plot=False, shuffle=False)
        # M_corrs = link_activity(do_generic=not do_generic, do_conn=do_conn,
        #                         split=split, combine_regions=combine_regions,
        #                         do_plot=False, shuffle=False)
        # continue
        permutation_test(n_sim=100, **kwargs)

if __name__ == '__main__':
    plot_by_split()

