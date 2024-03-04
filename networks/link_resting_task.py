import os

from tqdm import tqdm

from analyze_rs import load_act_conn, load_resting_data
from atlas_utils import get_atlas
from old.analyze_ROIs import setup_colors
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


def shuffle_rows(sn_roi_act):
    new_ar = np.full(sn_roi_act.shape, np.nan)
    sn_roi_act_sq = np.nanmean(sn_roi_act, axis=1)
    num_ea = sn_roi_act.shape[-1] // 3

    for j in range(sn_roi_act_sq.shape[0]):

        cnt_l = [0, 1, 2] * num_ea
        random.shuffle(cnt_l)
        # cnt_ar = []
        # for roi_j in range(sn_roi_act.shape[-2]):
        #     cnt_l_ = [0, 1, 2] * num_ea
        #     random.shuffle(cnt_l_)
        #     cnt_ar.append(cnt_l_)

        for i in range(sn_roi_act_sq.shape[-1]):
            # for roi_j in range(sn_roi_act.shape[-2]):
        #         cnt = cnt_ar[roi_j][i]
        #         new_ar[:, cnt, roi_j, i] = sn_roi_act_sq[:, roi_j, i]

            cnt = cnt_l[i]
            # print(f'{sn_roi_act_sq.shape=}')
            # print(f'{new_ar.shape=}')
            new_ar[j, cnt, :, i] = sn_roi_act_sq[j, :, i]
            # cnt += 1
            # print(f'{cnt=}')
            # if cnt % 3 == 0:
            #     cnt = 0
    return new_ar

def act2conn(sn_roi_act, flatten=True):
    # print(f'ac2conn: {sn_roi_act.shape}')
    t_st = time()
    sn_roi_act0 = sn_roi_act[..., :, None, :]
    sn_roi_act1 = sn_roi_act[..., None, :, :]
    # print('a')
    sn_roi_conn = sn_roi_act0 * sn_roi_act1
    if flatten:
        # print('b')
        trils = np.tril_indices(sn_roi_act.shape[-2], -1)
        sn_roi_conn = sn_roi_conn[..., trils[0], trils[1], :]
        # print(f'{sn_roi_conn.shape} | time needed: {time() - t_st:.2f} s')
    return sn_roi_conn

def plot_ranks(idx2rank, atlas, n_roi=246, do_conn=True):
    if do_conn:
        print(f'{idx2rank=}')
        trils = np.tril_indices(n_roi, -1)
        ar2rank = np.full((n_roi, n_roi), np.nan)
        ar2rank[trils] = idx2rank
        plot_connectivity(ar2rank, atlas=atlas, cbar_label='rank')
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
        quit()

def load_task_rs_data(combine_regions, only_cortical=False):
    sn_roi_act, _, df_sns_l, sns  = \
        pickle_wrap(get_dfs_conn_trials,
                    kwargs={'fp': 'obj7_fMRI', 'single': False,
                            'squeeze': False,
                            'combine_regions': combine_regions},
                    easy_override=False, RAM_cache=True)

    sns = [df['sn'].iloc[0] for df in df_sns_l]
    good_i = [i for i, sn in enumerate(sns) if sn != '133'] # bad rs
    sn_roi_act = sn_roi_act[good_i, :, :]
    sns = [sn for sn in sns if sn != '133'] # bad rs

    f = partial(load_resting_data, combine_regions=combine_regions)
    sn_roi_rs, sns_, _ = load_act_conn(False, f=f,
                                                easy_override=False,
                                       RAM_cache=True)
    assert sns == sns_

    bad_sns = {'116', '117', '110', '130', '232'}
    bad_sns = {'116'}
    bad_i = [i for i, sn in enumerate(sns) if sn in bad_sns]
    sns = [sn for sn in sns if sn not in bad_sns]
    sn_roi_act = np.delete(sn_roi_act, bad_i, axis=0)
    sn_roi_rs = np.delete(sn_roi_rs, bad_i, axis=0)
    n_roi = sn_roi_act.shape[-2]

    # print(f'{sn_roi_rs.shape=}')
    # quit()
    atlas = get_atlas(combine_regions=combine_regions)
    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        sn_roi_act[..., bad_j, :] = np.nan
        sn_roi_rs[..., bad_j, :] = np.nan
    return sn_roi_act, sn_roi_rs, atlas, n_roi, sns

def link_activity(do_generic=True, do_conn=True, verbose=0,
                  shuffle=False, combine_regions=True,
                  split=.3, do_plot=False):
    # Identify a person-specific list of ROIs
    # Identify a group-level list of ROIs
    # See if the fluctuation in resting-state is stronger when modeled
    #   person-specific than group-level

    sn_roi_act, sn_roi_rs, atlas, n_roi, sns = \
        load_task_rs_data(combine_regions, only_cortical=True)

    if do_conn:
        sn_roi_act = stdize(sn_roi_act, nans=True, axis=-1)
        sn_roi_act = act2conn(sn_roi_act)
        sn_roi_rs = stdize(sn_roi_rs, nans=True, axis=-1)
        sn_roi_rs = act2conn(sn_roi_rs)

    if shuffle:
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
    # high_cutoff -= n_nans


    if do_plot:
        gen_idx2rank[(gen_idx2rank > low_cutoff) &
                     (gen_idx2rank < high_cutoff)] = np.nan
        temp = generic_ts
        print(f'{low_cutoff=}')
        print(f'{high_cutoff=}')
        temp[np.isnan(gen_idx2rank)] = np.nan
        # test = (gen_idx2rank > low_cutoff) & (gen_idx2rank < high_cutoff)
        # print(list(gen_idx2rank))
        # print(f'{test=}')
        # print(temp)
        num_nans = np.sum(np.isnan(temp))
        # print(f'{num_nans=}')
        # quit()
        # gen_idx2rank = generic_ts
        plot_ranks(temp, atlas, n_roi=n_roi, do_conn=do_conn)

    # sanity_l = [0, 1, 3, 5, 7, 11, 13, 14, 15, 17, 18, 19, 21, 23, 25, 27, 29, 33, 35, 36, 37, 39, 43, 46, 51, 52, 53, 55, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 82, 83, 85, 87, 89, 92, 93, 94, 95, 102, 103, 108, 111, 112, 113, 116, 118, 120, 121, 123, 125, 127, 135, 143, 147, 149, 150, 151, 153, 154, 155, 156, 157, 159, 162, 163, 164, 166, 167, 169, 170, 171, 172, 173, 175, 178, 179, 180, 181, 188, 190, 191, 192, 193, 194, 195, 196, 197, 201, 202, 203, 204, 206, 207]


    corrs = []
    if verbose: print('Onto looping')
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


        # if sn_i == 0:
        #     n_sanity = len(sanity_l)
        #     n_overlap = len(set(inc_rois) & set(sanity_l))
        #     p_overlap = n_overlap / n_sanity
        #     n_overlap_con = len(set(con_rois) & set(sanity_l))
        #     p_overlap_con = n_overlap_con / n_sanity
            # print(f'{len(sanity_l)=}')
            # print(f'{len(inc_rois)=} | {len(con_rois)=}')
            # print(f'{p_overlap:.1%} | {p_overlap_con:.1%}')
        # print(f'{sorted(inc_rois)=}')
        # quit()

        rs_inc = sn_roi_rs[sn_i, inc_rois, :]
        rs_inc = np.nanmean(rs_inc, axis=0) # TODO: toggle to nan and exclude?
        rs_con = sn_roi_rs[sn_i, con_rois, :]
        rs_con = np.nanmean(rs_con, axis=0)

        n_con_nans = np.sum(np.isnan(rs_con))
        n_inc_nans = np.sum(np.isnan(rs_inc))
        if n_con_nans or n_inc_nans:
            assert n_con_nans == n_inc_nans == 206, \
                f'{n_con_nans=} {n_inc_nans=}'
            print(f'skip: {sns[sn_i]}')

        # plt.scatter(rs_inc, rs_con)
        # plt.show()
        # quit()

        corr, _ = stats.pearsonr(rs_inc, rs_con)
        corrs.append(corr)

        # print(np.sum(np.isnan(rs_con)))
        # rs_con = stdize(rs_con, nans=True)
    corrs = np.array(corrs)
    M_corr = np.mean(corrs)
    generic_str = 'Generic' if do_generic else 'Subject-specific'
    if shuffle:
        print(f'{generic_str} ({split}) | Shuffle: {M_corr=:.3f}')
    else:
        print(f'{generic_str} ({split}): {M_corr=:.3f}')
    return M_corr

def permutation_test_(split=.5, do_conn=False, n_sim=100,
                      combine_regions=False, do_generic=False):
    M_corrs = []
    for _ in range(n_sim):
        M_corr = link_activity(do_generic=do_generic, do_conn=do_conn,
                               split=split, combine_regions=combine_regions,
                               do_plot=False, shuffle=True)
        M_corrs.append(M_corr)
    return M_corrs


def permutation_test(split=.5, do_conn=False, n_sim=100, combine_regions=False,
                     do_generic=False):
    from connsearch import print_list_stats
    kwargs = {'split': split, 'do_conn': do_conn, 'n_sim': n_sim,
              'combine_regions': combine_regions, 'do_generic': do_generic}
    M_corrs = pickle_wrap(permutation_test_, kwargs=kwargs,
                          easy_override=False, RAM_cache=True)
    M_corrs = -np.array(M_corrs)
    print_list_stats(M_corrs)

def plot_by_split(do_conn=True, do_generic=True, combine_regions=True):
    for split in [.25]: # .5, .4,
        # M_corrs = link_activity(do_generic=False, do_conn=do_conn,
        #                       split=split, combine_regions=False,
        #                       do_plot=False, shuffle=False)
        M_corrs = link_activity(do_generic=do_generic, do_conn=do_conn,
                              split=split, combine_regions=combine_regions,
                              do_plot=False)
        permutation_test(split=split, do_conn=do_conn, n_sim=100,
                         do_generic=do_generic,
                         combine_regions=combine_regions)

if __name__ == '__main__':
    # test = np.array([2, 3, 5, -10, 1, 0])
    # print(np.argsort(np.argsort(test)))
    # quit()


    plot_by_split()
    quit()
    # a = {1:1, 2:1}
    # b = {1:1, 2:1}
    # print(a == b)
    # quit()

    # SANITY TEST WHICH RANDOMIZES BY CONDITION,
    #   EVEN THOUGH WE PERSONALLY DO NOT SEE AN A PRIORI REASON

    M_corrs = []
    for _ in range(100):
        corrs_ss = link_activity(do_generic=False, shuffle=True)
    corrs_gen = link_activity(do_generic=True)
    quit()
    # corrs_ss[corrs_ss > 1.0] = np.nan
    print(len(corrs_gen))

    # -0.313 gives p = .04 per my 100 sims
    dif = corrs_gen - corrs_ss


    # corrs_ss[dif > 1.0] = np.nan
    # corrs_gen[dif > 1.0] = np.nan
    # print(f'{dif=}')
    #
    # for x, y in zip(corrs_gen, corrs_ss):
    #     print(f'{x=:.3f} {y=:.3f}')

    # corrs_gen = corrs_gen[corrs_ss < 1.0]
    # corrs_ss = corrs_ss[corrs_ss < 1.0]
    # print(len(corrs_gen))

    t, p = stats.ttest_rel(corrs_gen, corrs_ss, nan_policy='omit')
    print(f'Generic vs. subject-specific: {t=:.3f} {p=:.3f}')
    dif = corrs_gen - corrs_ss
    np.set_printoptions(precision=2, suppress=True)
    print(f'{dif=}')

    plt.hist(dif, bins=10)
    plt.show()

