import os

from tqdm import tqdm

from HCP import load_HCP_act
from atlas_utils import get_atlas
from get_HCP_act import load_HCP
from old.analyze_ROIs import setup_colors
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from utils import pickle_wrap, stdize
from load_more import get_dfs_conn_trials, load_resting_data, load_act_conn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from time import time
from functools import partial
from pathlib import Path
from pprint import pprint
from colorama import Fore
from functools import cache

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

def shuffle_rows_only_13(sn_roi_act):
    sn_roi_act[:, 1] = np.nan
    new_ar = np.full(sn_roi_act.shape, np.nan)
    # print(sn_roi_act.shape)
    # quit()
    for j in range(sn_roi_act.shape[0]):
        for k in [0, 2]:
            non_k = 0 if k == 2 else 2
            n_non_nan = np.sum(~np.isnan(sn_roi_act[j, k, 0, :]))
            # print(f'{n_non_nan=}')
            # continue
            flipper = [False, True] * (n_non_nan // 2)
            flipper += list(random.choices([False, True], k=1))
            random.shuffle(flipper)
            cnt = 0
            for trial in range(sn_roi_act.shape[-1]):
                if np.isnan(sn_roi_act[j, k, 0, trial]):
                    continue
                # print(f'{cnt=}')
                # print(f'{len(flipper)=}')
                if flipper[cnt]:
                    new_ar[j, k, :, trial] = sn_roi_act[j, k, :, trial]
                else:
                    new_ar[j, non_k, :, trial] = sn_roi_act[j, k, :, trial]
                cnt += 1

    return new_ar


def shuffle_rows_only_13_(sn_roi_act):
    sn_roi_act[:, 1] = np.nan
    new_ar = np.full(sn_roi_act.shape, np.nan)
    sn_roi_act_sq = np.nanmean(sn_roi_act, axis=1)
    # print(sn_roi_act_sq.shape)
    # quit()
    for j in range(sn_roi_act_sq.shape[0]):
        non_nan_i = np.argwhere(~np.isnan(sn_roi_act_sq[j, 0]))[:, 0]
        random.shuffle(non_nan_i)
        i0 = non_nan_i[::2]
        i1 = non_nan_i[1::2]
        # print(sn_roi_act_sq.shape)
        # print(sn_roi_act_sq[j, :, i0])
        new_ar[j, 0, :, i0] = sn_roi_act_sq[j, :, i0]
        new_ar[j, 2, :, i1] = sn_roi_act_sq[j, :, i1]
        # print(new_ar[j, 0, 0, :])
        # print(new_ar[j, 2, 0, i1])
        # quit()
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

@ cache
def load_task_rs_data(combine_regions, fp='vis7_fMRI', only_cortical=False,
                      do_hit_hit=True, HCP=False, light=False, medium=False,
                      trad=False, other_task=False, near_OG=False,
                      true_OG=False, strict_sns=None, GSR=False):
    # sn_roi_act, _, df_sns_l, sns  = \
        # pickle_wrap(get_dfs_conn_trials,
        #             kwargs={'fp': fp, 'single': False,
        #                     'squeeze': False,
        #                     'combine_regions': combine_regions},
        #             easy_override=False, RAM_cache=True)
        #
    hcp_names = {'EMOTION'}
    if fp in hcp_names:
        kwargs = {'task': fp,
                  'lr_only': False,
                  'combine_regions': combine_regions}
        sn_roi_act = pickle_wrap(load_HCP, kwargs=kwargs)
        sns = list(range(sn_roi_act.shape[0]))
    else:
        if strict_sns is None:
            strict_sns = True if other_task else False
        kwargs = {'fp': fp,
                  'key': 'inc',
                  'atlas_name': 'BNA',
                  'key_vals': (1, 2, 3),
                  'get_df_sn': True,
                  'combine_regions': combine_regions,
                  'strict_sns': strict_sns
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
        # hcp_kwargs = {'N': 50, 'combine_regions': combine_regions,
        #               'GSR': False}
        f = partial(load_HCP_act, N=300,
                    RS=False, clean_confounds=True, LSS=False, LSA=False,
                    compcor=True, GSR=False,
                    combine_regions=combine_regions)

        sn_roi_rs, sns_ = pickle_wrap(f,#load_HCP_acTrue, verbose=0,
                                      easy_override=False,
                                      RAM_cache=True)
        print(f'{sn_roi_rs.shape=}')

        if GSR:
            sn_roi_rs = stdize(sn_roi_rs, axis=2, nans=True)
            sn_roi_rs = stdize(sn_roi_rs, axis=1, nans=True)


        sns = sns_
    else:
        if other_task:
            acts = []
            for fp in ['bl7_fMRI', 'con7_fMRI', 'vis7_fMRI']:
                kwargs['fp'] = fp
                _, _, _, sn_roi_rs, df_sns_l_ = \
                    pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                                easy_override=False, verbose=0, cache_dir='cache',
                                RAM_cache=True)
                sn_roi_rs = np.nanmean(sn_roi_rs, axis=1)
                acts.append(sn_roi_rs)
                # print(f'{fp} | {sn_roi_act.shape=}')
            sn_roi_rs = np.concatenate(acts, axis=-1)
            sns_ = [df['sn'].iloc[0] for df in df_sns_l_]

            # print(f'{sn_roi_rs.shape=}')
            # quit()
        else:
        # if light:
        #     f = partial(load_resting_data, combine_regions=combine_regions,
        #                 sns_key=fp, light=True)
        # else:
            f = partial(load_resting_data, combine_regions=combine_regions,
                        sns_key='all' if strict_sns else fp,
                        light=light, medium=medium, trad=trad,
                        near_OG=near_OG, true_OG=true_OG)
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
        print('Pruned subcortical')

    return sn_roi_act, sn_roi_rs, atlas, n_roi, sns

def get_effs(sn_inc_roi_Ms, regr=False):
    if regr:
        n_sn = sn_inc_roi_Ms.shape[0]
        sn_flat = sn_inc_roi_Ms.reshape(-1, sn_inc_roi_Ms.shape[-1])
        regressors = np.array([[-1, 0, 1] * n_sn]).T

        XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
        XTX_invX = np.dot(XTX_inv, regressors.T)
        betas = np.dot(XTX_invX, sn_flat)

        Y_pred = np.dot(regressors, betas)
        residual = sn_flat - Y_pred
        sigma_s = np.sum(residual ** 2, axis=0) / (n_sn * 2 - 2)
        ss_x = np.sum(regressors ** 2, axis=0)
        var_beta = sigma_s / ss_x

        z = betas / np.sqrt(var_beta)
        return z[0, :], None
    else:
        sn_roi_difs = sn_inc_roi_Ms[:, 2, :] - sn_inc_roi_Ms[:, 0, :]
        generic_difs = np.nanmean(sn_roi_difs, axis=0)
        generic_Ns = np.sum(~np.isnan(sn_roi_difs), axis=0)
        generic_SDs = np.nanstd(sn_roi_difs, axis=0)
        generic_SEs = generic_SDs / np.sqrt(generic_Ns)  # TODO: toggle vs. just dif
        generic_ts = generic_difs / generic_SEs
        return generic_ts, generic_difs

def link_activity(do_generic=True, do_conn=True, verbose=0,
                  shuffle=False, combine_regions=True,
                  split=.3, do_plot=False, fp_task='obj7_fMRI',
                  do_hit_hit=False, shuffle_ss=False, HCP=False,
                  light=False, medium=False, conn_euc=True, trad=True,
                  other_task=False, regr=False, near_OG=False,
                  M_after=False, true_OG=False, only_cortical=True,
                  alt_shuffle=False, task_and_rs=False,
                  alt_calc=False, GSR=False):
    if alt_shuffle:
        fp_task = 'bl7_fMRI'
    if task_and_rs:
        strict_sns = True
        other_task = False
    else:
        strict_sns = False

    sn_roi_act, sn_roi_rs, atlas, n_roi, sns = \
        load_task_rs_data(combine_regions, only_cortical=only_cortical,
                          fp=fp_task, do_hit_hit=do_hit_hit,
                          HCP=HCP, light=light, medium=medium,
                          trad=trad, other_task=other_task,
                          strict_sns=strict_sns,
                          near_OG=near_OG, true_OG=true_OG,
                          GSR=GSR)
    # print(f'{sn_roi_rs.shape=}')
    if task_and_rs:
        _, sn_roi_rs_, _, _, _ = \
            load_task_rs_data(combine_regions, only_cortical=only_cortical,
                              fp=fp_task, do_hit_hit=do_hit_hit,
                              HCP=HCP, light=light, medium=medium,
                              trad=trad, other_task=True,
                              strict_sns=strict_sns,
                              near_OG=near_OG, true_OG=true_OG)
        # print(f'{sn_roi_rs_.shape=}')
        sn_roi_rs = np.concatenate([sn_roi_rs, sn_roi_rs_], axis=-1)
    # print(f'{sn_roi_rs.shape=}')
    # quit()
    n_sn = len(sns)
    # print(f'TOAST {sn_roi_rs.shape=}')


    if shuffle and not shuffle_ss:
        # NOTE: On 3/15/2024, I moved this from after do_conn to before
        sn_roi_act = shuffle_rows(sn_roi_act)

    if do_conn:
        sn_roi_act = stdize(sn_roi_act, nans=True, axis=-1)
        sn_roi_act = act2conn(sn_roi_act, conn_euc=conn_euc)
        sn_roi_rs = stdize(sn_roi_rs, nans=True, axis=-1)
        sn_roi_rs = act2conn(sn_roi_rs, conn_euc=conn_euc)
        sn_roi_rs = stdize(sn_roi_rs, nans=True, axis=-1)

    sn_inc_roi_Ms = np.nanmean(sn_roi_act, axis=-1)
    generic_ts, sn_roi_difs = get_effs(sn_inc_roi_Ms, regr=regr)

    n_nans = np.sum(np.isnan(generic_ts))
    n_elements = len(generic_ts)
    nan_cutoff = n_elements - n_nans
    gen_rank2idx = generic_ts.argsort()

    gen_idx2rank = np.argsort(gen_rank2idx).astype(float)

    gen_idx2rank[gen_idx2rank >= nan_cutoff] = np.nan

    low_cutoff = int(nan_cutoff * split)
    high_cutoff = int(nan_cutoff * (1 - split))


    if do_plot:
        gen_idx2rank[(gen_idx2rank > low_cutoff) &
                     (gen_idx2rank < high_cutoff)] = np.nan
        temp = generic_ts

        temp[np.isnan(gen_idx2rank)] = np.nan

        plot_ranks(temp, atlas, n_roi=n_roi, do_conn=do_conn, split=split)

        idx_to_cnt_con = np.zeros(len(temp))
        idx_to_cnt_inc = np.zeros(len(temp))
        for sn_i in range(n_sn): # , position=0, leave=False)
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
    for sn_i in range(n_sn): # , position=0, leave=False)
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
        con_rois = rank2idx[high_cutoff:-n_nans_sn or None] # avoids -0

        assert np.all([(j not in inc_rois) for j in nan_idxs]), \
            f'Bad in inc_rois: {inc_rois=}'
        assert np.all([(j not in con_rois) for j in nan_idxs]), \
            f'Bad in con_rois: {inc_rois=}'

        i2con_rois.append(con_rois)
        i2inc_rois.append(inc_rois)

    i2con_rois = np.array(i2con_rois)
    i2inc_rois = np.array(i2inc_rois)

    if shuffle_ss:
        permuter = np.random.permutation(i2con_rois.shape[0])
        i2con_rois = i2con_rois[permuter]
        i2inc_rois = i2inc_rois[permuter]

    inc_rois_real = [974, 891, 964, 1008, 567, 601, 78, 105, 723, 109, 892, 107, 889, 897, 393, 111, 79, 709, 761, 301,
                      939, 308, 746, 887, 66, 933, 305, 808, 388, 576, 1018, 40, 387, 760, 765, 602, 568, 392, 1009, 82,
                      532, 986, 300, 1032, 747, 848, 759, 137, 685, 745, 95, 726, 425, 724, 389, 970, 578, 303, 988,
                      898, 220, 143, 727, 556, 710, 83, 629, 718, 45, 722, 56, 112, 403, 86, 476, 1020, 976, 766, 816,
                      982, 913, 46, 966, 113, 38, 762, 716, 1026, 70, 99, 574, 304, 181, 940, 483, 80, 90, 81, 416, 905,
                      91, 758, 145, 335, 972, 687, 142, 484, 475, 59, 764, 934, 856, 74, 962, 619, 914, 276, 391, 302,
                      623, 110, 213, 135, 754, 55, 653, 117, 708, 84, 907, 49, 402, 528, 221, 748, 412, 524, 97, 651,
                      599, 965, 201, 683, 96, 119, 643, 67, 309, 824, 717, 139, 26, 888, 148, 13, 575, 446, 93, 613, 60,
                      893, 542, 290, 836, 174, 728, 1030, 200, 546, 1031, 351, 75, 413, 525, 277, 585, 721, 205, 405,
                      24, 41, 929, 361, 612, 144, 671, 676, 689, 20, 544, 936, 210, 573, 784, 133, 149, 424, 882, 54,
                      386, 641, 838, 677, 806, 597, 134, 769, 540, 586, 415, 707, 291, 1027, 141, 12, 211, 640, 16, 533,
                      292, 456, 419, 451, 414, 515, 344, 807, 538, 370, 798, 909, 603, 352, 43, 513, 316, 1033, 212,
                      254, 559, 935, 106, 44, 1010, 620, 417, 743, 172, 22, 85, 810, 796, 87, 890, 720, 642, 610, 280,
                      930, 168, 673, 788, 390, 72, 543, 800, 535, 756, 307, 842, 1021, 903, 481, 992, 372, 288, 684,
                      336, 369, 474, 647, 281, 840, 496, 594, 646, 846, 100, 418, 530, 652, 990, 645, 42, 770, 1016,
                      318, 607, 755, 431, 453, 147, 615, 485, 987, 608, 162, 138, 545]
    # inc_rois_real = [974, 891, 964, 1008, 567, 601, 78, 105, 723, 109, 892, 107, 889, 897, 393, 111, 79, 709, 761, 301,
    #                   939, 308, 746, 887, 66, 933, 305, 808, 388, 576, 1018, 40, 387, 760, 765, 602, 568, 392, 1009, 82,
    #                   532, 986, 300, 1032, 747, 848, 759, 137, 685, 745, 95, 726, 425, 724, 389, 970, 578, 303, 988,
    #                   898, 220, 143, 727, 556, 710, 83, 629, 718, 45, 722, 56, 112, 403, 86, 476, 1020, 976, 766, 816,
    #                   982, 913, 46, 966, 113, 38, 762, 716, 1026, 70, 99, 574, 304, 181, 940, 483, 80, 90, 81, 416, 905,
    #                   91, 758, 145, 335, 972, 687, 142, 484, 475, 59, 764, 934, 856, 74, 962, 619, 914, 276, 391, 302,
    #                   623, 110, 213, 135, 754, 55, 653, 117, 708, 84, 907, 49, 402, 528, 221, 748, 412, 524, 97, 651,
    #                   599, 965, 201, 683, 96, 119, 643, 67, 309, 824, 717, 139, 26, 888, 148, 13, 575, 446, 93, 613, 60,
    #                   893, 542, 290, 836, 174, 728, 1030, 200, 546, 1031, 351, 75, 413, 525, 277, 585, 721, 205, 405,
    #                   24, 41, 929, 361, 612, 144, 671, 676, 689, 20, 544, 936, 210, 573, 784, 133, 149, 424, 882, 54,
    #                   386, 641, 838, 677, 806, 597, 134, 769, 540, 586, 415, 707, 291, 1027, 141, 12, 211, 640, 16, 533,
    #                   292, 456, 419, 451, 414, 515, 344, 807, 538, 370, 798, 909, 603, 352, 43, 513, 316, 1033, 212,
    #                   254, 559, 935, 106, 44, 1010, 620, 417, 743, 172, 22, 85, 810, 796, 87, 890, 720, 642, 610, 280,
    #                   930, 168, 673, 788, 390, 72, 543, 800, 535, 756, 307, 842, 1021, 903, 481, 992, 372, 288, 684,
    #                   336, 369, 474, 647, 281, 840, 496, 594, 646, 846, 100, 418, 530, 652, 990, 645, 42, 770, 1016,
    #                   318, 607, 755, 431, 453, 147, 615, 485, 987, 608, 162, 138, 545, 445, 487, 505, 132, 182, 579,
    #                   404, 283, 828, 839, 534, 590, 131, 850, 289, 108, 589, 941, 103, 486, 681, 114, 715, 98, 164, 921,
    #                   670, 455, 126, 504, 606, 71, 881, 1000, 565, 306, 444, 50, 931, 679, 948, 423, 968, 609, 136, 527,
    #                   346, 977, 925, 954, 497, 36, 355, 611, 672, 508, 282, 566, 398, 879, 880, 946, 634, 699, 64, 998,
    #                   458, 454, 285, 94, 400, 130, 151, 334, 757, 650, 488, 558, 298, 906, 163, 924, 686, 35, 173, 557,
    #                   923, 910, 975, 450, 247, 420, 407, 549, 501, 92, 852, 312, 482, 863, 841, 253, 901, 580, 146, 521,
    #                   782, 812, 194, 339, 473, 847, 284, 1028, 529, 865, 353, 262, 539, 644, 448, 408, 581, 421, 983,
    #                   820, 330, 801, 1001, 682, 18, 129, 548, 509, 869, 214, 447, 826, 263, 449, 1014, 261, 471, 736,
    #                   541, 605, 675, 287, 627, 744, 902, 600, 780, 216, 377, 636, 688, 617, 167, 776, 631, 942, 799,
    #                   313, 190, 1012, 248, 753, 512, 491, 738, 362, 478, 872, 725, 371, 466, 57, 217, 37, 192, 536, 719,
    #                   264, 34, 472, 854, 553, 279, 47, 33, 844, 53, 219, 32, 319, 635, 178, 950, 637, 311, 457, 396,
    #                   674, 583, 932, 894]

    inc_rois_real = set(inc_rois_real)
    con_rois_real = [225, 945, 461, 328, 1029, 229, 373, 519, 588, 360, 443, 154, 310, 587, 952, 775, 124, 255, 156, 333, 996, 984, 121, 571, 706, 814, 215, 429, 662, 638, 739, 118, 237, 693, 862, 700, 232, 240, 502, 786, 395, 441, 963, 29, 803, 442, 275, 694, 877, 938, 773, 233, 520, 28, 274, 832, 618, 422, 944, 115, 332, 953, 712, 257, 327, 433, 11, 204, 51, 230, 750, 704, 256, 259, 733, 837, 171, 779, 994, 470, 830, 690, 161, 957, 231, 851, 624, 385, 900, 493, 495, 614, 834, 490, 278, 325, 273, 324, 242, 52, 654, 188, 464, 555, 855, 375, 0, 596, 666, 246, 179, 1013, 835, 656, 1005, 272, 89, 258, 6, 25, 915, 813, 730, 236, 771, 805, 196, 668, 88, 997, 734, 440, 183, 228, 195, 947, 886, 927, 1024, 729, 234, 499, 857, 208, 185, 919, 633, 955, 384, 102, 329, 23, 657, 740, 951, 238, 165, 518, 155, 664, 169, 711, 632, 895, 778, 157, 980, 410, 363, 797, 9, 767, 10, 342, 981, 809, 293, 224, 961, 249, 31, 206, 160, 849, 222, 827, 908, 918, 197, 245, 772, 626, 815, 170, 267, 349, 294, 177, 338, 7, 260, 896, 701, 296, 969, 494, 331, 368, 364, 186, 958, 209, 271, 251, 297, 299, 658, 184, 1025, 125, 917, 1007, 341, 152, 819, 244, 845, 268, 226, 625, 866, 376, 176, 665, 469, 622, 166, 591, 17, 101, 202, 381, 122, 323, 366, 928, 153, 207, 794, 768, 227, 337, 985, 995, 783, 4, 781, 867, 379, 971, 1, 1002, 873, 792, 878, 868, 916, 825, 252, 661, 8, 787, 1015, 382, 833, 223, 439, 468, 159, 266, 463, 785, 697, 380, 438, 189, 123, 250, 959, 859, 295, 795, 823, 3, 875, 203, 378, 467, 562, 1003, 561, 660, 874, 876, 821, 175, 696, 920, 187, 793, 437]
    # con_rois_real = [452, 1017, 822, 911, 560, 1006, 411, 621, 459, 802, 1011, 523, 127, 39, 5, 630, 861, 695, 399,
    #                   552, 286, 649, 77, 973, 698, 741, 432, 522, 345, 663, 667, 235, 409, 320, 860, 63, 604, 514, 517,
    #                   477, 356, 680, 492, 354, 27, 577, 871, 537, 551, 15, 322, 270, 500, 199, 511, 198, 120, 14, 365,
    #                   989, 598, 19, 436, 993, 678, 76, 507, 506, 350, 714, 582, 763, 912, 128, 554, 616, 406, 937, 1022,
    #                   547, 428, 570, 593, 732, 584, 68, 691, 731, 648, 315, 503, 628, 73, 180, 804, 943, 516, 817, 592,
    #                   811, 639, 321, 340, 191, 564, 692, 65, 243, 1019, 21, 462, 239, 572, 359, 358, 426, 69, 1004, 343,
    #                   870, 61, 489, 465, 2, 427, 655, 116, 401, 367, 702, 705, 885, 140, 158, 853, 790, 749, 430, 843,
    #                   397, 434, 550, 858, 829, 58, 922, 752, 777, 569, 960, 347, 899, 864, 703, 460, 926, 713, 241, 883,
    #                   1034, 563, 774, 150, 48, 269, 595, 818, 531, 265, 669, 978, 735, 659, 193, 326, 991, 1023, 742,
    #                   967, 526, 218, 479, 949, 317, 884, 510, 30, 737, 104, 751, 394, 789, 791, 480, 62, 357, 374, 348,
    #                   904, 383, 498, 979, 314, 999, 956, 831, 435, 225, 945, 461, 328, 1029, 229, 373, 519, 588, 360,
    #                   443, 154, 310, 587, 952, 775, 124, 255, 156, 333, 996, 984, 121, 571, 706, 814, 215, 429, 662,
    #                   638, 739, 118, 237, 693, 862, 700, 232, 240, 502, 786, 395, 441, 963, 29, 803, 442, 275, 694, 877,
    #                   938, 773, 233, 520, 28, 274, 832, 618, 422, 944, 115, 332, 953, 712, 257, 327, 433, 11, 204, 51,
    #                   230, 750, 704, 256, 259, 733, 837, 171, 779, 994, 470, 830, 690, 161, 957, 231, 851, 624, 385,
    #                   900, 493, 495, 614, 834, 490, 278, 325, 273, 324, 242, 52, 654, 188, 464, 555, 855, 375, 0, 596,
    #                   666, 246, 179, 1013, 835, 656, 1005, 272, 89, 258, 6, 25, 915, 813, 730, 236, 771, 805, 196, 668,
    #                   88, 997, 734, 440, 183, 228, 195, 947, 886, 927, 1024, 729, 234, 499, 857, 208, 185, 919, 633,
    #                   955, 384, 102, 329, 23, 657, 740, 951, 238, 165, 518, 155, 664, 169, 711, 632, 895, 778, 157, 980,
    #                   410, 363, 797, 9, 767, 10, 342, 981, 809, 293, 224, 961, 249, 31, 206, 160, 849, 222, 827, 908,
    #                   918, 197, 245, 772, 626, 815, 170, 267, 349, 294, 177, 338, 7, 260, 896, 701, 296, 969, 494, 331,
    #                   368, 364, 186, 958, 209, 271, 251, 297, 299, 658, 184, 1025, 125, 917, 1007, 341, 152, 819, 244,
    #                   845, 268, 226, 625, 866, 376, 176, 665, 469, 622, 166, 591, 17, 101, 202, 381, 122, 323, 366, 928,
    #                   153, 207, 794, 768, 227, 337, 985, 995, 783, 4, 781, 867, 379, 971, 1, 1002, 873, 792, 878, 868,
    #                   916, 825, 252, 661, 8, 787, 1015, 382, 833, 223, 439, 468, 159, 266, 463, 785, 697, 380, 438, 189,
    #                   123, 250, 959, 859, 295, 795, 823, 3, 875, 203, 378, 467, 562, 1003, 561, 660, 874, 876, 821, 175,
    #                   696, 920, 187, 793, 437]

    con_rois_real = set(con_rois_real)



    for sn_i in range(sn_roi_rs.shape[0]):
        if sns[sn_i] == '138': continue
        if sns[sn_i] == '131': continue
        if sns[sn_i] == '135': continue


        inc_rois, con_rois = i2inc_rois[sn_i], i2con_rois[sn_i]
        # print(f'{list(inc_rois)=}')
        # print(f'{list(con_rois)=}')
        # quit()
        # inc_overlap = inc_rois_real.intersection(inc_rois)
        # p_inc_overlap = len(inc_overlap) / len(inc_rois)
        # con_overlap = con_rois_real.intersection(con_rois)
        # p_con_overlap = len(con_overlap) / len(con_rois)
        # print(f'{len(con_rois_real)=}|{len(con_rois)=}')
        p_inc_reverse_overlap = (len(con_rois_real.intersection(inc_rois)) /
                                 len(inc_rois))
        p_con_reverse_overlap = (len(inc_rois_real.intersection(con_rois)) /
                                    len(con_rois))
        # if sn_i == 0:
        #     print(f'{p_inc_overlap=:.3f} | {p_con_overlap=:.3f}')
            # print(f'\t{p_inc_reverse_overlap=:.3f} | {p_con_reverse_overlap=:.3f}')

        # print(f'{list(inc_rois)=}')
        # print(f'{list(con_rois)=}')
        # quit()
        # print(f'{sn_roi_rs.shape=}')
        # quit()
        # print(f'{sn_i.shape=}')
        # print(f'{inc_rois.shape=}')
        # print(f'{inc_rois=}')
        # print(f'{con_rois=}')
        # quit()

        rs_inc = sn_roi_rs[sn_i, inc_rois, :]
        rs_con = sn_roi_rs[sn_i, con_rois, :]
        # rs_inc_weights = rank2bf[inc_rois]
        # rs_con_weights = rank2bf[con_rois]
        # print(f'{rs_con_weights=}')
        # quit()

        n_con_nans = np.sum(np.isnan(rs_con))
        n_inc_nans = np.sum(np.isnan(rs_inc))
        # print(f'{n_con_nans=} | {n_inc_nans=}')
        if (n_con_nans or n_inc_nans) and not HCP:
            print(f'skip: {sns[sn_i]}')
            assert (n_con_nans == n_inc_nans) and (n_inc_nans in [206, 37]), \
                f'{n_con_nans=} {n_inc_nans=}'
            continue


        if alt_calc and M_after:
            pass
        elif alt_calc:
            # spl = len(rs_inc) // 2
            rs_inc0 = np.nanmean(rs_inc[::2], axis=0)  # TODO: toggle to nan and exclude?
            rs_inc1 = np.nanmean(rs_inc[1::2], axis=0)
            sim00, _ = stats.spearmanr(rs_inc0, rs_inc1)

            rs_con0 = np.nanmean(rs_con[::2], axis=0)
            rs_con1 = np.nanmean(rs_con[1::2], axis=0)
            sim11, _ = stats.spearmanr(rs_con0, rs_con1)

            dif01, _ = stats.spearmanr(rs_inc0, rs_con1)
            dif10, _ = stats.spearmanr(rs_inc1, rs_con0)
            dif00, _ = stats.spearmanr(rs_inc0, rs_con0)
            dif11, _ = stats.spearmanr(rs_inc1, rs_con1)

            sim00, sim11, dif01, dif10, dif11, dif00 = (
                np.arctanh([sim00, sim11, dif01, dif10, dif11, dif00]))
            corr = (sim00 + sim11) / 2
            corr -= (dif01 + dif10 + dif11 + dif00) / 4
            # corr = (sim00 + sim11 + dif01 + dif10) / 4
            # print(f'{sim00=:.3f} | {sim11=:.3f} | {dif01=:.3f} | '
            #       f'{dif10=:.3f} | {corr=:.3f}')

        elif M_after:
            # print(f'{np.nanmean(rs_inc, axis=-1)}')
            # quit()
            # print(f'{rs_con.shape=}')
            corr = rs_inc[:, None] * rs_con[None, :]
            # print(f'{corr.shape=}')
            corr = np.nanmean(corr, axis=-1)
            corr = np.arctanh(corr)
            corr = np.nanmean(corr, axis=(0, 1))
        else:
            if weighted:
                rs_inc = np.nansum(rs_inc * rs_inc_weights[:, None], axis=0)
                rs_inc /=  np.nansum(rs_inc_weights)
                rs_con = np.nansum(rs_con * rs_con_weights[:, None], axis=0)
            else:
                rs_inc = np.nanmean(rs_inc, axis=0) # TODO: toggle to nan and exclude?
                rs_con = np.nanmean(rs_con, axis=0)

            corr, _ = stats.spearmanr(rs_inc, rs_con) # TODO: toggle?
            corr = np.arctanh(corr)
        corrs.append(corr)

    corrs = np.array(corrs)
    # print(f'{list(corrs)=}')
    # quit()
    M_corr = np.mean(corrs)
    generic_str = 'Generic' if do_generic else 'Subject-specific'
    if shuffle_ss:
        print(f'{generic_str} ({split}) | Shuffle SS: {M_corr=:.4f}')
    elif shuffle:
        print(f'{generic_str} ({split}) | Shuffle: {M_corr=:.4f}')
    else:
        print(f'{generic_str} ({split}): '
              f'{Fore.LIGHTYELLOW_EX}{M_corr=:.3f}{Fore.RESET}')
    return M_corr

def permutation_test_(n_sim=1000, **kwargs):
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
                              # True if kwargs['alt_shuffle'] else False)
        M_corrs = -np.array(M_corrs)
        if kwargs['conn_euc']: M_corrs *= -1
        if kwargs['alt_calc']: M_corrs *= -1
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
                  do_hit_hit=False,
                  fp_task='EMOTION',
                  # fp_task='obj7_fMRI',
                  HCP=True,
                  light=False, medium=True,
                  near_OG=False, trad=False, true_OG=False,
                  other_task=False, regr=False,
                  conn_euc=False, M_after=False,
                  only_cortical=True, alt_shuffle=False,
                  task_and_rs=False, alt_calc=True,
                  GSR=True):
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
              'trad': trad,
              'other_task': other_task,
              'regr': regr,
              'near_OG': near_OG,
              'M_after': M_after,
              'true_OG': true_OG,
              'only_cortical': only_cortical,
              'alt_shuffle': alt_shuffle,
              'task_and_rs': task_and_rs,
              'alt_calc': alt_calc,
              'GSR': GSR,
              # 'weighted': weighted
              }
    assert not (do_conn and not combine_regions and M_after)
    assert not (do_conn and not combine_regions)
    for split in [.5, .4, .3, .2, .1, .05]: # . [0.3]:# .5, .4,   .2, .1, .05 .3,
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
        permutation_test(n_sim=1000, **kwargs)

if __name__ == '__main__':
    plot_by_split()

