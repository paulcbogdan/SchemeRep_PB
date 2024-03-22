import os
import pickle

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
                  alt_calc=True, GSR=False, alt_calc2=False,
                  abs_dist=False, rankdata=False):
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
    # print('TEST C')
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
    # print('TEST A')
    # print(sn_roi_rs.shape)
    # quit()
    if do_conn:
        sn_roi_act = stdize(sn_roi_act, nans=True, axis=-1)
        sn_roi_act = act2conn(sn_roi_act, conn_euc=conn_euc)

    # print('TEST B')
    sn_inc_roi_Ms = np.nanmean(sn_roi_act, axis=-1)
    generic_ts, sn_roi_difs = get_effs(sn_inc_roi_Ms, regr=regr)

    n_nans = np.sum(np.isnan(generic_ts))
    n_elements = len(generic_ts)
    nan_cutoff = n_elements - n_nans
    gen_rank2idx = generic_ts.argsort()

    gen_idx2rank = np.argsort(gen_rank2idx).astype(float)
    # with open(f'cache/gen_idx2rank_test.pkl', 'wb') as f:
    #     pickle.dump(gen_idx2rank, f)
    # print(f'{gen_idx2rank=}')
    # quit()

    gen_idx2rank[gen_idx2rank >= nan_cutoff] = np.nan

    low_cutoff = int(nan_cutoff * split)
    # print(f'{low_cutoff=}')
    high_cutoff = int(nan_cutoff * (1 - split))
    # print(f'{high_cutoff=}')
    # print(f'{nan_cutoff=}')
    # quit()

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
    if do_generic:
        rank2idx = gen_rank2idx
        n_nans_sn = n_nans
        nan_idxs = np.argwhere(np.isnan(generic_ts))[:, 0]

    for sn_i in range(n_sn): # , position=0, leave=False)
        if not do_generic:
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

    # print(f'{i2con_rois=}')
    # quit()

    # sn_roi_rs = stdize(sn_roi_rs, nans=True, axis=-1)
    # sn_roi_rs = act2conn(sn_roi_rs, conn_euc=conn_euc)

    for sn_i in range(sn_roi_rs.shape[0]):
        if sns[sn_i] == '138': continue
        if sns[sn_i] == '131': continue
        if sns[sn_i] == '135': continue


        inc_rois, con_rois = i2inc_rois[sn_i], i2con_rois[sn_i]
        max_inc_rois = np.nanmax(inc_rois)
        # print(f'{max_inc_rois=}')
        max_con_rois = np.nanmax(con_rois)
        # print(f'{max_con_rois=}')
        # print(f'{sn_roi_rs.shape=}')
        # quit()

        trils = np.tril_indices(sn_roi_rs[sn_i, :, :].shape[0], -1)
        trils = list(zip(*trils))
        rs_inc = []
        # for i, j in trils:
        # if i in inc_rois and j in inc_rois:
        for idx in inc_rois:
            i, j = trils[idx]
            rs_inc.append(sn_roi_rs[sn_i, i, :] * sn_roi_rs[sn_i, j, :])
        rs_inc = np.array(rs_inc)
        rs_con = []
        for idx in con_rois:
            i, j = trils[idx]
            rs_con.append(sn_roi_rs[sn_i, i, :] * sn_roi_rs[sn_i, j, :])
        rs_con = np.array(rs_con)

        # print(f'{rs_inc.shape=}')
        # print(f'{rs_con.shape=}')
        # print(len(trils))
        # quit()

        # rs_inc = sn_roi_rs[sn_i, inc_rois, :]
        # rs_con = sn_roi_rs[sn_i, con_rois, :]


        n_con_nans = np.sum(np.isnan(rs_con))
        n_inc_nans = np.sum(np.isnan(rs_inc))
        # print(f'{n_con_nans=} | {n_inc_nans=}')
        if (n_con_nans or n_inc_nans) and not HCP:
            print(f'skip: {sns[sn_i]}')
            assert (n_con_nans == n_inc_nans) and (n_inc_nans in [206, 37]), \
                f'{n_con_nans=} {n_inc_nans=}'
            continue

        rs_inc = stdize(rs_inc, nans=True, axis=-1, rankdata=rankdata)
        rs_con = stdize(rs_con, nans=True, axis=-1, rankdata=rankdata)
        if alt_calc and M_after:
            pass
        elif alt_calc2:
            rs_inc = np.nanmean(rs_inc, axis=0)  # TODO: toggle to nan and exclude?
            rs_con = np.nanmean(rs_con, axis=0)
            corr = np.mean(np.abs(rs_inc - rs_con))
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
            if abs_dist:
                corr = np.abs(rs_inc[:, None] - rs_con[None, :])
            else:
                corr = rs_inc[:, None] * rs_con[None, :]
            corr = np.nanmean(corr, axis=-1)
            corr = np.arctanh(corr)
            corr = np.nanmean(corr, axis=(0, 1))
        else:
            rs_inc = np.nanmean(rs_inc, axis=0) # TODO: toggle to nan and exclude?
            rs_con = np.nanmean(rs_con, axis=0)
            if abs_dist:
                corr = np.mean(np.abs(rs_inc - rs_con))
            else:
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
              f'{Fore.LIGHTYELLOW_EX}{M_corr=:.4f}{Fore.RESET}')
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



def plot_by_split(do_conn=True, do_generic=True, combine_regions=False,
                  # fp_task='EMOTION',
                  fp_task='obj7_fMRI',
                  HCP=True,
                  abs_dist=True, rankdata=False, alt_calc=False,
                  M_after=False, regr=True,

                  light=False, medium=True,
                  near_OG=False, trad=False, true_OG=False,
                  other_task=False,
                  conn_euc=False,
                  only_cortical=True, alt_shuffle=False,
                  task_and_rs=False,
                  GSR=False, alt_calc2=False,
                  do_hit_hit=False,
                  ):
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
              'alt_calc2': alt_calc2,
              'abs_dist': abs_dist,
              'rankdata': rankdata,
              # 'weighted': weighted
              }
    assert not (alt_calc and alt_calc2)
    # assert not (do_conn and not combine_regions)
    for split in [.1]: # . [0.3]:# .5, .4, .5, .4, .3, .2,   .2, .1, .05 .3,
        assert not (do_conn and not combine_regions and M_after and
                    split > 0.1)

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

