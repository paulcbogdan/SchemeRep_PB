import os

from tqdm import tqdm

from analyze_rs import load_act_conn, load_resting_data
from atlas_utils import get_atlas
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
        new_ar[:, cnt, :, i] = sn_roi_act_sq[:, :, i]
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

def plot_ranks(idx2rank, atlas, nroi=246, do_conn=True):
    if do_conn:
        trils = np.tril_indices(nroi, -1)
        ar2rank = np.full((nroi, nroi), np.nan)
        ar2rank[trils] = idx2rank
        plot_connectivity(ar2rank, atlas=atlas, cbar_label='rank')
        quit()



def link_activity(do_generic=True, do_conn=True, verbose=0,
                  shuffle=False, combine_regions=True,
                  split=.3):
    # Identify a person-specific list of ROIs
    # Identify a group-level list of ROIs
    # See if the fluctuation in resting-state is stronger when modeled
    #   person-specific than group-level

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
    bad_i = [i for i, sn in enumerate(sns) if sn in bad_sns]
    sns = [sn for sn in sns if sn not in bad_sns]
    sn_roi_act = np.delete(sn_roi_act, bad_i, axis=0)
    sn_roi_rs = np.delete(sn_roi_rs, bad_i, axis=0)
    nrois = sn_roi_act.shape[-2]
    # print(f'{sn_roi_rs.shape=}')
    # quit()

    atlas = get_atlas(combine_regions=combine_regions)

    if do_conn:
        sn_roi_act = act2conn(sn_roi_act)
        sn_roi_rs = act2conn(sn_roi_rs)

    corrs = []
    if shuffle:
        sn_roi_act = shuffle_rows(sn_roi_act)

    sn_roi_M = np.nanmean(sn_roi_act, axis=-1)
    sn_roi_eff = sn_roi_M[:, 2, :] - sn_roi_M[:, 0, :]

    generic_effs = np.nanmean(sn_roi_eff, axis=0)
    generic_effs /= np.nanstd(generic_effs, axis=0)
    gen_rank2idx = generic_effs.argsort()
    gen_idx2rank = np.argsort(gen_rank2idx)

    # split = 0.25
    low_cutoff = int(len(gen_rank2idx) * split)
    high_cutoff = int(len(gen_rank2idx) * (1 - split))
    # idx_cutoff = int(len(gen_rank2idx) * split)

    # print(f'{len(generic_rank)=}')
    # quit()

    # middle_rois = (generic_rank < high_cutoff) & (generic_rank > low_cutoff)
    # generic_rank[middle_rois] = np.nan

    # inc_rois = generic_rank > high_cutoff
    # generic_rank[inc_rois] = np.nan
    # con_rois = generic_rank < low_cutoff
    # generic_rank[con_rois] = np.nan

    # # print(f'{mid=}')
    #
    # generic_rank[:mid] = 0
    # generic_rank[-mid:] = 1
    # print('test')
    # plot_ranks(gen_idx2rank, atlas, nroi=nrois, do_conn=do_conn)

    # # plot_ranks(gen_idx2rank, atlas, nroi=nrois, do_conn=do_conn)
    # quit()


    if verbose: print('Onto looping')
    for sn_i in tqdm(range(sn_roi_eff.shape[0]), position=0, leave=True):
        if do_generic:
            rank2idx = gen_rank2idx
        else:
            sn_effs = sn_roi_eff[sn_i, :]
            rank2idx = sn_effs.argsort()
        idx2rank = np.argsort(rank2idx)

        # print(f'{ef_rank.shape=}')
        # quit()

        # inc_rois = np.argwhere(idx2rank < low_cutoff)[:, 0]
        # print(f'{inc_rois=}')
        # con_rois = rank2idx > high_cutoff

        inc_rois = rank2idx[:low_cutoff]
        con_rois = rank2idx[high_cutoff:]

        # inc_rois = rank2idx[:mid]
        # con_rois = rank2idx[-mid:]
        # print(sn_roi_rs.shape)
        # print(f'{inc_rois=}')
        # # print(f'{len(inc_rois)=}')
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

        corr, _ = stats.pearsonr(rs_inc, rs_con)
        corrs.append(corr)

        # print(np.sum(np.isnan(rs_con)))
        # rs_con = stdize(rs_con, nans=True)
    corrs = np.array(corrs)
    M_corr = np.mean(corrs)
    if shuffle:
        print(f'Shuffle: {M_corr=:.3f}')
    else:
        print(f'Genuine ({split}): {M_corr=:.3f}')
    return corrs

def plot_by_split(do_conn=True):
    corrs = link_activity(do_generic=True, do_conn=do_conn)
    quit()

    for split in [.3]: # [.5, .4, .3, .2, .1]
        corrs = link_activity(do_generic=False, do_conn=do_conn,
                              split=split, combine_regions=False)

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

