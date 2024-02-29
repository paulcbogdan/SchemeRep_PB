import os

from analyze_rs import load_act_conn

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from utils import pickle_wrap, stdize
from vendor_lmers import get_dfs_conn_trials
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def shuffle_rows(sn_roi_act):
    new_ar = np.full(sn_roi_act.shape, np.nan)
    sn_roi_act = np.nanmean(sn_roi_act, axis=1)
    cnt = 0
    for i in range(sn_roi_act.shape[-1]):
        new_ar[:, cnt, i] = sn_roi_act[:, i]
        cnt += 1
        if cnt % 3 == 0:
            cnt = 0
    return new_ar

def link_activity(do_generic=True):
    # Identify a person-specific list of ROIs
    # Identify a group-level list of ROIs
    # See if the fluctuation in resting-state is stronger when modeled
    #   person-specific than group-level

    sn_roi_act, _, df_sns_l, sns  = \
        pickle_wrap(get_dfs_conn_trials, kwargs={'fp': 'obj7_fMRI',
                                                 'single': False,
                                                 'squeeze': False},
                    easy_override=False)
    sns = [df['sn'].iloc[0] for df in df_sns_l]
    good_i = [i for i, sn in enumerate(sns) if sn != '133']
    sn_roi_act = sn_roi_act[good_i, :, :]
    sns = [sn for sn in sns if sn != '133'] # bad rs

    sn_roi_rs, sns_, _ = load_act_conn(False,
                                                easy_override=False)
    assert sns == sns_

    corrs = []
    # sn_roi_act = shuffle_rows(sn_roi_act)
    sn_roi_M = np.nanmean(sn_roi_act, axis=-1)
    sn_roi_eff = sn_roi_M[:, 2, :] - sn_roi_M[:, 0, :]
    # print(sn_roi_eff)
    # quit()

    generic_eff = np.nanmean(sn_roi_eff, axis=0)
    generic_rank = generic_eff.argsort()

    for sn_i in range(sn_roi_eff.shape[0]):
        if do_generic:
            ef_rank = generic_rank
        else:
            sn_effs = sn_roi_eff[sn_i, :]
            ef_rank = sn_effs.argsort()
        mid = ef_rank.size // 2
        inc_rois = ef_rank[mid:]
        con_rois = ef_rank[:mid]

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
    print(f'{M_corr=:.3f}')
    return corrs


if __name__ == '__main__':
    # a = {1:1, 2:1}
    # b = {1:1, 2:1}
    # print(a == b)
    # quit()

    # SANITY TEST WHICH RANDOMIZES BY CONDITION,
    #   EVEN THOUGH WE PERSONALLY DO NOT SEE AN A PRIORI REASON

    corrs_gen = link_activity(do_generic=True)
    corrs_ss = link_activity(do_generic=False)
    # corrs_ss[corrs_ss > 1.0] = np.nan
    print(len(corrs_gen))
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

