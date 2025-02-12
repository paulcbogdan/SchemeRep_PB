import os
import pathlib

import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
#
# from Study2A.rs_connectivity_funcs import get_hemi_ps
# from marinate.pkld import pkld

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

from pathlib import Path

import pickle

from Study1B.analyze_plot_Fig3 import get_sn_roi_ar, make_conn, get_combo, get_wl_contrast_conn
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions

from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
import scipy.stats as stats
from functools import cache
from time import time
from tqdm import tqdm
import itertools

# suppress RuntimeWarning
from warnings import simplefilter

simplefilter("ignore", category=RuntimeWarning)


def get_sn_roi_ar_std(**kw):
    ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, RAM_cache=False,
                     verbose=-1)
    assert len(ar.shape) == 2
    ar = stats.zscore(ar, axis=1)
    return ar


# @cache
def get_sns_roi_ar_std(sns, **kw):
    ars = []
    for sn in sns:
        ar = get_sn_roi_ar_std(sn=sn, **kw)
        if ar.shape[1] < 1200:
            ar = np.pad(ar, ((0, 0), (0, 1200 - ar.shape[1])), 'constant',
                        constant_values=np.nan)
        assert ar.shape[1] == 1200, f'{sn=}, {ar.shape=}'
        ars.append(ar)
    return np.array(ars)


@cache
def get_rs_conn_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos, **kw):
    if kw['lr'] == 'LR_RL':
        kw_lr = kw.copy()
        kw_lr['lr'] = 'LR'
        ar = get_sns_roi_ar_std(tuple(sns), **kw_lr)
        kw_rl = kw.copy()
        kw_rl['lr'] = 'RL'
        ar2 = get_sns_roi_ar_std(tuple(sns), **kw_rl)
        print(ar.shape)
        ar = np.concatenate([ar, ar2], axis=-1)
    else:
        ar = get_sns_roi_ar_std(tuple(sns), **kw)

    p_all = list(p_d_ant) + list(p_d_pos) + list(p_v_ant) + list(p_v_pos)
    p_d_ant_new = np.arange(len(p_d_ant))
    p_d_pos_new = np.arange(len(p_d_pos)) + len(p_d_ant)
    p_v_ant_new = np.arange(len(p_v_ant)) + len(p_d_ant) + len(p_d_pos)
    p_v_pos_new = np.arange(len(p_v_pos)) + len(p_d_ant) + len(p_d_pos) + len(p_v_ant)

    map2new = {}
    for i, p in enumerate(p_all):
        map2new[p] = i

    rs_conn_compressed = np.full((len(sns), len(p_all),
                                  len(p_all), ar.shape[2]), np.nan)
    rs_conn_compressed[:, *np.ix_(p_d_ant_new, p_d_pos_new), :] = (
            ar[:, p_d_ant, None, :] * ar[:, None, p_d_pos, :])
    rs_conn_compressed[:, *np.ix_(p_v_ant_new, p_v_pos_new), :] = (
            ar[:, p_v_ant, None, :] * ar[:, None, p_v_pos, :])
    rs_conn_compressed[:, *np.ix_(p_d_ant_new, p_v_ant_new), :] = (
            ar[:, p_d_ant, None, :] * ar[:, None, p_v_ant, :])
    rs_conn_compressed[:, *np.ix_(p_d_pos_new, p_v_pos_new), :] = (
            ar[:, p_d_pos, None, :] * ar[:, None, p_v_pos, :])
    # rs_conn_compressed

    return rs_conn_compressed, map2new


def uncompress_conn(conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                    n_roi):
    p_all = list(p_d_ant) + list(p_d_pos) + list(p_v_ant) + list(p_v_pos)
    conn_uncompressed = np.full((conn.shape[0], n_roi, n_roi, conn.shape[3]), np.nan)
    conn_uncompressed[:, *np.ix_(p_all, p_all), :] = conn
    return conn_uncompressed


def get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                   all_pda, all_pdp, all_pva, all_pvp, lr='LR',
                   combine_regions=False, bilateral=False,
                   reg_global=False, no_compcor=False,
                   ix=True):
    kw = {'lr': lr, 'combine_regions': combine_regions,
          'bilateral': bilateral, 'reg_global': reg_global,
          'no_compcor': no_compcor, 'rs': True}
    t_st = time()
    rs_conn, map2new = get_rs_conn_sns(tuple(sns),
                                       tuple(all_pda), tuple(all_pdp),
                                       tuple(all_pva), tuple(all_pvp), **kw)
    print(f'Make resting-state compressed time: {time() - t_st:.5f} s')
    p_d_ant = [map2new[p] for p in p_d_ant]
    p_d_pos = [map2new[p] for p in p_d_pos]
    p_v_ant = [map2new[p] for p in p_v_ant]
    p_v_pos = [map2new[p] for p in p_v_pos]

    if ix:
        dd = rs_conn[:, *np.ix_(p_d_ant, p_d_pos), :]
        dd = np.nanmean(dd, axis=(1, 2))
    else:
        dd = rs_conn[:, p_d_ant, p_d_pos, :]
        dd = np.nanmean(dd, axis=1)
    dd_ = stats.zscore(dd, axis=1)
    if ix:
        vv = rs_conn[:, *np.ix_(p_v_ant, p_v_pos), :]
        vv = np.nanmean(vv, axis=(1, 2))
    else:
        vv = rs_conn[:, p_v_ant, p_v_pos, :]
        vv = np.nanmean(vv, axis=1)

    vv_ = stats.zscore(vv, axis=1)
    dd_vv = dd_ + vv_
    if ix:
        dv_ant = rs_conn[:, *np.ix_(p_d_ant, p_v_ant), :]
        dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    else:
        dv_ant = rs_conn[:, p_d_ant, p_v_ant, :]
        dv_ant = np.nanmean(dv_ant, axis=1)

    dv_ant_ = stats.zscore(dv_ant, axis=1)
    if ix:
        dv_pos = rs_conn[:, *np.ix_(p_d_pos, p_v_pos), :]
        dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    else:
        dv_pos = rs_conn[:, p_d_pos, p_v_pos, :]
        dv_pos = np.nanmean(dv_pos, axis=1)

    dv_pos_ = stats.zscore(dv_pos, axis=1)
    dv_dv = dv_ant_ + dv_pos_

    ef = np.nanmean(np.abs(dd_vv - dv_dv), axis=1)
    return ef


def get_HCP_task_conn(combine_regions, focus='combo'):
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': focus, 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'wl':
        conn_highs, conn_lows, sns, conn_highs0, conn_highs1, conn_lows0, conn_lows1 = (
            get_wl_contrast_conn(kw, get6=True))
        # conn_highs, conn_lows, sns = (
        #     get_wl_contrast_conn(kw, easy_override=False))
    elif kw['only'] == 'combo':
        # conn_highs, conn_lows, sns = (
        #     get_wl_contrast_conn(kw, easy_override=False))
        conn_highs, conn_lows, sns, conn_highs0, conn_highs1, conn_lows0, conn_lows1 = (
            get_combo(kw, get6=True))
    else:
        kw['get6'] = True
        conn_highs, conn_lows, sns, conn_highs0, conn_highs1, conn_lows0, conn_lows1 = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    sn2conns = {}
    for sn, conn_high, conn_low in zip(sns, conn_highs, conn_lows):
        sn2conns[sn] = conn_high, conn_low
    sn2conns_lr = {}
    for sn, conn_high, conn_low in zip(sns, conn_highs0, conn_lows0):
        sn2conns_lr[sn] = conn_high, conn_low
    sn2conns_rl = {}
    for sn, conn_high, conn_low in zip(sns, conn_highs1, conn_lows1):
        sn2conns_rl[sn] = conn_high, conn_low
    return sn2conns, sn2conns_lr, sn2conns_rl


def get_dd_etc(conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    # print(f'{conn.shape=}')
    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    # print(conn)
    # print(p_d_pos)
    # quit()
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))
    itr = dd + vv - dv_ant - dv_pos
    return itr, dd, vv, dv_ant, dv_pos, M_overall


def get_itr_ef_from_sn2conns(sns, sn2conns, p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    conn_high = [sn2conns[sn][0] for sn in sns]
    conn_high = np.array(conn_high)
    itr_h, dd_h, vv_h, dv_ant_h, dv_pos_h, M_overall_h = (
        get_dd_etc(conn_high, p_d_ant, p_d_pos, p_v_ant, p_v_pos))
    # print(itr_h)
    # print(conn_high[:, 2, 4])
    # quit()

    conn_low = [sn2conns[sn][1] for sn in sns]
    conn_low = np.array(conn_low)
    itr_l, dd_l, vv_l, dv_ant_l, dv_pos_l, M_overall_l = (
        get_dd_etc(conn_low, p_d_ant, p_d_pos, p_v_ant, p_v_pos))
    # return dd_h
    return itr_l - itr_h


def get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                 combine_regions=False, focus='combo'):
    kw = {'combine_regions': combine_regions, 'focus': focus}
    sn2conns, sn2conns_lr, sn2conns_rl = (
        pickle_wrap(get_HCP_task_conn, kwargs=kw, easy_override=False,
                    RAM_cache=True))

    # print(sn2conns[sns[-1]][0].shape)
    # print(sn2conns[sns[-1]][0])
    # quit()

    assert set(sns) - set(sn2conns) == set(), f'{set(sns) - set(sn2conns)=}'
    itr_ef = get_itr_ef_from_sn2conns(sns, sn2conns, p_d_ant, p_d_pos,
                                      p_v_ant, p_v_pos)
    # print(itr_ef)
    # quit()
    # itr_ef = None
    itr_ef_lr = get_itr_ef_from_sn2conns(sns, sn2conns_lr, p_d_ant, p_d_pos,
                                         p_v_ant, p_v_pos)
    del sn2conns_lr
    itr_ef_rl = get_itr_ef_from_sn2conns(sns, sn2conns_rl, p_d_ant, p_d_pos,
                                         p_v_ant, p_v_pos)
    del sn2conns_rl
    # print(itr_ef)
    # quit()
    # plt.scatter(itr_ef_lr, itr_ef_rl)
    # plt.show()
    return itr_ef, itr_ef_lr, itr_ef_rl


def find_overlapping_sns(reg_global_task=True, no_compcor_task=True,
                         reg_global=False, no_compcor=False):
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')

    fns_task = os.listdir(r'HCP_gambling/LSA')
    if reg_global_task:
        fns_task = [fn for fn in fns_task if 'global' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'global' not in fn]
    if no_compcor_task:
        fns_task = [fn for fn in fns_task if 'nocc' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'nocc' not in fn]
    fns_task_LR = [fn for fn in fns_task if 'LR' in fn]
    fns_task_RL = [fn for fn in fns_task if 'RL' in fn]
    sns_task_LR = {fn.split('_')[0] for fn in fns_task_LR}
    print(f'LR task: N = {len(sns_task_LR)}')
    sns_task_RL = {fn.split('_')[0] for fn in fns_task_RL}
    print(f'RL task: N = {len(sns_task_RL)}')
    sns_task = sns_task_RL.intersection(sns_task_LR)

    fns = os.listdir(r'C:\HCP_RS_clean')
    if reg_global:
        fns = [fn for fn in fns if 'global' in fn]
    else:
        fns = [fn for fn in fns if 'global' not in fn]
    if no_compcor:
        fns = [fn for fn in fns if 'nocc' in fn]
    else:
        fns = [fn for fn in fns if 'nocc' not in fn]
    fns_LR = [fn for fn in fns if 'LR_clean' in fn]
    print(f'C drive: LR RS: N = {len(fns_LR)}')
    fns_RL = [fn for fn in fns if 'RL_clean' in fn]
    print(f'C drive: RL RS: N = {len(fns_RL)}')
    sns_rs_LR = {fn.split('_')[0] for fn in fns_LR}
    sns_rs_RL = {fn.split('_')[0] for fn in fns_RL}
    sns_rs = sns_rs_LR.intersection(sns_rs_RL)

    fns_E = os.listdir(r'E:\HCP_RS_clean')
    if reg_global:
        fns_E = [fn for fn in fns_E if 'global' in fn]
    else:
        fns_E = [fn for fn in fns_E if 'global' not in fn]
    if no_compcor:
        fns_E = [fn for fn in fns_E if 'nocc' in fn]
    else:
        fns_E = [fn for fn in fns_E if 'nocc' not in fn]
    fns_LR_E = [fn for fn in fns_E if 'LR_clean' in fn]
    print(rf'E drive: LR RS: N = {len(fns_LR_E)}')
    fns_RL_E = [fn for fn in fns_E if 'RL_clean' in fn]
    print(rf'E drive: RL RS: N = {len(fns_RL_E)}')
    sns_rs_LR_E = {fn.split('_')[0] for fn in fns_LR_E}
    sns_rs_RL_E = {fn.split('_')[0] for fn in fns_RL_E}
    sns_rs_E = sns_rs_LR_E.intersection(sns_rs_RL_E)

    sns_rs.update(sns_rs_E)
    print(f'Overall RS: N = {len(sns_rs)}')

    sns_overlap = sns_task.intersection(sns_rs)
    bad_sns = {'263436', }  # missing task file (at least on the computer running this)

    sns_overlap = sorted(list(sns_overlap - bad_sns))

    print(f'Number of overlapping non-bad sns: {len(sns_overlap)}')
    return sns_overlap


def get_final_HCP_sns():
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')
    sns = sns[::-1]
    return sns


def run_analysis_Study2B(num_test=10_000, skip_other=True,
                         combine_regions=False, focus='combo'):
    sns = get_final_HCP_sns()
    # sns = sns[:400]
    sns = sns[::-1]
    # print(sns)

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, combine_regions=combine_regions))

    p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all, p_no_all = (
        get_quads(skip_other=False, combine_regions=combine_regions))

    task_effs_all, task_efs_all_lr, task_efs_all_rl = (
        get_HCP_task(sns, p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                     combine_regions=combine_regions, focus=focus))
    # print(task_effs_all)
    t, p = stats.ttest_1samp(task_effs_all, 0)
    print(f'Overall task effect: {t=:.3f}, {p=:.3f}')
    t_lr, p_lr = stats.ttest_1samp(task_efs_all_lr, 0)
    print(f'Overall task effect (LR): {t_lr=:.3f}, {p_lr=:.3f}')
    t_rl, p_rl = stats.ttest_1samp(task_efs_all_rl, 0)
    print(f'Overall task effect (RL): {t_rl=:.3f}, {p_rl=:.3f}')
    # task_efs_all = np.abs(task_efs_all)
    # r, p = stats.spearmanr(rs_efs_all, task_efs_all, nan_policy='omit')
    task_reliability, _ = stats.spearmanr(task_efs_all_lr, task_efs_all_rl,
                                          nan_policy='omit')
    print(f'Overall task reliability: {task_reliability=:.3f}')

    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'Total number of ROI sets: {num_pos}')
    quit()

    np.random.seed(0)
    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    combos = list(combos)
    np.random.shuffle(combos)

    combos_ = []
    for (a, b, c, d) in combos:
        if len(combos_) >= num_test:
            continue
        if len({a, b, c, d}) < 4:
            continue
        combos_.append((a, b, c, d))
    combos = combos_

    rs_efs_l = []
    task_efs_l = []
    reliability_l = []
    reliability_task_l = []
    task_efs_l_lr = []
    task_efs_l_rl = []
    for (a, b, c, d) in tqdm(combos):
        if skip_other:
            pda_i = [a, a + 1]
            pdp_i = [b, b + 1]
            pva_i = [c, c + 1]
            pvp_i = [d, d + 1]
        else:
            pda_i = [a]
            pdp_i = [b]
            pva_i = [c]
            pvp_i = [d]

        t_st = time()
        # print(pda_i)
        # quit()

        rs_efs1 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                 lr='LR', combine_regions=combine_regions, bilateral=False, )
        rs_efs2 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                 lr='RL', combine_regions=combine_regions, bilateral=False, )
        rs_efs = np.nanmean([rs_efs1, rs_efs2], axis=0)
        print(f'Resting time: {time() - t_st:.5f} s')

        # reliability, _ = stats.spearmanr(rs_efs1, rs_efs2, nan_policy='omit')
        # reliability_l.append(reliability)
        # if len(reliability_l) > 1:
        #     M_reliability = np.nanmean(reliability_l)
        #     SD_reliability = np.nanstd(reliability_l)
        #     print(f'rs reliability: {reliability=:.4f} ({a}, {b}, {c}, {d}) | '
        #           f'{M_reliability=:.3f} [{SD_reliability:.3f}]')
        # continue

        # print(f'Resting time: {time() - t_st:.5f} s')
        # combined regions within-subject effects depend on reg_global?

        t_st = time()
        task_efs, task_efs_lr, task_efs_rl = (
            get_HCP_task(sns, pda_i, pdp_i, pva_i, pvp_i,
                         combine_regions=combine_regions,
                         focus=focus))
        print(f'\tTask time ({focus}): {time() - t_st:.5f} s')
        # print(f'{task_efs=}')
        # print(task_efs_lr)
        # print(task_efs_rl)
        # quit()

        task_reliability, _ = stats.spearmanr(task_efs_lr, task_efs_rl,
                                              nan_policy='omit')
        reliability_task_l.append(task_reliability)
        if len(reliability_task_l) > 1:
            M_reliability = np.nanmean(reliability_task_l)
            SD_reliability = np.nanstd(reliability_task_l)
            SE_reliability = SD_reliability / np.sqrt(len(reliability_task_l))
            print(f'task reliability: {task_reliability=:.4f} ({a}, {b}, {c}, {d}) | '
                  f'{M_reliability=:.3f} [{SD_reliability:.3f}, {SE_reliability:.3f}]')
        # continue
        num_nans = np.sum(np.isnan(rs_efs))
        assert num_nans == 0, f'{num_nans=}, f{rs_efs.shape=}'

        task_efs_l.append(task_efs)
        rs_efs_l.append(rs_efs)
        task_efs_l_lr.append(task_efs_lr)
        task_efs_l_rl.append(task_efs_rl)

        if len(task_efs_l) % 5 == 0:
            # print(f'{len(task_efs_l)=}, {len(rs_efs_l)=}')
            ttest_on_correlations(task_efs_l, rs_efs_l)
            task_efs_l_lr_ = np.array(task_efs_l_lr)
            task_efs_l_rl_ = np.array(task_efs_l_rl)

            within_subj_r = []
            for sn_j in range(task_efs_l_lr_.shape[1]):
                r, p = stats.spearmanr(task_efs_l_lr_[:, sn_j],
                                       task_efs_l_rl_[:, sn_j])
                within_subj_r.append(r)
            print(f'{len(within_subj_r)=}')
            M_within_reilability = np.nanmean(within_subj_r)
            SE_within = np.nanstd(within_subj_r) / np.sqrt(len(within_subj_r))
            print(f'Within-subject task reliability: {M_within_reilability=:.3f} [{SE_within:.3f}]')
            # print(f'{M_within_reilability=}')

            task_efs_l_lr_ -= np.nanmean(task_efs_l_lr_, axis=1, keepdims=True)
            task_efs_l_rl_ -= np.nanmean(task_efs_l_rl_, axis=1, keepdims=True)
            within_subj_r = []
            for sn_j in range(task_efs_l_lr_.shape[1]):
                r, p = stats.spearmanr(task_efs_l_lr_[:, sn_j],
                                       task_efs_l_rl_[:, sn_j])
                within_subj_r.append(r)
            M_within_reilability = np.nanmean(within_subj_r)
            SE_within = np.nanstd(within_subj_r) / np.sqrt(len(within_subj_r))
            print(f'ROI-regr, within-subject task reliability: {M_within_reilability=:.3f} [{SE_within:.3f}]')
            # print(f'{M_within_reilability=}')

    # quit()
    str_shape = '_' + str(np.array(task_efs_l).shape)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_task_{focus}.pkl'
    Path(fp_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(fp_pkl, 'wb') as f:
        pickle.dump(task_efs_l, f)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_rs_{focus}.pkl'
    with open(fp_pkl, 'wb') as f:
        pickle.dump(rs_efs_l, f)

    ttest_on_correlations(task_efs_l, rs_efs_l)


def ttest_on_correlations(task_efs_l, rs_efs_l):
    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(np.array(task_efs_l).T,
                                               np.array(rs_efs_l).T)):
        # print(f'{task_efs.shape=}')
        # print(f'{rs_efs.shape=}')
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)

    N = len(t_l)
    M_r = np.nanmean(t_l)
    t_within, p_within = stats.ttest_1samp(t_l, 0)

    t_l = []

    task_efs_l_ = np.array(task_efs_l)
    for i, (task_efs, rs_efs) in enumerate(zip(task_efs_l_, rs_efs_l)):
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)

    print(f'Within-subj: Mean r = {M_r:.3f}, '
          f't[{N - 1}] = {t_within:.2f}, p={p_within:.3f}')

    N = len(t_l)
    M_r = np.nanmean(t_l)
    t, p = stats.ttest_1samp(t_l, 0)
    print(f'Across-subject: Mean r = {M_r:.3f}, '
          f't[{N - 1}] = {t:.2f}, {p=:.3f}')


# get_quads_schaefer()

@cache
def get_quads(skip_other=False, combine_regions=False, p_no_override=False, ):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                             combine_regions=combine_regions)

    n_roi = 54 if combine_regions else 246
    p_no = [i for i in range(n_roi) if i not in p_d_ant + p_d_pos +
            p_v_ant + p_v_pos]

    if skip_other:
        p_d_ant = p_d_ant[::2]
        p_d_pos = p_d_pos[::2]
        p_v_ant = p_v_ant[::2]
        p_v_pos = p_v_pos[::2]

    if p_no_override:
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                                 anat_ver=3, combine_regions=combine_regions)
        p_no = [i for i in range(n_roi) if i not in p_d_ant_ + p_d_pos_ +
                p_v_ant_ + p_v_pos_]

    return p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no


def load_rs_HCP_BOLD(schaefer=False, combine_regions=False,
                     lr='LR'):
    combine_regions_ = combine_regions
    if schaefer:
        assert not schaefer == True
        combine_regions = (combine_regions, ('schaefer', 400))
    else:
        combine_regions = combine_regions
    kw = {'lr': lr, 'combine_regions': combine_regions,
          'bilateral': False, 'reg_global': False,
          'no_compcor': False, 'rs': True}
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')
    if combine_regions_:
        sns = tuple(sns)
        sn_roi_act = get_sns_roi_ar_std(tuple(sns), **kw)
        sn_roi_act = sn_roi_act[:, :, :120]
        # sn_roi_act = sn_roi_act[:, :, ::4]
    else:
        sns = tuple(sns[::5])
        sn_roi_act = get_sns_roi_ar_std(tuple(sns), **kw)
        sn_roi_act = sn_roi_act[:, :, :40]
    # print(sn_roi_act.shape)
    # quit()
    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]
    return sn_roi_act, sns, conn_trials


# get_HCP_rs_BOLD()

if __name__ == '__main__':
    run_analysis_Study2B(focus='combo')
    # run_analysis_Study2B(focus='loss')
    # run_analysis_Study2B(focus='wl')
