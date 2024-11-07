import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

from pathlib import Path

import pickle

from Study1B.analyze_plot_Fig3 import get_sn_roi_ar, make_conn, get_combo
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


@cache
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
    print(f'Make compressed time: {time() - t_st:.5f} s')
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


def get_HCP_task_conn(combine_regions):
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'combo':
        conn_highs, conn_lows, sns = get_combo(kw)
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    sn2conns = {}
    for sn, conn_high, conn_low in zip(sns, conn_highs, conn_lows):
        sn2conns[sn] = conn_high, conn_low
    return sn2conns


def get_dd_etc(conn, p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
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


def get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                 combine_regions=False):
    kw = {'combine_regions': combine_regions, }
    sn2conns = pickle_wrap(get_HCP_task_conn, kwargs=kw, easy_override=True,
                           RAM_cache=True)

    assert set(sns) - set(sn2conns) == set(), f'{set(sns) - set(sn2conns)=}'
    conn_high = [sn2conns[sn][0] for sn in sns]
    conn_high = np.array(conn_high)
    itr_h, dd_h, vv_h, dv_ant_h, dv_pos_h, M_overall_h = (
        get_dd_etc(conn_high, p_d_ant, p_d_pos, p_v_ant, p_v_pos))

    conn_low = [sn2conns[sn][1] for sn in sns]
    conn_low = np.array(conn_low)
    itr_l, dd_l, vv_l, dv_ant_l, dv_pos_l, M_overall_l = (
        get_dd_etc(conn_low, p_d_ant, p_d_pos, p_v_ant, p_v_pos))
    return itr_l - itr_h


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
    print(f'C:\ LR RS: N = {len(fns_LR)}')
    fns_RL = [fn for fn in fns if 'RL_clean' in fn]
    print(f'C:\ RL RS: N = {len(fns_RL)}')
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
    print(rf'E:\ LR RS: N = {len(fns_LR_E)}')
    fns_RL_E = [fn for fn in fns_E if 'RL_clean' in fn]
    print(rf'E:\ RL RS: N = {len(fns_RL_E)}')
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


def run_analysis_Study2B(num_test=10_000, skip_other=True,
                         combine_regions=False):
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, combine_regions=combine_regions))

    p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all, p_no_all = (
        get_quads(skip_other=False, combine_regions=combine_regions))

    rs_efs_all = get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                lr='LR', combine_regions=combine_regions, bilateral=False,
                                )
    rs_efs_all_rl = get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                                   p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                   lr='RL', combine_regions=combine_regions, bilateral=False,
                                   )
    rs_efs_all = np.nanmean([rs_efs_all, rs_efs_all_rl], axis=0)

    task_efs_all = get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                                combine_regions=combine_regions, )
    task_efs_all = np.abs(task_efs_all)
    r, p = stats.spearmanr(rs_efs_all, task_efs_all, nan_policy='omit')
    print(f'Overall: {r=:.4f}, {p=:.2f}')

    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'Total number of ROI sets: {num_pos}')

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

        rs_efs = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                lr='LR', combine_regions=combine_regions, bilateral=False, )
        rs_efs2 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                 lr='RL', combine_regions=combine_regions, bilateral=False, )
        rs_efs = np.nanmean([rs_efs, rs_efs2], axis=0)
        print(f'Resting time: {time() - t_st:.5f} s')
        # combined regions within-subject effects depend on reg_global?

        t_st = time()
        task_efs = get_HCP_task(sns, pda_i, pdp_i, pva_i, pvp_i,
                                combine_regions=combine_regions)

        print(f'\tTask time: {time() - t_st:.5f} s')
        num_nans = np.sum(np.isnan(rs_efs))
        assert num_nans == 0, f'{num_nans=}, f{rs_efs.shape=}'

        task_efs_l.append(task_efs)
        rs_efs_l.append(rs_efs)

        if len(task_efs_l) % 10 == 0:
            ttest_on_correlations(task_efs_l, rs_efs_l)

    str_shape = '_' + str(np.array(task_efs_l).shape)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_task.pkl'
    Path(fp_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(fp_pkl, 'wb') as f:
        pickle.dump(task_efs_l, f)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_rs.pkl'
    with open(fp_pkl, 'wb') as f:
        pickle.dump(rs_efs_l, f)

    ttest_on_correlations(task_efs_l, rs_efs_l)


def ttest_on_correlations(task_efs_l, rs_efs_l):
    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(np.array(task_efs_l).T,
                                               np.array(rs_efs_l).T)):
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


@cache
def get_quads(skip_other=False, combine_regions=False, p_no_override=False):
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


if __name__ == '__main__':
    run_analysis_Study2B()
