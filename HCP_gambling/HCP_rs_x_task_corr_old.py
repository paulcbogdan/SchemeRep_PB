import os

from Study1B.analyze_plot_Fig3 import get_sn_roi_ar, make_conn, get_combo
from Study2B.analyze_Study2B import get_quads

from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy import stats
from functools import cache
from tqdm import tqdm
import itertools

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

os.chdir(r'C:\PycharmProjects\SchemeRep')

@cache
def get_rs_conn(**kw):
    ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, RAM_cache=True)

    assert len(ar.shape) == 2
    ar = stats.zscore(ar, axis=1)
    rs_conn = ar[:, None, :] * ar[None, :, :]
    return rs_conn

def get_HCP_rs(sn, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
               lr='LR',
               combine_regions=False, bilateral=False,
               reg_global=True, no_compcor=True, anat_ver=4):
    # atlas = get_atlas(combine_regions=combine_regions,
    #                   combine_bilateral=bilateral,
    #                   HCP=True)
    kw = {'sn': sn, 'lr': lr, 'combine_regions': combine_regions,
          'bilateral': bilateral,
          'reg_global': reg_global, 'no_compcor': no_compcor,
          'rs': True}

    rs_conn = get_rs_conn(**kw)

    # p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
    #     get_vendor_partitions(age='healthy', anat=True, weighted=False,
    #                           flip=True, thr=.9, scrub=False, anat_ver=anat_ver,
    #                           combine_regions=combine_regions)

    dd = rs_conn[*np.ix_(p_d_ant, p_d_pos), :]
    dd = np.nanmean(dd, axis=(0, 1))
    dd_ = stats.zscore(dd)
    vv = rs_conn[*np.ix_(p_v_ant, p_v_pos), :]
    vv = np.nanmean(vv, axis=(0, 1))
    vv_ = stats.zscore(vv)
    dd_vv = dd_ + vv_
    dv_ant = rs_conn[*np.ix_(p_d_ant, p_v_ant), :]
    dv_ant = np.nanmean(dv_ant, axis=(0, 1))
    dv_ant_ = stats.zscore(dv_ant)
    dv_pos = rs_conn[*np.ix_(p_d_pos, p_v_pos), :]
    dv_pos = np.nanmean(dv_pos, axis=(0, 1))
    dv_pos_ = stats.zscore(dv_pos)
    dv_dv = dv_ant_ + dv_pos_

    return dd_vv, dv_dv, dd, vv, dv_ant, dv_pos

def get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                   lr='LR', combine_regions=False, bilateral=False,
                   reg_global=True, no_compcor=True, anat_ver=4):
    dd_vv_l, dv_dv_l, dd_l, vv_l, dv_ant_l, dv_pos_l = [], [], [], [], [], []
    efs = []
    for sn in sns:
        dd_vv, dv_dv, dd, vv, dv_ant, dv_pos = get_HCP_rs(
            sn, p_d_ant, p_d_pos, p_v_ant, p_v_pos, lr=lr,
            combine_regions=combine_regions, bilateral=bilateral,
            reg_global=reg_global, no_compcor=no_compcor, anat_ver=anat_ver)
        ef = np.nanmean(np.abs(dd_vv - dv_dv))
        efs.append(ef)


    ef = np.array(efs)
    return ef
    # return dd_vv_l, dv_dv_l, dd_l, vv_l, dv_ant_l, dv_pos_l

def get_HCP_task_conn(sns, combine_regions, bilateral):
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'neut_as_PE': None, 'drop_neut': False, 'only': None,
          'num_sns': 1000, 'cont_PE': 0.3, 'cont_PE_by_event': True,
          'regr_M': True, 'lr_separate': False,
          'reg_global': True, 'no_compcor': True,
          'sns_set': list(sns)}

    # kw = {'combine_regions': True, 'bilateral': False, 'neut_as_PE': None, 'drop_neut': False, 'only': None, 'num_sns': 1000, 'cont_PE': 0.3, 'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True, 'reg_global': True, 'no_compcor': True}
    # print(kw)
    # kw2 = {'combine_regions': True, 'bilateral': False, 'neut_as_PE': None, 'drop_neut': False, 'only': None, 'num_sns': 1000, 'cont_PE': 0.3, 'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True, 'reg_global': True, 'no_compcor': True}
    # # quit()
    # print(kw2)
    # quit()

    if kw['neut_as_PE']:
        kw['drop_neut'] = False
        kw['only'] = None
        kw['cont_PE'] = None
        kw['cont_PE_by_event'] = False

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
                 combine_regions=False, bilateral=False):
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'sns': sns}
    sn2conns = pickle_wrap(get_HCP_task_conn, kwargs=kw, easy_override=False,
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


def get_HCP_rs_efs():
    pass

def find_overlapping_sns():
    reg_global = True
    no_compcor = True
    fns_task = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    if reg_global:
        fns_task = [fn for fn in fns_task if 'global' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'global' not in fn]
    if no_compcor:
        fns_task = [fn for fn in fns_task if 'nocc' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'nocc' not in fn]
    sns_task = {fn.split('_')[0] for fn in fns_task}

    fns = os.listdir(r'E:\HCP_RS_clean')
    if reg_global:
        fns = [fn for fn in fns if 'global' in fn]
    else:
        fns = [fn for fn in fns if 'global' not in fn]
    if no_compcor:
        fns = [fn for fn in fns if 'nocc' in fn]
    else:
        fns = [fn for fn in fns if 'nocc' not in fn]
    fns = [fn for fn in fns if 'LR_clean' in fn]
    sns_rs = {fn.split('_')[0] for fn in fns}
    # sns_rs = sorted(list(sns_rs))

    reg_global = False
    no_compcor = False

    fns_task = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    if reg_global:
        fns_task = [fn for fn in fns_task if 'global' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'global' not in fn]
    if no_compcor:
        fns_task = [fn for fn in fns_task if 'nocc' in fn]
    else:
        fns_task = [fn for fn in fns_task if 'nocc' not in fn]
    sns_task2 = {fn.split('_')[0] for fn in fns_task}

    fns = os.listdir(r'E:\HCP_RS_clean')
    if reg_global:
        fns = [fn for fn in fns if 'global' in fn]
    else:
        fns = [fn for fn in fns if 'global' not in fn]
    if no_compcor:
        fns = [fn for fn in fns if 'nocc' in fn]
    else:
        fns = [fn for fn in fns if 'nocc' not in fn]
    fns = [fn for fn in fns if 'LR_clean' in fn]
    sns_rs2 = {fn.split('_')[0] for fn in fns}
    # sns_rs2 = sorted(list(sns_rs))
    sns_overlap = sns_task2.intersection(sns_rs2).intersection(
        sns_task).intersection(sns_rs)
    return sns_overlap

def do_analysis(num_test=1_000, ctrl_group=False,
                skip_other=False, all_roi=False, ix=True, anat_ver=3,
                combine_regions=True, n='7', std_d=True,
                shuffle_seed=None):

    sns = find_overlapping_sns()
    bad_sns = {'150423', '171734'}
    sns = sorted(list(sns - bad_sns))
    print(f'Found overlapping sns: {len(sns)=}')
    sns = sorted(sns)[:400]

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = get_quads(skip_other,
                                                         all_roi=all_roi,
                                                         anat_ver=anat_ver,
                                                         combine_regions=combine_regions)

    # rs_efs_all = get_HCP_rs_sns(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
    #                         lr='LR', combine_regions=combine_regions, bilateral=False,
    #                         reg_global=True, no_compcor=True, anat_ver=3)
    # task_efs_all = get_HCP_task(sns, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
    #                         combine_regions=combine_regions, bilateral=False)
    # r, p = stats.spearmanr(rs_efs_all, task_efs_all, nan_policy='omit')
    # print(f'Overall: {r=:.2f}, {p=:.2f}')


    p_no = tuple(p_no)
    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'{num_pos=}')

    np.random.seed(0)
    print('Itertools...')
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

    # print('Running...')

    task_x_rs = []
    efs = []
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

        rs_efs = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                   lr='LR', combine_regions=combine_regions, bilateral=False,
                   reg_global=True, no_compcor=True, anat_ver=3)
        # rs_efs -= rs_efs_all
        task_efs = get_HCP_task(sns, pda_i, pdp_i, pva_i, pvp_i,
                                combine_regions=combine_regions, bilateral=False)
        r, p = stats.spearmanr(rs_efs, task_efs)
        task_x_rs.append(r)
        if len(task_x_rs) > 1:
            t, p = stats.ttest_1samp(task_x_rs, 0)
            N = len(task_x_rs)
            print(f't[{N-1}] = {t:.2f}, {p=:.3f}')

        task_efs_l.append(task_efs)
        rs_efs_l.append(rs_efs)

        test_corrs(task_efs_l, rs_efs_l)

def test_corrs(task_efs_l, rs_efs_l):
    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(task_efs_l, rs_efs_l)):
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)
    N = len(t_l)
    t, p = stats.ttest_1samp(t_l, 0)
    print(f'Across-subject: t[{N - 1}] = {t:.2f}, {p=:.3f}')

    t_l = []
    for i, (task_efs, rs_efs) in enumerate(zip(np.array(task_efs_l).T,
                                               np.array(rs_efs_l).T)):
        r, p = stats.spearmanr(task_efs, rs_efs)
        t_l.append(r)
    N = len(t_l)
    t, p = stats.ttest_1samp(t_l, 0)
    print(f'Within-subj: t[{N - 1}] = {t:.2f}, {p=:.3f}')

if __name__ == '__main__':
    do_analysis()

