import os

from HCP_gambling.HCP_behavior import get_gambling_behavior_good

os.chdir(r'C:\PycharmProjects\SchemeRep')

import pandas as pd
from pingouin import partial_corr
from tqdm import tqdm

import numpy as np
import scipy.stats as stats
from datetime import datetime

from Study1B.preprocess_Study1B import get_df_PE
from Study1B.run_analysis_plot_Fig3 import get_sn_roi_ar, make_conn, get_vd_ef
from atlas_utils import get_atlas
from Study1A.partition_VD_PA import get_VD_PA_partitions
# from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap
# from vendor_partitioning import get_vendor_partitions, do_regression

def partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no,
                      ctrl=True):
    if ctrl:
        df = pd.DataFrame({'dd_vv': dd_vv, 'dv_dv': dv_dv,
                           'pd_no': pd_no, 'ad_no': ad_no,
                           'av_no': av_no, 'pv_no': pv_no})
        try:
            r = partial_corr(df, x='dd_vv', y='dv_dv',
                             covar=['pd_no', 'ad_no', 'av_no', 'pv_no'],)
            r = r['r'].values[0]
            return r
        except AssertionError: # NaN
            return np.nan
    else:
        r, p = stats.spearmanr(dd_vv, dv_dv)
        return r

def get_HCP_vendor(sn, lr='LR', combine_regions=False, bilateral=False,
                   reg_global=True, no_compcor=True, anat_ver=4):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral,
                      HCP=True)

    kw = {'sn': sn, 'lr': lr, 'combine_regions': combine_regions,
          'bilateral': bilateral,
          'reg_global': reg_global, 'no_compcor': no_compcor,
          'rs': True}
    if no_compcor:
        from datetime import datetime
        dt_max = datetime(2024, 9, 17, 17, 0, 0, 0)
    else:
        dt_max = None

    ar = pickle_wrap(get_sn_roi_ar, kwargs=kw, dt_max=dt_max)
    assert len(ar.shape) == 2
    ar = stats.zscore(ar, axis=1)
    rs_conn = ar[:, None, :] * ar[None, :, :]

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True, weighted=False,
                             do_PA=True, thr=.9, scrub=False, anat_ver=anat_ver,
                             combine_regions=combine_regions)

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

    # pd_no = np.nanmean(rs_conn[p_d_ant, p_no, :], axis=0)
    # ad_no = np.nanmean(rs_conn[p_d_pos, p_no, :], axis=0)
    # av_no = np.nanmean(rs_conn[p_v_ant, p_no, :], axis=0)
    # pv_no = np.nanmean(rs_conn[p_v_pos, p_no, :], axis=0)

    # r = partial_corr_fluc(dd_vv, dv_dv, pd_no, ad_no, av_no, pv_no)
    # return r


def analyze_HCP_rs(combine_regions=False, bilateral=False,
                   reg_global=False, no_compcor=False, anat_ver=3):

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
    sns_rs = sorted(list(sns_rs))



    Mvs = []
    DDs = []
    VVs = []
    dv_ants = []
    dv_poss = []
    p_changes = []
    PE_bhv_efs = []
    bad_sns = []
    # sns_rs = sns_rs[:-1]
    # sns_rs = sns_rs[5::6]
    # sns_rs = sns_rs[::-2]
    good_sns = []

    dt_max = datetime(2024, 9, 18, 17, 30, 0, 0)

    sns_rs = sns_rs[:460]
    print(f'Candidate sns: {len(sns_rs)}')
    for sn in tqdm(sns_rs, desc='rs-fMRI loading', position=0, leave=True):
        kw = {'sn': sn, 'combine_regions': combine_regions,
              'bilateral': bilateral, 'reg_global': reg_global,
              'no_compcor': no_compcor, 'anat_ver': anat_ver}
        try:
            kw['lr'] = 'RL'
            dd_vv_rl, dv_dv_rl, dd_rl, vv_rl, dv_ant_rl, dv_pos_rl = pickle_wrap(
                get_HCP_vendor, kwargs=kw, easy_override=False, dt_max=dt_max)
            kw['lr'] = 'LR'
            dd_vv, dv_dv, dd, vv, dv_ant, dv_pos = pickle_wrap(
                get_HCP_vendor, kwargs=kw, easy_override=False, dt_max=dt_max)

            Mv = np.nanmean(np.abs(dd_vv - dv_dv))
            Mv_rl = np.nanmean(np.abs(dd_vv_rl - dv_dv_rl))
            rs_dd = np.nanmean(dd + dd_rl) / 2
            rs_vv = np.nanmean(vv + vv_rl) / 2
            rs_dv_ant = np.nanmean(dv_ant + dv_ant_rl) / 2
            rs_dv_pos = np.nanmean(dv_pos + dv_pos_rl) / 2
            M_vendor = np.mean([Mv, Mv_rl])
        except EOFError:
            bad_sns.append(sn)
            print(f'{bad_sns=}')
            continue
        except Exception as e:
            bad_sns.append(sn)
            print(f'ERROR RESTING ({sn}): {e=}')
            print(f'{bad_sns=}')
            continue
        p_change, PE_bhv, high_PE, low_PE = (
            get_gambling_behavior_good(sn, ))
        if PE_bhv is None:
            bad_sns.append(sn)
            print(f'Bad behavior files: {sn}')
            continue

        #     p_change_LR, PE_bhv_LR = get_gambling_behavior(sn, 'LR')
        #     p_change_RL, PE_bhv_RL = get_gambling_behavior(sn, 'RL')
        #     p_change = np.mean([p_change_LR, p_change_RL])
        #     PE_bhv = np.mean([PE_bhv_LR, PE_bhv_RL])
        # except FileNotFoundError:
        #     continue
        # except TypeError as e:
        #     bad_sns.append(sn)
        #     print(f'Missing files: {sn}, {e=}')
        #     continue
        # except KeyError as e:
        #     bad_sns.append(sn)
        #     print(f'Missing responses so NaN responses to high/low_PE: {sn}, {e=}')
        #     continue
        # except Exception as e:
        #     print(f'ERROR TASK ({sn}): {e=}')
        #     print(f'{bad_sns=}')
        #     continue

        Mvs.append(M_vendor)
        DDs.append(rs_dd)
        VVs.append(rs_vv)
        dv_ants.append(rs_dv_ant)
        dv_poss.append(rs_dv_pos)
        p_changes.append(p_change)
        PE_bhv_efs.append(PE_bhv)
        good_sns.append(sn)

    M_PE_ef = np.mean(PE_bhv_efs)

    t_PE_bhv, p_PE_bhv = stats.ttest_1samp(PE_bhv_efs, 0)
    N = len(good_sns)
    print('-*-***-*-')
    print(f'PE behavior effect: t[{N - 1}] = {t_PE_bhv:.2f}, p = {p_PE_bhv:.2f}, '
          f'M ef = {M_PE_ef:.2f}')
    print('-*-***-*-')

    r, p = stats.spearmanr(Mvs, p_changes)
    print(f'PRE OVERLAP EXCLUDE | RS x change-freq: {r=:.2f}, {p=:.2f}')
    r, p = stats.spearmanr(Mvs, PE_bhv_efs)
    print(f'PRE OVERLAP EXCLUDE | RS x PE-bhv-response: {r=:.2f}, {p=:.2f}')
    sns_rs = good_sns
    task_ef, dd_ef, vv_ef, dv_ant_ef, dv_pos_ef, sns_task = (
        get_task_ef(sns_rs, combine_regions=combine_regions,
                    anat_ver=anat_ver))
    overlapping_sns = set(sns_task) & set(sns_rs)
    overlapping_sns = sorted(list(overlapping_sns))
    sns_only_in_rs = set(sns_rs) - set(sns_task)
    sns_only_in_task = set(sns_task) - set(sns_rs)
    print(f'Overlapping sns: {len(overlapping_sns)}\n'
          f'\tOnly in RS: {len(sns_only_in_rs)}\n'
          f'\tOnly in task: {len(sns_only_in_task)}')

    idx_rs = [sns_rs.index(sn) for sn in overlapping_sns]
    Mvs = [Mvs[i] for i in idx_rs]
    DDs = [DDs[i] for i in idx_rs]
    VVs = [VVs[i] for i in idx_rs]
    dv_ants = [dv_ants[i] for i in idx_rs]
    dv_poss = [dv_poss[i] for i in idx_rs]
    print(f'{len(Mvs)=}')
    idx_task = [sns_task.index(sn) for sn in overlapping_sns]
    task_ef = task_ef[idx_task]
    p_changes = [p_changes[i] for i in idx_task]
    PE_bhv_efs = [PE_bhv_efs[i] for i in idx_task]
    # print(f'{len(itrs)=}')
    # quit()

    # assert set(sns_itr) - set(sns) == set()
    # idxs = [sns_itr.index(sn) for sn in sns]
    # itrs = itrs[idxs]

    r, p = stats.spearmanr(Mvs, p_changes)
    print(f'RS x change-freq: {r=:.2f}, {p=:.2f}')
    r, p = stats.spearmanr(Mvs, PE_bhv_efs)
    print(f'RS x PE-bhv-response: {r=:.2f}, {p=:.2f}')
    # r, p = stats.spearmanr(Mvs, task_ef)
    # print(f'RS x task: {r=:.2f}, {p=:.2f}')
    # r, p = stats.spearmanr(p_changes, task_ef)
    # print(f'Change-freq x task: {r=:.2f}, {p=:.2f}')
    # r, p = stats.spearmanr(PE_bhv_efs, task_ef)
    # print(f'PE-bhv-response x task: {r=:.2f}, {p=:.2f}')

    df_out = pd.DataFrame({'rs_vendor': Mvs,
                           'rs_dd': DDs, 'rs_vv': VVs,
                           'rs_dv_ant': dv_ants, 'rs_dv_pos': dv_poss,
                           'sns': overlapping_sns, 'p_changes': p_changes,
                           'PE_bhv_efs': PE_bhv_efs})

    names = ['itr', 'dd', 'vv', 'dv_ant', 'dv_pos']
    task_vals = [task_ef, dd_ef, vv_ef, dv_ant_ef, dv_pos_ef]
    for name, vals in zip(names, task_vals):
        df_out[name] = vals
        r, p = stats.spearmanr(Mvs, vals)
        print(f' -*- {name} -*-')
        print(f'\tRS x task-{name}: {r=:.2f}, {p=:.2f}')
        r, p = stats.spearmanr(p_changes, vals)
        print(f'\tChange-freq x task-{name}: {r=:.2f}, {p=:.2f}')
        r, p = stats.spearmanr(PE_bhv_efs, vals)
        print(f'\tPE-bhv-response x task-{name}: {r=:.2f}, {p=:.2f}')

    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    fn_out = fr'fMRI_HCP_results_{anat_ver}{glob_str}{cc_str}.csv'
    fp_out = fr'C:\PycharmProjects\SchemeRep\{fn_out}'
    df_out.to_csv(fp_out, index=False)

def get_task_ef(sns_in, combine_regions=False, anat_ver=3):
    sns_in = sns_in[::-1]
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'neut_as_PE': None, 'drop_neut': True, 'only': None,
          'num_sns': None, 'cont_PE': 0.3, 'cont_PE_by_event': True,
          'regr_M': True, 'lr_separate': True,
          'reg_global': True, 'no_compcor': True,
          'sns_set': sns_in}
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    # assert set(sns_in) - set(sns) == set()

    ef_h, dd_h, vv_h, dv_ant_h, dv_pos_h, M_overall_h = (
        get_vd_ef(conn_highs, combine_regions=combine_regions,
                  combine_bilateral=False, anat_ver=anat_ver))
    ef_l, dd_l, vv_l, dv_ant_l, dv_pos_l, M_overall_l = (
        get_vd_ef(conn_lows, combine_regions=combine_regions,
                  combine_bilateral=False, anat_ver=anat_ver))
    task_ef = ef_l - ef_h

    dd_ef = dd_l - dd_h - M_overall_l + M_overall_h
    vv_ef = vv_l - vv_h - M_overall_l + M_overall_h
    dv_ant_ef = dv_ant_l - dv_ant_h - M_overall_l + M_overall_h
    dv_pos_ef = dv_pos_l - dv_pos_h - M_overall_l + M_overall_h
    return task_ef, dd_ef, vv_ef, dv_ant_ef, dv_pos_ef, sns


def get_gambling_behavior(sn, lr, key='RT_next'):
    df_trials = get_df_PE(sn, lr, cont_pe_by_event=True)
    run_num = 2 if lr == 'LR' else 1
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\GAMBLING_run{run_num}_TAB.txt'
    if not os.path.exists(fp):
        print(f'No file: {fp}')
        return None, None
    df = pd.read_csv(fp, delimiter='\t')
    df = df[df['Procedure[Trial]'] == r'GamblingTrialPROC'].reset_index()
    n_nans = df['QuestionMark.RESP'].isna().sum()
    df.dropna(subset=['QuestionMark.RESP', 'QuestionMark.RT'],
              inplace=True)

    df['RESP_next'] = df['QuestionMark.RESP'].shift(-1)
    df['RESP_CHANGE'] = df['QuestionMark.RESP'] != df['RESP_next']
    p_change = df['RESP_CHANGE'].sum() / df['RESP_CHANGE'].count()

    df['RT'] = df['QuestionMark.RT']
    df['RT_next'] = df['RT'].shift(-1)
    df['RT_slower'] = df['RT_next'] - df['RT']
    df['PE'] = df_trials['trial_type']
    df['event'] = df_trials['event']

    # df = df.groupby('PE')['RT_next'].mean()

    df_win = df[df['event'] == 'win']
    df_loss = df[df['event'] == 'loss']
    # print(df['PE'].value_counts())
    # print(df_win['PE'].value_counts())
    # print(df_loss['PE'].value_counts())
    # print('-------------')


    df_PE_win = df_win.groupby('PE')['RT_next'].mean()
    df_PE_loss = df_loss.groupby('PE')['RT_next'].mean()
    PE_ef_win = df_PE_win['high_PE'] - df_PE_win['low_PE']
    PE_ef_loss = df_PE_loss['high_PE'] - df_PE_loss['low_PE']
    PE_ef = PE_ef_win + PE_ef_loss
    return p_change, PE_ef

def get_gambling_behavior_OLD(sn, lr):
    # QuestionMark.RESP

    df_trials = get_df_PE(sn, lr)

    run_num = 2 if lr == 'LR' else 1
    fp = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\GAMBLING_run{run_num}_TAB.txt'
    # df = pd.read_csv(fp, delimiter='\t')
    # df.dropna(subset=['QuestionMark.RESP'], inplace=True)
    # df.reset_index(drop=True, inplace=True)

    df = pd.read_csv(fp, delimiter='\t')
    df = df[df['Procedure[Trial]'] == r'GamblingTrialPROC'].reset_index()
    n_nans = df['QuestionMark.RESP'].isna().sum()
    df.dropna(subset=['QuestionMark.RESP', 'QuestionMark.RT'],
              inplace=True)

    df['RESP_prev'] = df['QuestionMark.RESP'].shift(1)
    df['RESP_next'] = df['QuestionMark.RESP'].shift(-1)
    df['RESP_CHANGE'] = df['QuestionMark.RESP'] != df['RESP_next']
    # df = df.iloc[1:] # drop first row NaN
    df['RESP_CHANGE'] = df['RESP_CHANGE'].astype(float)
    df.loc[df['RESP_next'].isna(), 'RESP_CHANGE'] = np.nan

    df['event'] = df_trials['event']
    # print(df['event'])
    # quit()
    # df.loc[df['RESP_prev'].isna(), 'RESP_CHANGE'] = pd.NA

    # df = df[df['event'] == 'win']


    p_change = df['RESP_CHANGE'].sum() / df['RESP_CHANGE'].count()

    n_nan = df['RESP_CHANGE'].isna().sum()

    df['PE'] = df_trials['trial_type']
    df['event'] = df_trials['event']
    # df = df[df['event'] == 'win']

    df_PE = df.groupby('PE')['RESP_CHANGE'].mean()
    PE_ef = df_PE['high_PE'] - df_PE['low_PE']

    df_win = df[df['event'] == 'win']
    df_loss = df[df['event'] == 'loss']

    df_PE_win = df_win.groupby('PE')['RESP_CHANGE'].mean()
    df_PE_loss = df_loss.groupby('PE')['RESP_CHANGE'].mean()
    PE_ef_win = df_PE_win['high_PE'] - df_PE_win['low_PE']
    PE_ef_loss = df_PE_loss['high_PE'] - df_PE_loss['low_PE']
    PE_ef = PE_ef_win + PE_ef_loss
    # PE_ef = PE_ef_loss

    assert n_nan <= 1
    return p_change, PE_ef


if __name__ == '__main__':
    analyze_HCP_rs()