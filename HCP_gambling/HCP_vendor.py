import os
import time
import zlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import image
from scipy import stats as stats
from tqdm import tqdm

from HCP_gambling.preproc_gambling import get_df_events
from atlas_utils import get_atlas
from networks.old.network_funcs import load_FC_for_Lifu
from networks.vendor_partitioning import get_vendor_partitions, do_regression
from old.plot_gen import plot_connectivity
from utils import pickle_wrap


def bar_vendor(conn_highs, conn_lows, combine_regions, bilateral):
    itr, dd, vv, dv_ant, dv_pos, M_overall = get_vd_ef(conn_highs, combine_regions=combine_regions,
                                                       combine_bilateral=bilateral)
    df= pd.DataFrame({'high_PA': dd + vv, 'high_VD': dv_ant + dv_pos,
                       'high_dd': dd, 'high_vv': vv, 'high_dv_ant': dv_ant,
                       'high_dv_pos': dv_pos, })

    itr, dd, vv, dv_ant, dv_pos, M_overall = get_vd_ef(conn_lows, combine_regions=combine_regions,
                                                       combine_bilateral=bilateral)
    # df = df_high.copy()
    df['low_PA'] = dd + vv
    df['low_VD'] = dv_ant + dv_pos
    df['low_dd'] = dd
    df['low_vv'] = vv
    df['low_dv_ant'] = dv_ant
    df['low_dv_pos'] = dv_pos
    # df_low = df[['low_PA', 'low_VD', 'low_dd', 'low_vv', 'low_dv_ant', 'low_dv_pos']]

    for ef in ['PA', 'VD', 'dd', 'vv', 'dv_ant', 'dv_pos']:
        df[f'{ef}_diff'] = df[f'high_{ef}'] - df[f'low_{ef}']
        t, p = stats.ttest_rel(df[f'high_{ef}'], df[f'low_{ef}'])
        N = np.sum(~np.isnan(df[f'high_{ef}']))
        print(f'{ef}: t[{N - 1}] = {t:.2f}, {p=:.4f}')

    # M_PA = (df['high_PA'] + df['low_PA']) / 2
    # df['high_PA'] -= M_PA
    # df['low_PA'] -= M_PA
    # M_VD = (df['high_VD'] + df['low_VD']) / 2
    # df['high_VD'] -= M_VD
    # df['low_VD'] -= M_VD

    # df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['high_VD'].to_list() +
    #                          df['low_PA'].to_list() + df['low_VD'].to_list(),
    #                    'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
    #                    'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2})

    df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['low_PA'].to_list() +
                             df['high_VD'].to_list() + df['low_VD'].to_list(),
                       'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
                       'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2})

    plt.rcParams.update({'font.size': 21,
                         'font.sans-serif': 'Arial'})
    g = sns.catplot(x='high_low', y='FC', hue='PA_VD', data=df,
                        kind='bar',
                        # errci=68,
                        errorbar=('ci', 68),
                        # errwidth=1.5,
                        edgecolor='k',
                        # capsize=0.1, height=4,
                        alpha=0.7, linewidth=.7,#.7,
                        errwidth=1.2,
                        capsize=0.05,
                        # palette=sns.color_palette()
                        palette=['dodgerblue', 'red'],
                        height=5, aspect=0.8
                        )
    plt.ylabel('Mean connectivity')
    g._legend.remove()
    g.set_xticklabels(['High PE', 'Low PE'])
    plt.tight_layout()
    plt.xlabel('')
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.xlim(-.5, 1.5)
    fp = fr'result_pics/other/Study_1B_vendor.png'
    plt.savefig(fp, dpi=600)
    plt.show()


def get_sn_roi_ar(sn, lr, combine_regions=False, bilateral=False,
                  reg_global=False, no_compcor=False, rs=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral,
                      HCP=True)

    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    if rs:
        fp_lsa_lr = fr'E:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
        if not os.path.exists(fp_lsa_lr):
            fp_lsa_lr = fr'C:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
            assert os.path.exists(fp_lsa_lr)
    else:
        fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA{glob_str}{cc_str}.nii'
    try:
        img_lsa_lr = image.load_img(fp_lsa_lr)
    except EOFError:
        print('EOFError')
        print(f'{sn=}, {lr=}')
        print(f'{fp_lsa_lr=}')
        raise EOFError
    except zlib.error:
        print('zlib.error')
        print(f'{sn=}, {lr=}')
        print(f'{fp_lsa_lr=}')
        raise zlib.error
    data_lsa_lr = img_lsa_lr.get_fdata()
    # df_events = get_df_events(sn, 'LR')

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    # ROI2vecs = {}
    # region2vecs = defaultdict(list)
    ar = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        roi_data_lsa_lr = data_lsa_lr[atlas_roi]
        vals = roi_data_lsa_lr.mean(axis=0)
        ar.append(vals)
    ar = np.array(ar)
    return ar


def get_conn_sn(sn, combine_regions=False, bilateral=False, drop_neut=False,
                neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
                cont_PE_by_event=False, lr_separate=False,
                reg_global=False, no_compcor=False, median_split=True,
                drop_first=False, both_bhv=False, reset_trial0=False):

    if both_bhv:
        try:
            df_rl, df_lr = get_df_events(sn, 'both', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first, reset_trial0=reset_trial0)
        except Exception as e:
            print(f'ERROR in getting df: {sn}, {e=}')
            # bad_sns.append(sn)
            time.sleep(1)
            return None, sn
    else:
        try:
            df_lr = get_df_events(sn, 'LR', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first,
                                  reset_trial0=reset_trial0)
            df_rl = get_df_events(sn, 'RL', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first,
                                  reset_trial0=reset_trial0)
        except Exception as e:
            print(f'ERROR: {sn}, {e=}')
            # bad_sns.append(sn)
            time.sleep(1)
            return None, sn



    try:
        ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=reg_global, no_compcor=no_compcor)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        # bad_sns.append(sn)
        time.sleep(1)
        return None, sn

    if only:
        df_lr.loc[df_lr['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_lr.loc[df_lr['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high = ar[:, df_lr['trial_type'] == 'neut']
    else:
        ar_high = ar[:, df_lr['trial_type'] == 'high_PE']

    ar_low = ar[:, df_lr['trial_type'] == 'low_PE']
    # print(f'{ar_low.shape=}')
    # print(f'{ar_high.shape=}')
    # quit()

    try:
        ar = get_sn_roi_ar(sn, 'RL', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=reg_global,
                           no_compcor=no_compcor)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        # bad_sns.append(sn)
        return None, sn
    if only:
        df_rl.loc[df_rl['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_rl.loc[df_rl['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high2 = ar[:, df_rl['trial_type'] == 'neut']
    else:
        ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
    ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']
    # print(df_rl)
    # quit()
    if lr_separate:
        conn_high0 = np.corrcoef(ar_high)
        conn_high0[np.diag_indices_from(conn_high0)] = np.nan
        conn_low0 = np.corrcoef(ar_low)
        conn_low0[np.diag_indices_from(conn_low0)] = np.nan
        conn_high1 = np.corrcoef(ar_high2)
        conn_high1[np.diag_indices_from(conn_high1)] = np.nan
        conn_low1 = np.corrcoef(ar_low2)
        conn_low1[np.diag_indices_from(conn_low1)] = np.nan
        conn_high = (conn_high0 + conn_high1) / 2
        conn_low = (conn_low0 + conn_low1) / 2
    else:
        ar_high = np.concatenate([ar_high, ar_high2], axis=1)
        ar_low = np.concatenate([ar_low, ar_low2], axis=1)
        print(f'{ar_high.shape=} | {ar_low.shape=}')

        conn_high = np.corrcoef(ar_high)
        conn_high[np.diag_indices_from(conn_high)] = np.nan
        conn_low = np.corrcoef(ar_low)
        conn_low[np.diag_indices_from(conn_low)] = np.nan
    # print(conn_low1)
    # quit()

    # TODO: Lateralized connectivity.
    #  high R-A/high R-P and low L-A/low L-P means A-P connectivity
    return conn_high, conn_low


def make_conn(combine_regions=False, bilateral=False, drop_neut=False,
              neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
              cont_PE_by_event=False, lr_separate=True, num_sns=None,
              n_jobs=1, reg_global=False, no_compcor=False,
              sns_set=None, median_split=True,
              drop_first=False, both_bhv=False, reset_trial0=False):

    fns = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    if reg_global:
        fns = [fn for fn in fns if 'global' in fn]
    else:
        fns = [fn for fn in fns if 'global' not in fn]
    if no_compcor:
        fns = [fn for fn in fns if 'nocc' in fn]
    else:
        fns = [fn for fn in fns if 'nocc' not in fn]
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    print(f'{len(sns)=}')
    if sns_set is not None:
        sns = [sn for sn in sns if sn in sns_set]
    if num_sns is None:
        num_sns = 10_000
    # if sns is not None:
    #     sns = sns[:num_sns]
    # sns = sns[:-1]
    # sns = sns[::-1]

    conn_highs = []
    conn_lows = []
    bad_sns = []
    kw = {'combine_regions': combine_regions,  'bilateral': bilateral,
          'neut_as_PE': neut_as_PE, 'drop_neut': drop_neut, 'regr_M': regr_M,
          'only': only, 'cont_PE': cont_PE, 'cont_PE_by_event': cont_PE_by_event,
          'lr_separate': lr_separate, 'reg_global': reg_global,
          'no_compcor': no_compcor, 'median_split': median_split,
          'drop_first': drop_first, 'both_bhv': both_bhv,
          'reset_trial0': reset_trial0}

    if not no_compcor:
        del kw['no_compcor']

    # if no_compcor:
    #     from datetime import datetime
    #     dt_max = datetime(2024, 9, 15, 19, 50, 0)
    # elif cont_PE_by_event:
    #     from datetime import datetime
    #     dt_max = datetime(2024, 9, 15, 11, 0, 0)
    # else:
    #     dt_max = None

    from datetime import datetime
    dt_max = datetime(2024, 9, 21, 18, 45, 0)

    good_sns = []
    while len(good_sns) < num_sns and (len(sns) > 0):#, total=num_sns):
        sn = sns.pop()
        kw['sn'] = sn
        conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                             easy_override=False,
                                             dt_max=dt_max)
        if conn_high is None:
            print(f'Bad conn: {sn}, attempting to redo')
            conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                                 easy_override=True,
                                                 dt_max=dt_max)
        if conn_high is None:
            print('BAD CONN??')
            bad_sns.append(sn)
            continue

        conn_highs.append(conn_high)
        conn_lows.append(conn_low_sn)
        good_sns.append(sn)

    print(f'{bad_sns=}')
    conn_highs = np.array(conn_highs)
    conn_lows = np.array(conn_lows)

    print(f'Final sns: {len(good_sns)=}')
    return conn_highs, conn_lows, good_sns


def get_vd_ef(conn, combine_regions=False, combine_bilateral=False,
              anat_ver=3):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=True, weighted=False,
                              flip=True, thr=.9, scrub=False, anat_ver=anat_ver,
                              combine_regions=combine_regions)
    # print(f'{p_d_ant=},\n{p_d_pos=},\n{p_v_ant=},\n{p_v_pos=}')

    if combine_bilateral:
        p_d_ant = np.array(p_d_ant[::2]) // 2
        p_d_pos = np.array(p_d_pos[::2]) // 2
        p_v_ant = np.array(p_v_ant[::2]) // 2
        p_v_pos = np.array(p_v_pos[::2]) // 2

    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))


    return dd + vv - dv_ant - dv_pos, dd, vv, dv_ant, dv_pos, M_overall


def get_combo(kw):
    kw['only'] = 'loss'
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    kw['only'] = 'win'
    conn_highs2, conn_lows2, sns2 = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    assert sns == sns2
    conn_highs = np.mean([conn_highs, conn_highs2], axis=0)
    conn_lows = np.mean([conn_lows, conn_lows2], axis=0)
    return conn_highs, conn_lows, sns


def test_vendor(combine_regions=False, bilateral=False, corr_z=True,
                sub_ROI_expected=False):

    # kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
    #       'neut_as_PE': None, 'drop_neut': True, 'only': None,
    #       'num_sns': 500, 'cont_PE': 0.3, 'cont_PE_by_event': False,
    #       'regr_M': True, 'lr_separate': False,
    #       'reg_global': True, 'no_compcor': True,
    #       'median_split': True, 'drop_first': True,
    #       'both_bhv': True}
    #
    # kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
    #       'neut_as_PE': None, 'drop_neut': True, 'only': 'combo',
    #       'num_sns': 1000, 'cont_PE': 0.3,
    #       'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True,
    #       'reg_global': True, 'no_compcor': True, 'median_split': True,
    #       'drop_first': True, 'both_bhv': True,
    #       'reset_trial0': False}
    #
    #
    # kw = {'combine_regions': False, 'bilateral': False,
    #       'neut_as_PE': None, 'drop_neut': True,
    #       'only': None, 'num_sns': 1000,
    #       'cont_PE': 0.3, 'cont_PE_by_event': True,
    #       'regr_M': True, 'lr_separate': False,
    #       'reg_global': True, 'no_compcor': True,
    #       'median_split': True, 'drop_first': True,
    #       'both_bhv': True, 'reset_trial0': True}

    kw = {'combine_regions': combine_regions, 'bilateral': False, 'neut_as_PE': None,
          'drop_neut': True, 'only': 'combo', 'num_sns': 1000, 'cont_PE': 0.3,
          'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True,
          'reg_global': True, 'no_compcor': True, 'median_split': True,
          'drop_first': True, 'both_bhv': True, 'reset_trial0': True}

    # kw  = {'combine_regions': False, 'bilateral': False, 'neut_as_PE': None,
    #      'drop_neut': True, 'only': None, 'num_sns': 1000, 'cont_PE': 0.3,
    #      'cont_PE_by_event': True, 'regr_M': True, 'lr_separate': True,
    #      'reg_global': True, 'no_compcor': True, 'median_split': True,
    #      'drop_first': False, 'both_bhv': True, 'reset_trial0': True}



    print(f'{kw=}')

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


    if combine_regions:
        conn_highs[:, :, 46:] = np.nan
        conn_highs[:, 46:, :] = np.nan
        conn_lows[:, :, 46:] = np.nan
        conn_lows[:, 46:, :] = np.nan
    else:
        conn_highs[:, :, 210:] = np.nan
        conn_highs[:, 210:, :] = np.nan
        conn_lows[:, :, 210:] = np.nan
        conn_lows[:, 210:, :] = np.nan

    if sub_ROI_expected:
        ROI_expected = np.nanmean(conn_highs, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_highs -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_highs /= ROI_expected[None]
        ROI_expected = np.nanmean(conn_lows, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_lows -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_lows /= ROI_expected[None]

    bar_vendor(conn_highs, conn_lows, combine_regions, bilateral)

    dif = conn_highs - conn_lows
    M = np.nanmean(dif, axis=0)
    SE = stats.sem(dif, axis=0, nan_policy='omit')
    t = M / SE

    print(t.shape)


    if corr_z:
        t_flat = t[np.tril_indices_from(t, k=-1)]
        z_both = get_SchemeRep_regr(combine_regions=combine_regions, plot=False)
        # print(z_both.shape)
        # quit()
        z_flat = z_both[np.tril_indices_from(z_both, k=-1)]
        r, p = stats.spearmanr(t_flat, z_flat, nan_policy='omit')
        print(f'Gambling x SchemeRep: {r=:.2f}, {p=:.3f}')
    else:
        r = None


        # quit()

    ef_high = get_vd_ef(conn_highs, combine_regions=combine_regions,
                        combine_bilateral=bilateral)[0]
    ef_low = get_vd_ef(conn_lows, combine_regions=combine_regions,
                       combine_bilateral=bilateral)[0]
    itr = ef_low - ef_high
    t_final, p = stats.ttest_1samp(itr, 0)
    # plt.hist(itr)
    # plt.show()
    # quit()
    N = itr.shape[0]
    nans = np.sum(np.isnan(itr))
    F = t_final ** 2
    print(f't[{N - nans - 1}/{N - 1}] = {t_final:.2f}, {p=:.4f}, F = {F:.2f}')
    print(kw)



    if not bilateral:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral, HCP=True,
                          lifu_labels=combine_regions)


        title = str(kw)
        title_ = ''
        for i in range(len(title) // 50):
            title_ += title[i * 50:(i + 1) * 50] + '\n'
        title_ += title[(i + 1) * 50:]
        title = title_

        title += f'\nt[{N - nans - 1}/{N - 1}] = {t_final:.2f}'
        if r is not None:
            title += f', {r=:.2f}'
        # quit()

        # p_v_pos = [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
        # p_d_pos = [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145]
        # plt.imshow(t[np.ix_(p_v_pos, p_d_pos)])
        # plt.colorbar()
        # plt.show()
        # t[np.abs(t) < 3] = np.nan
        M_high = np.nanmean(conn_highs, axis=0)
        M_low = np.nanmean(conn_lows, axis=0)
        # print(len(atlas['tick_labels']))
        # print(len(atlas['ticks']))
        # quit()
        if combine_regions:
            atlas['ticks'] = atlas['ticks'][:23]
            atlas['tick_labels'] = atlas['tick_labels'][:23]
            atlas['tick_lows'] = atlas['tick_lows'][:23]
            t = t[:46, :46]
            fp_out = r'C:\PycharmProjects\SchemeRep\result_pics\other\Study_1B_PE_matrix.png'
        else:
            fp_out = None

        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title=title, tile=.01,
                          no_avg=True, cbar_label='t-value',
                          vmin=-6, vmax=6, fp=fp_out)

        # plot_connectivity(M_high, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # plot_connectivity(M_low, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # plot_connectivity(M_high - M_low, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # quit()

        # z_threshed = z_both
        # z_threshed[np.abs(z_threshed) < 2] = np.nan
        #
        # z_threshed[np.abs(t) < 3] = np.nan
        # plot_connectivity(z_threshed, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   vmin=-4, vmax=4)
        # conjunct = np.logical_and(np.abs(t) > 3, np.abs(z_both) > 2)

    quit()
    n, bins, patches = plt.hist(itr, range=(-0.4, 0.4), bins=40)
    plt.plot([0, 0], [0, np.max(n)], 'r--')
    plt.show()

    plt.hist(itr, range=(-0.4, 0.4), bins=40,
             cumulative=True, density=True)
    plt.plot([0, 0], [0, 1], 'r--')
    plt.plot([-.4, .4], [0.5, 0.5], 'r--')
    plt.xlim(-0.4, 0.4)
    plt.ylim(0, 1)
    plt.show()
    quit()


def get_SchemeRep_regr(regress=False, combine_regions=False, plot=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    z_both = do_regression(sn_inc_conn, flip=False) # False = (Incongruent > Congruent)

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False, HCP=True,
                      lifu_labels=True)
    if combine_regions:
        z_both = z_both[:54, :54]
    if plot:
        plot_connectivity(z_both, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title='SchemeRep matrix', tile=.01,
                          no_avg=True, cbar_label='Correlation (r)')

    return z_both

if __name__ == '__main__':
    # get_SchemeRep_regr(combine_regions=False, plot=True)
    # test_corr()
    # fp = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    # img = image.load_img(fp)
    # print(img.shape)
    # LSS_gambling()
    test_vendor()