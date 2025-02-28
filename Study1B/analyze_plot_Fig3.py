import os
import pathlib

from pygments.unistring import combine

from marinate.pkld import pkld

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import os
import time
import zlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import image
from scipy import stats as stats

from Study1B.preprocess_Study1B import get_df_PE
from Utils.atlas_funcs import get_atlas
from Study1A.load_Study1A_funcs import load_FC
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions, get_regression_matrix
from Utils.plotting_funcs import plot_connectivity
from Utils.pickle_wrap_funcs import pickle_wrap


def plot_Fig2B_bars(conn_highs, conn_lows, combine_regions, bilateral,
                    schaefer=False):
    itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
        conn_highs, combine_regions=combine_regions, combine_bilateral=bilateral,
        schaefer=schaefer)
    df = pd.DataFrame({'high_PA': dd + vv, 'high_VD': dv_ant + dv_pos,
                       'high_dd': dd, 'high_vv': vv, 'high_dv_ant': dv_ant,
                       'high_dv_pos': dv_pos, })

    itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
        conn_lows, combine_regions=combine_regions, combine_bilateral=bilateral,
        schaefer=schaefer)
    df['low_PA'] = dd + vv
    df['low_VD'] = dv_ant + dv_pos
    df['low_dd'] = dd
    df['low_vv'] = vv
    df['low_dv_ant'] = dv_ant
    df['low_dv_pos'] = dv_pos

    for ef in ['PA', 'VD', 'dd', 'vv', 'dv_ant', 'dv_pos']:
        df[f'{ef}_diff'] = df[f'high_{ef}'] - df[f'low_{ef}']
        t, p = stats.ttest_rel(df[f'high_{ef}'], df[f'low_{ef}'])
        N = np.sum(~np.isnan(df[f'high_{ef}']))
        if ef in ['PA', 'dd', 'vv']:
            extra = ' (expected negative)'
        elif ef in ['VD', 'dv_ant', 'dv_pos']:
            extra = ' (expected positive)'
        else:
            raise ValueError
        print(f'{ef}: t[{N - 1}] = {t:.2f}, {p=:.4f} {extra}')

    df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['low_PA'].to_list() +
                             df['high_VD'].to_list() + df['low_VD'].to_list(),
                       'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
                       'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2,
                       'sn': list(range(len(df))) * 4})

    plt.rcParams.update({'font.size': 21,
                         'font.sans-serif': 'Arial'})

    df['FC'] -= df.groupby(['sn', 'high_low'])['FC'].transform('mean')
    df_PA = df[df['PA_VD'] == 'PA']
    df_PA['FC'] -= df_PA.groupby('sn')['FC'].transform('mean')
    df_VD = df[df['PA_VD'] == 'VD']
    df_VD['FC'] -= df_VD.groupby('sn')['FC'].transform('mean')

    g = sns.catplot(x='high_low', y='FC', data=df_PA,
                    kind='boxen',
                    linecolor='k',
                    palette=['dodgerblue'],
                    legend=False,
                    saturation=0.9,
                    height=5, aspect=0.75,
                    flier_kws={'edgecolor': ['k'],
                               'linewidth': 0.8,
                               'marker': '.',
                               },
                    )
    for ax in g.axes.flat:
        for line in ax.lines:
            if line.get_linestyle() == '-':
                line.set_color('white')
                line.set_linewidth(1.5)

    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.xlim(-.5, 1.5)
    plt.ylim(-.15, .15)
    plt.yticks([-.1, 0, .1])
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.ylabel('Mean connectivity')
    plt.xlabel('')
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    fp = fr'result_pics/Fig3/Fig3B_left_boxen.png'
    plt.savefig(fp, dpi=600)
    plt.show()

    g = sns.catplot(x='high_low', y='FC', data=df_VD,
                    kind='boxen',
                    linecolor='k',
                    palette=['red'],
                    legend=False,
                    saturation=0.9,
                    height=5, aspect=0.75,
                    flier_kws={'edgecolor': ['k'],
                               'linewidth': 0.5,
                               'marker': '.',
                               }
                    )
    for ax in g.axes.flat:
        for line in ax.lines:
            if line.get_linestyle() == '-':
                line.set_color('white')
                line.set_linewidth(1.5)

    plt.ylabel('Mean connectivity', color='w')
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.xlabel('')
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.xlim(-.5, 1.5)
    plt.ylim(-.15, .15)
    plt.yticks([-.1, 0, .1])
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tight_layout()
    fp = fr'result_pics/Fig3/Fig3B_right_boxen.png'
    plt.savefig(fp, dpi=600)
    plt.show()


def get_sn_roi_ar(sn, lr, combine_regions=False, bilateral=False,
                  reg_global=False, no_compcor=False, rs=False):
    if isinstance(combine_regions, tuple):
        assert (combine_regions[1][0].lower() == 'schaefer' or
                combine_regions[1].lower() == 'schaefer')
        if isinstance(combine_regions[1], tuple):
            schaefer = (True, combine_regions[1][1])
        else:
            schaefer = True
        atlas = get_atlas(combine_regions=combine_regions[0],
                          combine_bilateral=bilateral,
                          HCP=True, schaefer=schaefer)
    else:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral,
                          HCP=True)
    # print('TEST TEST')
    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    if rs:
        fp_lsa_lr = fr'E:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
        if not os.path.exists(fp_lsa_lr):
            fp_lsa_lr = fr'C:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
            assert os.path.exists(fp_lsa_lr)
    else:
        fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA{glob_str}{cc_str}.nii'
        # fp_lsa_lr = fr'H:\HCP_gambling\LSA\{sn}_{lr}_LSA{glob_str}{cc_str}.nii'

    # print('toast')
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

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']

    ar = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        roi_data_lsa_lr = data_lsa_lr[atlas_roi]
        vals = roi_data_lsa_lr.mean(axis=0)
        ar.append(vals)
    ar = np.array(ar)
    return ar


@pkld(overwrite=False)
def get_study1b_ar(sn, combine_regions=True, bilateral=False,
                   only=None, learning_rate=None,
                   drop_first=False, reset_trial0=False,
                   rl_lr='both'):
    # try:
    df_rl, df_lr = get_df_PE(sn, 'both',
                             learning_rate=learning_rate,
                             drop_first=drop_first,
                             reset_trial0=reset_trial0)
    # except Exception as e:
    #     print(f'Bad df: {sn}')
    #     return None, None

    try:
        ar = get_sn_roi_ar(sn, rl_lr, combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=True,
                           no_compcor=True)
    except ValueError:
        print(f'Bad fMRI: {sn}')
        return None, None
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        return None, None

    if rl_lr.lower() == 'lr':
        df = df_lr
    elif rl_lr.lower() == 'rl':
        df = df_rl
    else:
        raise ValueError

    if only:
        df.loc[df['event'] != only, 'trial_type'] = 'only'

    df.loc[df['event'] == 'neut', 'trial_type'] = 'neut'
    ar_high = ar[:, df['trial_type'] == 'high_PE']
    ar_low = ar[:, df['trial_type'] == 'low_PE']

    return ar_high, ar_low


def get_conn_sn(sn, combine_regions=True, bilateral=False,
                only=None, learning_rate=None,
                drop_first=False, reset_trial0=False,
                rl_lr='both'
                ):

    ar_high, ar_low = get_study1b_ar(sn, combine_regions=combine_regions,
                                     bilateral=bilateral,
                                     only=only, learning_rate=learning_rate,
                                     drop_first=drop_first, reset_trial0=reset_trial0,
                                     rl_lr='lr')
    if ar_high is None:
        print(f'Failed get_conn_sn: {sn}')
        return None, sn, None, None, None, None

    ar_high2, ar_low2 = get_study1b_ar(sn, combine_regions=combine_regions,
                                       bilateral=bilateral,
                                       only=only, learning_rate=learning_rate,
                                       drop_first=drop_first, reset_trial0=reset_trial0,
                                       rl_lr='rl')
    if ar_high2 is None:
        print(f'Failed get_conn_sn (lr): {sn}')
        return None, sn, None, None, None, None

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
    return conn_high, conn_low, conn_high0, conn_high1, conn_low0, conn_low1


def make_conn(combine_regions=False, bilateral=False,
              only=None, learning_rate=None,
              num_sns=None, drop_first=False,
              reset_trial0=False, get6=False):
    # final subjects established as ones with both task-fMRI LR/RL and resting-state LR/RL
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')

    conn_highs, conn_lows = [], []
    conn_highs0, conn_highs1, conn_lows0, conn_lows1 = [], [], [], []
    bad_sns = []
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'only': only, 'learning_rate': learning_rate,
          'drop_first': drop_first, 'reset_trial0': reset_trial0,
          }

    from datetime import datetime
    dt_max = datetime(2024, 9, 21, 18, 45, 0)
    sns = sns[::-1]

    good_sns = []
    t_st = time.time()
    while len(good_sns) < num_sns and (len(sns) > 0):
        sn = sns.pop()
        kw['sn'] = sn
        t_st = time.time()

        # conn_high_sn, conn_low_sn, conn_high0, conn_high1, conn_low0, conn_low1
        conn_high_sn, conn_low_sn, conn_high0_sn, conn_high1_sn, conn_low0_sn, conn_low1_sn = (
            pickle_wrap(get_conn_sn, kwargs=kw, easy_override=False, dt_max=dt_max))

        if conn_high_sn is None:
            print(f'Bad conn: {sn}, attempting to redo')
            conn_high_sn, conn_low_sn, conn_high0_sn, conn_high1_sn, conn_low0_sn, conn_low1_sn = (
                pickle_wrap(get_conn_sn, kwargs=kw, easy_override=True, dt_max=dt_max))

        print(f'Time needed for sn ({sn}): {time.time() - t_st:.2f} s')
        # quit()

        if conn_high_sn is None:
            print('BAD CONN??')
            bad_sns.append(sn)
            continue

        conn_highs.append(conn_high_sn)
        conn_lows.append(conn_low_sn)
        good_sns.append(sn)
        if get6:
            conn_highs0.append(conn_high0_sn)
            conn_highs1.append(conn_high1_sn)
            conn_lows0.append(conn_low0_sn)
            conn_lows1.append(conn_low1_sn)

    print(f'{bad_sns=}')
    conn_highs = np.array(conn_highs)
    conn_lows = np.array(conn_lows)

    print(f'Final sns: {len(good_sns)=}')
    if get6:
        return (conn_highs, conn_lows, good_sns, conn_highs0, conn_highs1,
                conn_lows0, conn_lows1)
    else:
        return conn_highs, conn_lows, good_sns


def get_PE_x_Conn_effect(conn, combine_regions=False, combine_bilateral=False,
                         anat_ver=3, schaefer=False):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True,
                             do_PA=True, thr=.9, anat_ver=anat_ver,
                             combine_regions=combine_regions,
                             schaefer=schaefer)
    # print(f'{schaefer=}')
    # print(f'{np.max(p_d_ant)=}')
    # print(f'{np.max(p_d_pos)=}')
    # print(f'{np.max(p_v_ant)=}')
    # print(f'{np.max(p_v_pos)=}')
    # quit()
    # print(p_dorsal)
    if schaefer:
        # if combine_regions:
        #     assert np.max(p_dorsal + p_ventral) > 60
        # else:
        if not combine_regions:
            assert np.max(p_dorsal + p_ventral) > 250
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


def get_combo(kw, easy_override=False, get6=False):
    kw = kw.copy()
    kw['only'] = 'loss'
    kw['get6'] = get6
    tup = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    if get6:
        (conn_highs_loss, conn_lows_loss, sns,
         conn_highs0_loss, conn_highs1_loss, conn_lows0_loss, conn_lows1_loss) = tup
    else:
        conn_highs_loss, conn_lows_loss, sns = tup

    kw['only'] = 'win'
    tup = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    if get6:
        (conn_highs_win, conn_lows_win, sns2,
         conn_highs0_win, conn_highs1_win, conn_lows0_win, conn_lows1_win) = tup
    else:
        conn_highs_win, conn_lows_win, sns2 = tup
    assert sns == sns2
    conn_highs = np.mean([conn_highs_loss, conn_highs_win], axis=0)
    conn_lows = np.mean([conn_lows_loss, conn_lows_win], axis=0)
    if get6:
        conn_highs0 = np.mean([conn_highs0_loss, conn_highs0_win], axis=0)
        conn_highs1 = np.mean([conn_highs1_loss, conn_highs1_win], axis=0)
        conn_lows0 = np.mean([conn_lows0_loss, conn_lows0_win], axis=0)
        conn_lows1 = np.mean([conn_lows1_loss, conn_lows1_win], axis=0)
        return conn_highs, conn_lows, sns, conn_highs0, conn_highs1, conn_lows0, conn_lows1
    else:
        return conn_highs, conn_lows, sns

@pkld
def get_wl_contrast_conn(kw, easy_override=False, get6=False):
    # easy_override = True
    kw['only'] = 'loss'
    kw['get6'] = get6
    tup = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    if get6:
        (conn_highs_loss, conn_lows_loss, sns,
         conn_highs0_loss, conn_highs1_loss, conn_lows0_loss, conn_lows1_loss) = tup
    else:
        conn_highs_loss, conn_lows_loss, sns = tup
    kw['only'] = 'win'
    tup = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    if get6:
        (conn_highs_win, conn_lows_win, sns2,
         conn_highs0_win, conn_highs1_win, conn_lows0_win, conn_lows1_win) = tup
    else:
        conn_highs_win, conn_lows_win, sns2 = tup

    conn_wins = np.mean([conn_highs_win, conn_lows_win], axis=0)
    conn_loss = np.mean([conn_highs_loss, conn_lows_loss], axis=0)
    if get6:
        conn_wins0 = np.mean([conn_highs0_win, conn_lows0_win], axis=0)
        conn_wins1 = np.mean([conn_highs1_win, conn_lows1_win], axis=0)
        conn_loss0 = np.mean([conn_highs0_loss, conn_lows0_loss], axis=0)
        conn_loss1 = np.mean([conn_highs1_loss, conn_lows1_loss], axis=0)
        return conn_wins, conn_loss, sns, conn_wins0, conn_wins1, conn_loss0, conn_loss1
    else:
        return conn_wins, conn_loss, sns

def run_Study1B_analysis(combine_regions=False, bilateral=False, corr_z=False,
                         sub_ROI_expected=False, schaefer=False):#(True, 400)):
    if isinstance(schaefer, tuple):
        combine_regions = (combine_regions, ('schaefer', schaefer[1]))
    elif schaefer:
        combine_regions = (combine_regions, 'schaefer')
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'wl':
        conn_highs, conn_lows, sns = (
            get_wl_contrast_conn(kw, easy_override=False))
    elif kw['only'] == 'combo':
        # can either be run while averaging a loss matrix & win matrix ('combo')
        #   or just making a single one covering both PE ('both')
        # the manuscript uses 'combo'
        conn_highs, conn_lows, sns = (
            get_combo(kw, easy_override=False))
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    if schaefer:
        combine_regions = combine_regions[0]

    # print(conn_highs[0])
    # print(sns[0])
    # quit()

    if schaefer:
        pass
    else:
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
        ROI_expected = np.nanmean(conn_lows, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_lows -= ROI_expected[None]

    # print(f'{schaefer=}')
    # quit()
    # if not combine_regions:
    plot_Fig2B_bars(conn_highs, conn_lows, combine_regions, bilateral,
                    schaefer=schaefer)

    overall = (conn_highs + conn_lows) / 2

    plt.rcParams.update({'font.size': 16,
                         'font.sans-serif': 'Arial'})

    dif = conn_highs - conn_lows
    M = np.nanmean(dif, axis=0)
    SE = stats.sem(dif, axis=0, nan_policy='omit')
    t = M / SE

    if corr_z:
        t_flat = t[np.tril_indices_from(t, k=-1)]
        z_both = get_Study1A_matrix_for_corr(combine_regions=combine_regions, plot=False)
        z_flat = z_both[np.tril_indices_from(z_both, k=-1)]
        r, p = stats.spearmanr(t_flat, z_flat, nan_policy='omit')
        print(f'Gambling x SchemeRep: {r=:.2f}, {p=:.3f}')
    else:
        r = None


    ef_high = get_PE_x_Conn_effect(conn_highs, combine_regions=combine_regions,
                                   combine_bilateral=bilateral,
                                   schaefer=schaefer)[0]
    ef_low = get_PE_x_Conn_effect(conn_lows, combine_regions=combine_regions,
                                  combine_bilateral=bilateral,
                                  schaefer=schaefer)[0]
    itr = ef_low - ef_high
    # print(ef_high)
    # print(sns)
    print(conn_highs[:, 2, 4])
    t_final, p = stats.ttest_1samp(itr, 0)

    N = itr.shape[0]
    F = t_final ** 2
    print(f't[{N - 1}] = {t_final:.2f}, {p=:.4f}, F = {F:.2f}')
    quit()

    if not bilateral:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral, HCP=True,
                          lifu_labels=combine_regions,
                          schaefer=schaefer)

        title = f'\nt[{N - 1}] = {t_final:.2f}, f[{N - 1}] = {F:.2f}'
        if r is not None:
            title += f', {r=:.2f}'

        if schaefer:
            fp_out = None
        elif combine_regions:
            atlas['ticks'] = atlas['ticks'][:23]
            atlas['tick_labels'] = atlas['tick_labels'][:23]
            atlas['tick_lows'] = atlas['tick_lows'][:23]
            t = t[:46, :46]
            fp_out = r'C:\PycharmProjects\SchemeRep\result_pics\Fig3\Fig3A_matrix.png'
        else:
            fp_out = None

        for i in range(len(atlas['ticks'])):
            tick = atlas['ticks'][i]
            label = atlas['tick_labels'][i]
            low = atlas['tick_lows'][i]
            print(f'{i} | {tick=:<3}, {low=:<4}, {label=}')

        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title=title, tile=.01,
                          no_avg=True, cbar_label='t-value',
                          vmin=-4, vmax=4, fp=fp_out)


def get_Study1A_matrix_for_corr(combine_regions=False, plot=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    z_both = get_regression_matrix(sn_inc_conn, flip=False)  # False = (Incongruent > Congruent)

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
    # run_Study1B_analysis()
    run_Study1B_analysis(combine_regions=False)
