import os
import pathlib

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


def plot_Fig2B_bars(conn_highs, conn_lows, combine_regions, bilateral):
    itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
        conn_highs, combine_regions=combine_regions, combine_bilateral=bilateral)
    df = pd.DataFrame({'high_PA': dd + vv, 'high_VD': dv_ant + dv_pos,
                       'high_dd': dd, 'high_vv': vv, 'high_dv_ant': dv_ant,
                       'high_dv_pos': dv_pos, })

    itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
        conn_lows, combine_regions=combine_regions, combine_bilateral=bilateral)
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
        print(f'{ef}: t[{N - 1}] = {t:.2f}, {p=:.4f}')

    df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['low_PA'].to_list() +
                             df['high_VD'].to_list() + df['low_VD'].to_list(),
                       'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
                       'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2,
                       'sn': list(range(len(df))) * 4})

    plt.rcParams.update({'font.size': 21,
                         'font.sans-serif': 'Arial'})

    df_PA = df[df['PA_VD'] == 'PA']
    df_PA['FC'] -= df_PA.groupby('sn')['FC'].transform('mean')

    df_VD = df[df['PA_VD'] == 'VD']
    df_VD['FC'] -= df_VD.groupby('sn')['FC'].transform('mean')

    fig, axs = plt.subplots(1, 2, figsize=(6, 5))

    plt.sca(axs[0])
    g = sns.barplot(x='high_low', y='FC', hue='PA_VD', data=df_PA,
                    errorbar=('ci', 68), edgecolor='k',
                    alpha=0.7, linewidth=.7, errwidth=1.2,
                    capsize=0.05, palette=['dodgerblue'],
                    ax=axs[0], legend=False,
                    )
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.xlim(-.5, 1.5)
    plt.ylim(-.026, .026)
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.ylabel('Mean connectivity')
    plt.xlabel('')
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    plt.sca(axs[1])
    g = sns.barplot(x='high_low', y='FC', hue='PA_VD', data=df_VD,
                    errorbar=('ci', 68), edgecolor='k',
                    alpha=0.7, linewidth=.7, errwidth=1.2,
                    capsize=0.05, palette=['red'],  # height=5,
                    ax=axs[1], legend=False,
                    )
    plt.ylabel('')
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.xlabel('')
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.yticks([-.01, 0, .01])
    plt.xlim(-.5, 1.5)
    plt.ylim(-.013, .013)
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tight_layout()
    fp = fr'result_pics/Fig3/Fig3A_matrix.png'
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


def get_conn_sn(sn, combine_regions=False, bilateral=False,
                only=None, learning_rate=None,
                drop_first=False, reset_trial0=False,
                ):
    try:
        df_rl, df_lr = get_df_PE(sn, 'both',
                                 learning_rate=learning_rate,
                                 drop_first=drop_first,
                                 reset_trial0=reset_trial0)
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        return None, sn

    try:
        ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=True,
                           no_compcor=True)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        return None, sn

    if only:
        df_lr.loc[df_lr['event'] != only, 'trial_type'] = 'only'

    df_lr.loc[df_lr['event'] == 'neut', 'trial_type'] = 'neut'

    ar_high = ar[:, df_lr['trial_type'] == 'high_PE']

    ar_low = ar[:, df_lr['trial_type'] == 'low_PE']

    try:
        ar = get_sn_roi_ar(sn, 'RL', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=True,
                           no_compcor=True)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        return None, sn
    if only:
        df_rl.loc[df_rl['event'] != only, 'trial_type'] = 'only'

    df_rl.loc[df_rl['event'] == 'neut', 'trial_type'] = 'neut'

    ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
    ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']

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

    return conn_high, conn_low


def make_conn(combine_regions=False, bilateral=False,
              only=None, learning_rate=None,
              num_sns=None, drop_first=False,
              reset_trial0=False, ):
    # final subjects established as ones with both task-fMRI LR/RL and resting-state LR/RL
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')

    conn_highs = []
    conn_lows = []
    bad_sns = []
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'only': only, 'learning_rate': learning_rate,
          'drop_first': drop_first, 'reset_trial0': reset_trial0,
          }

    from datetime import datetime
    dt_max = datetime(2024, 9, 21, 18, 45, 0)

    good_sns = []
    while len(good_sns) < num_sns and (len(sns) > 0):
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


def get_PE_x_Conn_effect(conn, combine_regions=False, combine_bilateral=False,
                         anat_ver=3):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True,
                             do_PA=True, thr=.9, anat_ver=anat_ver,
                             combine_regions=combine_regions)

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


def get_combo(kw, easy_override=False):
    kw['only'] = 'loss'
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    kw['only'] = 'win'
    conn_highs2, conn_lows2, sns2 = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=easy_override))
    assert sns == sns2
    conn_highs = np.mean([conn_highs, conn_highs2], axis=0)
    conn_lows = np.mean([conn_lows, conn_lows2], axis=0)
    return conn_highs, conn_lows, sns


def run_Study1B_analysis(combine_regions=True, bilateral=False, corr_z=True,
                         sub_ROI_expected=False):
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'combo':
        # can either be run while averaging a loss matrix & win matrix ('combo')
        #   or just making a single one covering both PE ('both')
        # the manuscript uses 'combo'
        conn_highs, conn_lows, sns = get_combo(kw, easy_override=True)
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
        ROI_expected = np.nanmean(conn_lows, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_lows -= ROI_expected[None]

    plot_Fig2B_bars(conn_highs, conn_lows, combine_regions, bilateral)

    overall = (conn_highs + conn_lows) / 2
    print(f'{overall.shape=}')

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
                                   combine_bilateral=bilateral)[0]
    ef_low = get_PE_x_Conn_effect(conn_lows, combine_regions=combine_regions,
                                  combine_bilateral=bilateral)[0]
    itr = ef_low - ef_high
    t_final, p = stats.ttest_1samp(itr, 0)

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

        if combine_regions:
            atlas['ticks'] = atlas['ticks'][:23]
            atlas['tick_labels'] = atlas['tick_labels'][:23]
            atlas['tick_lows'] = atlas['tick_lows'][:23]
            t = t[:46, :46]
            fp_out = r'C:\PycharmProjects\SchemeRep\result_pics\Fig3\Fig3B_PE_x_Conn_bars.png'
        else:
            fp_out = None

        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title=title, tile=.01,
                          no_avg=True, cbar_label='t-value',
                          vmin=-6, vmax=6, fp=fp_out)


def get_Study1A_matrix_for_corr(combine_regions=False, plot=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
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
    run_Study1B_analysis()
