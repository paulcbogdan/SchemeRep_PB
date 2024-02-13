# import os
# os.chdir(r'C:\PycharmProjects_C\SchemeRep\networks')
# import sys
# sys.path.extend([r'C:\PycharmProjects_C\SchemeRep'])
from atlas_utils import get_atlas
# from corr_RSA_x_vendor import get_module_trialwise_z
from emotemporal_bar import get_stars
from network_IRAF import get_formula_cols

from functools import cache
from collections import defaultdict

import numpy as np
import pandas as pd

from old.modularity import get_partition_cross, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from vendor_partitioning import get_vendor_partitions, scrub_plot_p
from utils import timing, pickle_wrap, stdize
import scipy.stats as stats
from warnings import filterwarnings
import seaborn as sns
import matplotlib.pyplot as plt

filterwarnings('ignore', category=UserWarning)

def get_module_cross_trialwise_z(conn_trials, p_mod0, p_mod1, trialwise=True):
    conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)
    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    if trialwise:
        agg_zs = np.nanmean(flat_cross, axis=-1)
        agg_zs = np.nanmean(agg_zs, axis=1) # omit inc axis
        return agg_zs
    else:
        rs = np.nanmean(flat_cross, axis=2)
        agg_rs = np.nanmean(rs, axis=-1)
        return agg_rs

def get_module_trialwise_z(sn_inc_conn_trials, p_module):
    # sn_inc_conn_trials = sn_inc_activity_std[..., None, :] * \
    #                      sn_inc_activity_std[..., None, :, :]
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1) # omit inc axis
    return sn_agg_trials_dd

def get_memory_p():
    atlas = get_atlas()
    ROIs_mem = ['Hipp']
    p_mem = []
    for i, roi in enumerate(atlas['ROIs']):
        for key_mem in ROIs_mem:
            if key_mem in roi:
                p_mem.append(i)
                break
    return p_mem

def get_DMN_p(exclude_ps=None):
    atlas = get_atlas()

    from nichord.coord_labeler import get_idx_to_label
    idx_to_label = pickle_wrap(None, get_idx_to_label,
                               kwargs={'coords': atlas['coords'],
                                       'atlas': 'yeo'},
                               )
    DMN_idxs = [idx for idx, label in idx_to_label.items() if 'DMN' in label]
    # print(DMN_idxs)


    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=False, scrub=False)
    exclude_ps = p_dorsal + p_ventral

    for i in DMN_idxs:
        BNA_label = atlas['ROIs'][i]
        if i in p_d_pos:
            print(f'{BNA_label} ({i}): In: Dor-Pos')
        elif i in p_d_ant:
            print(f'{BNA_label} ({i}): In: Dor-Ant')
        elif i in p_v_pos:
            print(f'{BNA_label} ({i}): In: Ven-Pos')
        elif i in p_v_ant:
            print(f'{BNA_label} ({i}): In: Ven-Ant')
        else:
            print(f'{BNA_label} ({i}): Not in any')

    print('--------')

    cnt = defaultdict(lambda: 0)
    for i in p_d_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_d_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Ant: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Ant: {cnt}')


def get_df_p_x_p(p0, p1, key, fp='obj7_fMRI'):

    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]

    if p1 is None:
        sn_agg_trials_dd = get_module_trialwise_z(conn_trials, p0)

    else:
        sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials, p0, p1)
    for i, df_sn in enumerate(df_sns_l):
        df_sn[key] = sn_agg_trials_dd[i, :]

    df_sns = pd.concat(df_sns_l)
    df_sns = df_sns[['sn', 'obj', key]]
    return df_sns



@timing
@cache
def get_vendor_df(fp='obj7_fMRI', thr=2.0, scrub=False, anat=True,
                  z_score=False):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    # if anat:
    #     p_d_ant, p_d_pos, p_v_ant, p_v_pos = get_anat_vendor_partitions()
    # else:
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat, scrub=scrub)

    # print(f'{p_d_ant=}')
    # print(f'{p_d_pos=}')
    # print(f'{p_v_ant=}')
    # print(f'{p_v_pos=}')

    # quit()

    # plot_ROI_scores(ts, results_coords, fp_out='trash.png', show=True,
    #                 vmin=0, vmax=2, title='all')

    assert set(p_d_pos).intersection(p_d_ant) == set()
    assert set(p_d_pos).intersection(p_v_ant) == set()
    assert set(p_d_pos).intersection(p_v_pos) == set()

    n_rois = sn_inc_activity.shape[2]
    matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]
                            # idk why I can't just broadcast matrix_mask

    sn_inc_act_M_pos_d = np.nanmean(sn_inc_activity[:, :, p_d_pos, :],
                                    axis=(1, 2))
    sn_inc_act_M_ant_d = np.nanmean(sn_inc_activity[:, :, p_d_ant, :],
                                    axis=(1, 2))
    sn_inc_act_M_pos_v = np.nanmean(sn_inc_activity[:, :, p_v_pos, :],
                                    axis=(1, 2))
    sn_inc_act_M_ant_v = np.nanmean(sn_inc_activity[:, :, p_v_ant, :],
                                    axis=(1, 2))
    sn_inc_act_M_overall = np.nanmean(sn_inc_activity, axis=(1, 2))

    sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                    p_d_pos, p_d_ant)
    sn_agg_trials_vv = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                    p_v_pos, p_v_ant)
    # plt.hist(sn_agg_trials_dd.flatten(), bins=25, range=(-0.5, 0.5))
    # plt.show()
    # print(sn_agg_trials_vv.shape)
    # quit()

    sn_agg_trials_dv_ant = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                        p_d_ant, p_v_ant)
    sn_agg_trials_dv_pos = get_module_cross_trialwise_z(conn_trials,#sn_inc_activity_std,
                                                        p_d_pos, p_v_pos)
    for i, df_sn in enumerate(df_sns_l):
        old_cols = set(df_sn.columns)
        df_sn['pd_M'] = sn_inc_act_M_pos_d[i, :]
        df_sn['ad_M'] = sn_inc_act_M_ant_d[i, :]
        df_sn['d_M'] = df_sn['pd_M'] + df_sn['ad_M']
        df_sn['pv_M'] = sn_inc_act_M_pos_v[i, :]
        df_sn['av_M'] = sn_inc_act_M_ant_v[i, :]
        df_sn['v_M'] = df_sn['pv_M'] + df_sn['av_M']
        df_sn['all_M'] = df_sn['d_M'] + df_sn['v_M']
        df_sn['brain_M'] = sn_inc_act_M_overall[i, :]

        df_sn['dd'] = sn_agg_trials_dd[i, :]
        df_sn['vv'] = sn_agg_trials_vv[i, :]
        df_sn['dv_ant'] = sn_agg_trials_dv_ant[i, :]
        df_sn['dv_pos'] = sn_agg_trials_dv_pos[i, :]
        df_sn['dd_vv'] = df_sn['dd'] + df_sn['vv'] #- \
                       # df_sn['dv_ant'] - df_sn['dv_pos']
        df_sn['cross'] = df_sn['dv_ant'] + df_sn['dv_pos']
        df_sn['vendor'] = df_sn['dd_vv'] - df_sn['cross']
        new_cols = list(set(df_sn.columns) - old_cols)
        df_sn[new_cols] = df_sn[new_cols].apply(stats.zscore, axis=0,
                                                nan_policy='omit')
    df_sns = pd.concat(df_sns_l)
    df_sns['age'] = df_sns['sn'].apply(lambda x: int(x[0]))

    return df_sns

def vendor_lmer():
    df = pickle_wrap(None, get_vendor_df, kwargs={'fp': 'obj7_fMRI',
                                                  'scrub': True},
                     cache_dir='cache', easy_override=True)

    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    df['brain_M'] = stats.zscore(df['brain_M'], nan_policy='omit')
    formula_gen = 'vv ~ 1 + dd*age + inc + brain_M + dv_ant + dv_pos + ' \
                  '(1 + dd*age | sn)'
    cols = get_formula_cols(df, formula_gen)
    df_vals = df[cols].dropna()
    from pymer4 import Lmer
    model = Lmer(formula_gen, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

def vendor_lmer_Feb12(fp='obj7_fMRI'):
    df = pickle_wrap(None, get_vendor_df, kwargs={'fp': fp,
                                                  'scrub': False,
                                                  'anat': True},
                     cache_dir='cache', easy_override=False)

    p_mem = get_memory_p()
    df_mem = get_df_p_x_p(p_mem, None, 'hc', fp=fp)
    df = df.merge(df_mem, on=['sn', 'obj'], how='left')

    for col in ['age', 'dd_vv', 'cross', 'dd', 'vv',
                'hit_hit', 'hc']: # 'con_hit', 'vis_hit',
        df[col] = stats.zscore(df[col], nan_policy='omit')

    iv = 'hit_hit'
    brain = 'vv'
    dvs = ['dd_vv', 'cross', 'dd', 'vv']
    for dv in dvs:
        formula_gen = f'{dv} ~ 1 + {brain}*{iv} + age*{iv} + ' \
                      f'(1 + {brain}*{iv} + age*{iv} | sn)'
        cols = get_formula_cols(df, formula_gen)
        df_vals = df[cols].dropna()
        from pymer4 import Lmer
        model = Lmer(formula_gen, data=df_vals)
        model.fit(REML=True, verbose=False, summary=False)
        print(model.summary())
        print('-'*10)
        print(f'{dv=}')
        print('-'*50)




def meta_corr_triangle():
    df = pickle_wrap(None, get_vendor_df, kwargs={'fp': 'obj7_fMRI',
                                                  'scrub': True,
                                                  'z_score': True},
                     cache_dir='cache', easy_override=True)

    sides = ['dd', 'vv', 'dv_ant', 'dv_pos']
    for dv in sides:
        df[dv] = stats.zscore(df[dv], nan_policy='omit')
    df['age_str'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    # df['age'] -= 1.5
    # print(df['age'].value_counts())
    # quit()
    # corr = df[sides].corr()

    # outliers = (df[sides].abs() > 5).any(axis=1)
    # df = df[~outliers]


    df.reset_index(inplace=True, drop=True)
    # print(f'{sides=}')
    # sides = ['dd', 'vv', 'rand']
    # df['rand'] = np.random.normal(size=len(df))
    # df['dd'] += df['rand'] * 100
    corr = df[sides].corr()
    print(corr)
    df[sides].dropna(inplace=True)
    # df['vendor'] = df['dd'] - df['vv']
    #
    # quit()
    sides_names = {'dd': 'Dorsal',
                   'vv': 'Ventral',
                   'dv_ant': '(Anterior)\nDorsal x Ventral',
                   'dv_pos': '(Posterior)\nDorsal x Ventral',
                   'age_str': 'Age',
                   'inc_str': 'inc_str'}
    df_names = df.rename(columns=sides_names)
    df_names = df_names[sides_names.values()]
    sns.set(font_scale=1.15)
    g = sns.pairplot(df_names,
                     # hue='Age',
                     kind="reg",
                     # palette = 'orange',
                     # palette={'YA': 'dodgerblue', 'OA': 'red'},
                     # palette={'YA': 'green', 'OA': 'magenta'},
                     plot_kws={'scatter_kws': {'alpha': .1,
                                               'color': 'orange'}},
                               # 'line_kws': {'color': 'orange'}},
                     diag_kws={'common_norm': False})
    for i in range(len(sides)):
        for j in range(len(sides)):
            if i == j:
                continue
            g.axes[i][j].set_xlim(-11, 11)
            g.axes[i][j].set_ylim(-11, 11)
    # for lh in g._legend.legendHandles:
    #     lh.set_alpha(1)
    #     lh._sizes = [50]
    plt.show()
    quit()


    ar_main = []
    ar_itr = []
    for i, v in enumerate(sides):
        row_main = []
        row_itr = []
        for j, w in enumerate(sides):
            if i == j:
                row_main.append('-')
                row_itr.append('-')
                continue
            # print(f'{v} x {w}')
            beta_main, p_main, beta_itr, p_itr = meta_corr(df, v, w)
            stars_main = get_stars(p_main, pad=True)
            stars_itr = get_stars(p_itr, pad=True)
            row_main.append(f'{beta_main:.2f} {stars_main}')
            row_itr.append(f'{beta_itr:.2f} {stars_itr}')
            print(f'{v} x {w}: {beta_main=:.2f}, {p_main=:.3f}| '
                  f'{beta_itr=:.2f}, {p_itr=:.3f}')
        ar_main.append(row_main)
        ar_itr.append(row_itr)
    df_main = pd.DataFrame(ar_main, columns=sides, index=sides)
    print(df_main)
    print('-'*50)
    df_itr = pd.DataFrame(ar_itr, columns=sides, index=sides)
    print(df_itr)

def meta_corr(df, dv, iv, random_slops=True):
    pd.set_option('display.precision', 4)
    np.set_printoptions(precision=4)
    formula_gen = '{dv} ~ 1 + dd + inc + brain_M + dv_ant + dv_pos + ' \
                  '{itr} + (1 + {itr} | sn)'
    formula_gen = formula_gen.replace(f'{dv} + ', '')
    itr = f'{iv}*age*inc_str'
    formula = formula_gen.format(dv=dv, itr=itr)

    from pymer4 import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())
    summary = model.coefs
    print('-----------')
    # print(summary)
    # beta_main = summary[iv].iloc['Estimate']
    # beta_itr = summary[itr.replace('*', ':')].iloc['Estimate']
    beta_main = summary.loc[iv, 'Estimate']
    p_main = summary.loc[iv, 'P-val']
    beta_itr = summary.loc[itr.replace('*', ':'), 'Estimate']
    p_itr = summary.loc[itr.replace('*', ':'), 'P-val']
    return beta_main, p_main, beta_itr, p_itr

def plot_meta_corr_matrix():
    df_vdr = pickle_wrap(None, get_vendor_df, kwargs={'fp': 'obj7_fMRI'},
                         cache_dir='cache', easy_override=False)
    cols = ['dd', 'vv', 'dv_ant', 'dv_pos']
    # TODO: for the DV_pos/ant x dd/vv, make sure that there are no overlapping
    #   edges
    corr = df_vdr[cols].corr()
    np.set_printoptions(edgeitems=10)
    np.set_printoptions(linewidth=200)
    ar_str = np.full((len(cols), len(cols)), '', dtype=object)
    for i, row in enumerate(corr.values):
        for j, r in enumerate(row):
            if r > .99999:
                ar_str[i][j] = '-'
                continue
            z = np.arctanh(r)
            z_std = 1 / np.sqrt(len(df_vdr) - 3)
            z_low = z - 1.96 * z_std
            z_high = z + 1.96 * z_std
            r_low = np.tanh(z_low)
            r_high = np.tanh(z_high)
            ar_str[i][j] = f'{r:.2f}'# ({r_low:.2f}, {r_high:.2f})'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df_main = pd.DataFrame(ar_str, columns=cols, index=cols)
    print(df_main)


    # TODO: maybe also investigate a limbic + PhG anatomical module



if __name__ == '__main__':
    # meta_corr_triangle()
    # plot_conn_matrix()
    # vendor_lmer()
    # vendor_lmer_Feb12()
    # get_trialwise_vendor()
    plot_meta_corr_matrix()