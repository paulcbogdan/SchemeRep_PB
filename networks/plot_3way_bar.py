import os

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from old.modularity import get_partition_matrix, get_partition_cross
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap, stdize
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from vendor_analysis import get_module_cross_trialwise_z
from vendor_partitioning import get_vendor_partitions

os.chdir('C:\PycharmProjects_C\SchemeRep')


def conn_partition_3bar(fp='obj7_fMRI', thr=2.0, anat=True):
    # Age x Con x (Within/Between partitions)
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=anat)
    n_rois = sn_inc_activity.shape[2]
    matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]

    dd_flat = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_d_ant,
                                             trialwise=False)
    vv_flat = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_v_ant,
                                             trialwise=False)
    dv_ant = get_module_cross_trialwise_z(conn_trials, p_d_ant, p_v_ant,
                                            trialwise=False)
    dv_pos = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_pos,
                                            trialwise=False)

    flat_within = np.stack((dd_flat, vv_flat), axis=-1)
    agg_within = np.nanmean(flat_within, axis=-1)
    # print(agg_within.shape)
    # quit()
    flat_between = np.stack((dv_ant, dv_pos), axis=-1)
    agg_between = np.nanmean(flat_between, axis=-1)
    # print(agg_between.shape)

    # # print(dd_flat.shape)
    # # quit()
    # #
    # # print(f'{p_d_ant=}')
    # # print(f'{p_d_pos=}')
    # # print(f'{p_v_ant=}')
    # # print(f'{p_v_pos=}')
    #
    # # TODO: backward to forward for each module... ant to post
    # conn_dd = get_partition_matrix(sn_inc_conn, p_dorsal)
    # tridx_dd = np.tril_indices(conn_dd.shape[-1], k=-1)
    # flat_dd = conn_dd[:, :, tridx_dd[0], tridx_dd[1]]
    # # conn_dd = get_partition_cross(sn_inc_conn, p_dorsal_pos, p_dorsal_ant)
    # # flat_dd =  np.reshape(conn_dd, (conn_dd.shape[0], conn_dd.shape[1], -1))
    # conn_vv = get_partition_matrix(sn_inc_conn, p_ventral)
    # tridx_vv = np.tril_indices(conn_vv.shape[-1], k=-1)
    # flat_vv = conn_vv[:, :, tridx_vv[0], tridx_vv[1]]
    # # conn_vv = get_partition_cross(sn_inc_conn, p_vent_pos, p_vent_ant)
    # # flat_vv =  np.reshape(conn_vv, (conn_vv.shape[0], conn_vv.shape[1], -1))
    # flat_within = np.concatenate((flat_dd, flat_vv), axis=-1)
    # agg_within = np.nanmean(flat_within, axis=-1)
    # print(agg_within.shape)
    # quit()
    n_sn = agg_within.shape[0]
    vals = list(agg_within.T.reshape(-1))
    incs = ['Inc'] * n_sn + ['Con'] * n_sn
    ages = (['YA']*len(age2idxs[1]) + ['OA']*len(age2idxs[2]))*2
    wbs = ['Within'] * (2 * n_sn)
    subj_nums = list(range(n_sn)) * 2

    # # conn_dv = get_partition_cross(sn_inc_conn, p_dorsal, p_ventral)
    # # flat_dv = np.reshape(conn_dv, (conn_dv.shape[0], conn_dv.shape[1], -1))
    # # agg_between = np.nanmean(flat_dv, axis=-1)
    # conn_dv_ant = get_partition_cross(sn_inc_conn, p_d_ant, p_v_ant)
    # flat_dv_ant = np.reshape(conn_dv_ant, (conn_dv_ant.shape[0],
    #                                        conn_dv_ant.shape[1], -1))
    # conn_dv_pos = get_partition_cross(sn_inc_conn, p_d_pos, p_v_pos)
    # flat_dv_pos = np.reshape(conn_dv_pos, (conn_dv_pos.shape[0],
    #                                        conn_dv_pos.shape[1], -1))
    # flat_between = np.concatenate((flat_dv_ant, flat_dv_pos), axis=-1)
    # agg_between = np.nanmean(flat_between, axis=-1)

    vals += list(agg_between.T.reshape(-1))

    incs *= 2
    ages *= 2
    wbs += ['Between'] * (2 * n_sn)
    subj_nums *= 2

    print(f'{len(vals)=}, {len(incs)=}, {len(ages)=}, {len(wbs)=}')


    d = {'vals': vals, 'inc': incs, 'within_between': wbs,
                           'age': ages, 'subj_num': subj_nums}
    for key, l in d.items():
        print(f'{key=}, {len(l)=}')
    quit()
    df_agg = pd.DataFrame(d)

    df_agg.dropna(inplace=True)
    plot_sb_bars(df_agg)


def plot_sb_bars(df_agg):
    plot_params = {
        # 'data': df_agg,
        'y': 'vals',
        'x': 'inc',
        'hue': 'within_between',
        'hue_order': ['Within', 'Between'],
        'col': 'age',
        'kind': 'bar'
    }

    from statannotations.Annotator import Annotator

    g = sns.catplot(edgecolor="black", errcolor="black", errwidth=1.5,
                    capsize=0.1, height=4, aspect=.7, alpha=0.5,
                    ci="sd", data=df_agg, **plot_params)
    g.map(sns.stripplot, plot_params["x"], plot_params["y"],
          plot_params["hue"],
          hue_order=plot_params["hue_order"],  # order=plot_params["order"],
          palette=sns.color_palette(), dodge=True, alpha=0.6, ec='k',
          linewidth=1)

    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(0.9, 0.5), ncol=1,
        title=None, frameon=False,
    )

    pairs = [
        [('Inc', 'Between'), ('Con', 'Between')],
        [('Inc', 'Within'), ('Con', 'Within')],
    ]

    for name, ax in g.axes_dict.items():
        ax.set_ylabel('Connectivity')
        ax.set_xlabel('')

        # subset the table otherwise the stats were calculated on the whole dataset
        annot = Annotator(ax, pairs, **plot_params,
                          data=df_agg.loc[df_agg['age'] == name, :])
        annot.configure(test='t-test_paired', text_format='simple',
                        show_test_name=False,
                        # loc='inside',
                        verbose=2)
        # annot.apply_test().annotate()
        annot.apply_and_annotate()

    plt.tight_layout(rect=(0, 0, 0.85, 0.95))

    df_pivot = df_agg.pivot_table(index=['age', 'subj_num'],
                                  columns=['inc', 'within_between'],
                                  values='vals', aggfunc='mean')
    df_pivot['between_ef'] = df_pivot[('Con', 'Between')] - \
                             df_pivot[('Inc', 'Between')]
    df_pivot['within_ef'] = df_pivot[('Con', 'Within')] - \
                            df_pivot[('Inc', 'Within')]
    df_pivot['two_way'] = df_pivot['between_ef'] - df_pivot['within_ef']
    print('-' * 50)
    t_wit, p_wit = stats.ttest_ind(df_pivot.loc['OA', :]['within_ef'],
                                   df_pivot.loc['YA', :]['within_ef'])
    M_OA_wit = df_pivot.loc['OA', :]['within_ef'].mean()
    M_YA_wit = df_pivot.loc['YA', :]['within_ef'].mean()
    print(f'Within, age x congruency: '
          f'{t_wit=:.2f}, {p_wit=:.4f} | '
          f'Effects: {M_OA_wit=:.3f}, {M_YA_wit=:.3f}')
    t_three, p_three = stats.ttest_ind(df_pivot.loc['OA', :]['two_way'],
                                       df_pivot.loc['YA', :]['two_way'])

    t_bet, p_bet = stats.ttest_ind(df_pivot.loc['OA', :]['between_ef'],
                                   df_pivot.loc['YA', :]['between_ef'])
    M_OA_bet = df_pivot.loc['OA', :]['between_ef'].mean()
    M_YA_bet = df_pivot.loc['YA', :]['between_ef'].mean()
    print(f'Between, age x congruency: '
          f'{t_bet=:.2f}, {p_bet=:.4f} | '
          f'Effects: {M_OA_bet=:.3f}, {M_YA_bet=:.3f}')

    t_two_OA, p_two_OA = stats.ttest_1samp(df_pivot.loc['OA', :]['two_way'], 0)
    print(f'\t{t_two_OA=:.2f}, {p_two_OA=:.4f}')
    t_two_YA, p_two_YA = stats.ttest_1samp(df_pivot.loc['YA', :]['two_way'], 0)
    print(f'\t{t_two_YA=:.2f}, {p_two_YA=:.4f}')
    print(f'\t{t_three=:.2f}, {p_three=:.4f}')

    supt = f'Three way interaction: p = {p_three:.3f}\n'
    plt.suptitle(supt)
    plt.show()


if __name__ == '__main__':
    conn_partition_3bar()


