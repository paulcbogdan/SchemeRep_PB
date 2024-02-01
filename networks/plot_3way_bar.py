import os

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from modularity import get_main_partitions, get_partition_matrix, get_partition_cross
from network_based_statistic import get_stats_graphs
from network_funcs import load_FC_for_Lifu
from utils import pickle_wrap
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C:\PycharmProjects_C\SchemeRep')

def get_vendor_partitions_(sn_inc_conn, age2idxs, age=2, thr=.95, flip=False):
    M1_graph, SD1_graph, SE1_graph, N1_graph, t2_graph, p1_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[age], 0, :, :],
                         sn_inc_conn[age2idxs[age], 1, :, :])
    z2_graph = -z2_graph if flip else z2_graph
    age2str = {1: 'YA', 2: 'OA'}
    dir_out = f'{age2str[age]}3_ttest_modules_thr{thr}_flip{flip}'
    partitions, matrix_mask = \
        get_main_partitions(z2_graph, coords=None, plot=False, threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out=dir_out)
    return partitions, matrix_mask

def get_vendor_partitions(sn_inc_conn, age2idxs, age=2, thr=.95, flip=False):
    fp = f'cache/{age}_ttest_modules_thr{thr}_flip{flip}.pkl'
    partitions, matrix_mask = \
        pickle_wrap(fp, lambda: get_vendor_partitions_(sn_inc_conn=sn_inc_conn,
                                                       age2idxs=age2idxs,
                                                       age=age, thr=thr,
                                                       flip=flip))
    atlas = get_atlas()
    coords = atlas['coords']
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos

def anterior_posterior_split(p_dorsal, coords):
    p_dorsal_ys = [coords[i][1] for i in p_dorsal]
    p_dorsal_y_med = np.median(p_dorsal_ys)
    p_dorsal_ant = [i for i in p_dorsal if coords[i][1] < p_dorsal_y_med]
    p_dorsal_pos = [i for i in p_dorsal if coords[i][1] >= p_dorsal_y_med]
    return p_dorsal_ant, p_dorsal_pos

def conn_partition_3bar(fp='obj7_fMRI', thr=2.0):
    # Age x Con x (Within/Between partitions)
    kwargs = {'fp': fp,
              # 'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    # for age in [2, 1]:
    #     M1_graph, SD1_graph, SE1_graph, N1_graph, t2_graph, p1_graph, z2_graph = \
    #         get_stats_graphs(sn_inc_conn[age2idxs[age], 0, :, :],
    #                          sn_inc_conn[age2idxs[age], 1, :, :])
    #     flip_t = True
    #     z2_graph = -z2_graph if flip_t else z2_graph
    #     age2str = {1: 'YA', 2: 'OA'}
    #     dir_out = f'{age2str[age]}3_ttest_modules_thr{thr}_flip{flip_t}'
    #     partitions, matrix_mask = \
    #         get_main_partitions(z2_graph, coords=None, plot=True, threshold=thr,
    #                             fn_str='', overlapping=False,
    #                             dir_out=dir_out)

    atlas = get_atlas()
    coords = atlas['coords']

    p_dorsal, p_ventral, p_dorsal_ant, p_dorsal_pos, p_vent_ant, p_vent_pos = \
        get_vendor_partitions(sn_inc_conn, age2idxs)

    # TODO: backward to forward for each module... ant to post
    conn_dd = get_partition_matrix(sn_inc_conn, p_dorsal)
    tridx_dd = np.tril_indices(conn_dd.shape[-1], k=-1)
    flat_dd = conn_dd[:, :, tridx_dd[0], tridx_dd[1]]
    # conn_dd = get_partition_cross(sn_inc_conn, p_dorsal_pos, p_dorsal_ant)
    # flat_dd =  np.reshape(conn_dd, (conn_dd.shape[0], conn_dd.shape[1], -1))
    conn_vv = get_partition_matrix(sn_inc_conn, p_ventral)
    tridx_vv = np.tril_indices(conn_vv.shape[-1], k=-1)
    flat_vv = conn_vv[:, :, tridx_vv[0], tridx_vv[1]]
    # conn_vv = get_partition_cross(sn_inc_conn, p_vent_pos, p_vent_ant)
    # flat_vv =  np.reshape(conn_vv, (conn_vv.shape[0], conn_vv.shape[1], -1))
    flat_within = np.concatenate((flat_dd, flat_vv), axis=-1)
    agg_within = np.nanmean(flat_within, axis=-1)
    n_sn = agg_within.shape[0]

    vals = list(agg_within.T.reshape(-1))
    incs = ['Inc'] * n_sn + ['Con'] * n_sn
    ages = (['YA']*len(age2idxs[1]) + ['OA']*len(age2idxs[2]))*2
    wbs = ['Within'] * (2 * n_sn)

    # conn_dv = get_partition_cross(sn_inc_conn, p_dorsal, p_ventral)
    # flat_dv = np.reshape(conn_dv, (conn_dv.shape[0], conn_dv.shape[1], -1))
    # agg_between = np.nanmean(flat_dv, axis=-1)
    conn_dv_ant = get_partition_cross(sn_inc_conn, p_dorsal_ant, p_vent_ant)
    flat_dv_ant = np.reshape(conn_dv_ant, (conn_dv_ant.shape[0],
                                           conn_dv_ant.shape[1], -1))
    conn_dv_pos = get_partition_cross(sn_inc_conn, p_dorsal_pos, p_vent_pos)
    flat_dv_pos = np.reshape(conn_dv_pos, (conn_dv_pos.shape[0],
                                           conn_dv_pos.shape[1], -1))
    flat_between = np.concatenate((flat_dv_ant, flat_dv_pos), axis=-1)
    agg_between = np.nanmean(flat_between, axis=-1)

    vals += list(agg_between.T.reshape(-1))
    incs *= 2
    ages *= 2
    wbs += ['Between'] * (2 * n_sn)

    print(f'{len(vals)=}, {len(incs)=}, {len(ages)=}, {len(wbs)=}')

    df_agg = pd.DataFrame({'vals': vals, 'inc': incs, 'within_between': wbs,
                           'age': ages})

    df_agg.dropna(inplace=True)

    plot_params = {
        # 'data': df_agg,
        'y': 'vals',
        'x': 'inc',
        'hue': 'within_between',
        'hue_order': ['Within', 'Between'],
        'col': 'age',
        'kind': 'bar'
    }
    # g = sns.catplot(**plot_params)
    # g.set_axis_labels('', 'Connectivity (z)')
    # plt.xlabel('')
    # plt.show()r
    # quit()

    # ax = plt.gca()
    # pairs=[
    #     (('OA', 'Inc', 'Within'), ('YA', 'Con', 'Within')),
    # ],
    # pairs = [[('Inc', 'Within'), ( 'Con', 'Within')]]
    from statannotations.Annotator import Annotator
    # annotator = Annotator(ax, pairs, **plot_params)
    # annotator.configure(test="Mann-Whitney").apply_and_annotate()

    g = sns.catplot(edgecolor="black", errcolor="black", errwidth=1.5,
                    capsize=0.1, height=4, aspect=.7, alpha=0.5,
                    ci="sd", data=df_agg, **plot_params)
    g.map(sns.stripplot, plot_params["x"], plot_params["y"],
          plot_params["hue"],
          hue_order=plot_params["hue_order"], #order=plot_params["order"],
          palette=sns.color_palette(), dodge=True, alpha=0.6, ec='k',
          linewidth=1)

    sns.move_legend(
        plt.gca(), "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    # plt.xlabel('')
    # pairs = [
    #     (("Male", "Yes"), ("Male", "No")),
    #     (("Female", "Yes"), ("Female", "No"))
    # ]
    #
    pairs=[
        [('Inc', 'Within'), ('Con', 'Within')],
        [('Inc', 'Between'), ('Con', 'Between')],
        # [('Inc', 'Within'), ('Con', 'Within')],
    ]

    for name, ax in g.axes_dict.items():
        ax.set_ylabel('Connectivity')
        ax.set_xlabel('')

        # subset the table otherwise the stats were calculated on the whole dataset
        annot = Annotator(ax, pairs, **plot_params,
                          data=df_agg.loc[df_agg['age'] == name, :])
        annot.configure(test='t-test_paired', text_format='simple',
                        show_test_name=False,
                        #loc='inside',
                        verbose=2)
        # annot.apply_test().annotate()
        annot.apply_and_annotate()


    plt.suptitle('TOAST TOAST')
    plt.tight_layout()

    # import statannot
    #
    # statannot.add_stat_annotation(
    #     ax,
    #     data=df_agg,
    #     x='inc',
    #     y='vals',
    #     hue='within_between',
    #     box_pairs=[
    #         # ('Inc', 'Con'),
    #         (('Inc', 'Within'), ('Con', 'Within')),
    #     ],
    #     test="t-test_ind",
    #     text_format="star",
    #     loc="outside",
    # )

    plt.show()



# df = sns.load_dataset("titanic")
# print(df[['age', 'class', 'sex']])
# quit()

if __name__ == '__main__':
    conn_partition_3bar()


