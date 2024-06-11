import os

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from atlas_utils import get_atlas
from old.modularity import get_partition_matrix, get_partition_cross
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap, stdize
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

from load_more import get_module_cross_trialwise_z
from vendor_partitioning import get_vendor_partitions
from statannotations.Annotator import Annotator

os.chdir(r'E:\PycharmProjects_E\SchemeRep')
pd.DataFrame.iteritems = pd.DataFrame.items  # fix: https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems

import statsmodels.formula.api as smf
from connsearch import print_list_stats

from warnings import filterwarnings
filterwarnings('ignore', category=SettingWithCopyWarning,)

np.float = float
np.bool = bool
np.int = int


def conn_partition_3bar(fp='obj7_fMRI', anat=True, weighted=False,
                        anat_ver=3):
    # Age x Con x (Within/Between partitions)
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')
    sns = [df['sn'].unique()[0] for df in df_sns]
    print(f'{sns=}')
    quit()

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=anat, weighted=weighted,
                              flip=True, thr=.9, scrub=False, anat_ver=anat_ver)
    n_rois = sn_inc_activity.shape[2]

    # matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
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

    dv_cross = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_ant,
                                            trialwise=False)
    vd_cross = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_d_ant,
                                            trialwise=False)

    flat_within = np.stack((dd_flat, vv_flat), axis=-1)
    agg_within = np.nanmean(flat_within, axis=-1)



    flat_between = np.stack((dv_ant, dv_pos), axis=-1)
    agg_between = np.nanmean(flat_between, axis=-1)


    n_sn = agg_within.shape[0]

    incs = []
    ages = []
    wbs = []
    subj_nums = []
    vals = []

    keys = ['Within', 'Between', 'dd', 'vv', 'dv_ant', 'dv_pos',
            'dv_cross', 'vd_cross']
    data = [agg_within, agg_between, dd_flat, vv_flat, dv_ant, dv_pos,
            dv_cross, vd_cross]
    for key, flat in zip(keys, data):
        incs += ['Inc'] * n_sn + ['Neu'] * n_sn + ['Con'] * n_sn
        ages += (['YA']*len(age2idxs[1]) + ['OA']*len(age2idxs[2]))*3
        wbs += [key] * (3 * n_sn)
        subj_nums += list(range(n_sn)) * 3
        vals += list(flat.T.reshape(-1))

    print(f'{len(vals)=}, {len(incs)=}, {len(ages)=}, {len(wbs)=}')

    d = {'vals': vals, 'inc': incs, 'within_between': wbs,
         'age': ages, 'sn': subj_nums}

    df_agg = pd.DataFrame(d)
    p = plot_con_vs_inc(df_agg)
    quit()

    ps = []
    for i in range(1000):
        for sn, df_sn in df_agg.groupby('sn'):
            df_agg.loc[df_agg['sn'] == sn, 'inc'] = (
                df_sn['inc'].sample(frac=1).values)

        p = plot_con_vs_inc(df_agg, skip_plot=True)
        ps.append(1 - p)
        print(f'{p=:.3f}')
        if len(ps) % 10 == 0 and len(ps) > 1:
            # print(f'{i=}')
            print_list_stats(ps)

def plot_con_vs_inc(df_agg, skip_plot=False):
    plot_params = {
        # 'data': df_agg,
        'y': 'vals',
        'x': 'inc',
        'hue': 'within_between',
        # 'hue_order': ['Within', 'Between'],
        'kind': 'bar'
    }

    cond_sets = [('Within', 'Between'),
                 ]
    # ('dd', 'dv_pos'),
    #                  ('vv', 'dv_ant'),

    # cond_sets = [('Within', 'Between')]
    plt.rcParams.update({'font.size': 20,
                         'font.sans-serif': 'Arial'})

    # plt.rcParams['font.sans-serif'] = 'Arial'

    # sns.set_theme(rc={'figure.figsize': (11.7, 8.27)})


    for cond_set in cond_sets:
        df_set = df_agg.loc[df_agg['within_between'].isin(cond_set)]
        # extreme sns (53 for inc between, 53 for inc within)
        df_set = df_set[df_set['sn'] != 53]
        for wb in ['Within', 'Between']:
            for sn in df_set['sn'].unique():
                # match = (df_set['sn'] == sn) # (df_set['within_between'] == wb) &
                # df_set.loc[match, 'vals'] -= df_set.loc[match, 'vals'].mean()

                match = (df_set['sn'] == sn) & (df_set['within_between'] == wb)
                df_set.loc[match, 'vals'] -= df_set.loc[match, 'vals'].mean()

        pd.set_option('display.max_rows', None)

        df_inc = df_set[df_set['inc'] == 'Inc']
        df_inc_w = df_inc[df_inc['within_between'] == 'Within'].reset_index()
        df_inc_b = df_inc[df_inc['within_between'] == 'Between'].reset_index()
        df_con = df_set[df_set['inc'] == 'Con']
        df_con_w = df_con[df_con['within_between'] == 'Within'].reset_index()
        df_con_b = df_con[df_con['within_between'] == 'Between'].reset_index()
        # t_within, p_within = stats.ttest_rel(df_inc_w['vals'],
        #                                         df_inc_b['vals'])
        df_set['inc_num'] = df_set['inc'].map({'Inc': -1, 'Neu': 0, 'Con': 1})
        df_set['sn_str'] = df_set['sn'].astype(str)
        # formula = 'vals ~ inc_num + sn_str'
        # # print(df_set)
        # model = smf.ols(formula=formula,
        #                 data=df_set[df_set['within_between'] == 'Between'])
        # res = model.fit()
        # print(res.summary())
        # quit()

        t_within, p_within = stats.ttest_1samp(df_inc_w['vals'] - df_con_w['vals'],
                                               0, nan_policy='omit')

        # print(df_set.groupby(['inc', 'within_between'])['vals'].mean())

        print(f'{t_within=:.3f}, {p_within=:.4f}')
        t_between, p_between = stats.ttest_rel(df_inc_b['vals'],
                                                df_con_b['vals'])
        print(f'{t_between=:.3f}, {p_between=:.4f}')

        t_anova, p_anova = stats.ttest_1samp(df_inc_w['vals'] - df_inc_b['vals'] -
                                             df_con_w['vals'] + df_con_b['vals'],
                                             0, nan_policy='omit')
        print(f'{t_anova=:.3f}, {p_anova=:.4f}')

        # print(df_set)
        # quit()

        df_set['PE'] = df_set['inc'].map({'Inc': 'High PE', 'Neu': 'Med. PE',
                                          'Con': 'Low PE'})
        g = sns.catplot(x=plot_params["x"], y=plot_params["y"],
                        hue=plot_params['hue'],
                        data=df_set[['inc', 'vals', 'within_between']],
                        kind='bar', ci=None,
                        # errorbar=None,
                        # errwidth=1.5,
                        edgecolor='k',
                        # capsize=0.1, height=4,
                        alpha=0.5, linewidth=.7,
                        # palette=sns.color_palette()
                        palette=['dodgerblue', 'red'],
                        height=5, aspect=0.8
                        )
        # plt.show()
        # quit()

        df_set['inc_num'] = df_set['inc'].map({'Inc': -1, 'Neu': 0, 'Con': 1})
        df_set['wb_num'] = df_set['within_between'].map(
            {'Within': -0.5, 'Between': 0.5,})

        # df_set = df_set[df_set['inc'] != 'Neu']
        df_set['sn_str'] = df_set['sn'].astype(str)

        formula = 'vals ~ inc_num*within_between + within_between*sn_str'
        # print(df_set)

        # df_set['vals'] = stats.zscore(df_set['vals'])
        model = smf.ols(formula=formula, data=df_set)
        res = model.fit()
        key = cond_set[0]
        p_reg = res.pvalues.iloc[-1]#f'inc_num:within_between[T.{key}]']
        if p_reg < 0.05:
            print(res.summary())
        if skip_plot:
            continue

        # sns.color_palette()

        g.map(sns.stripplot, x=plot_params["x"], y=plot_params["y"],
              hue=plot_params['hue'], alpha=.35,
              data=df_set[['inc', 'vals', 'within_between']],
              palette=['dodgerblue', 'red'], dodge=True, edgecolor='k',
              linewidth=0.7)
        # plt.legend([], [], frameon=False)

        if cond_set == ('Within', 'Between'):

            df_pivot = df_set.pivot_table(index=['sn'],
                                          columns=['inc', 'within_between'],
                                          values='vals', aggfunc='mean')
            df_pivot['between_ef'] = df_pivot[('Con', 'Between')] - \
                                     df_pivot[('Inc', 'Between')]
            df_pivot['within_ef'] = df_pivot[('Con', 'Within')] - \
                                    df_pivot[('Inc', 'Within')]

            t_itr, p_itr = stats.ttest_rel(df_pivot['between_ef'],
                                           df_pivot['within_ef'])
            supt = (f'Two-level [Inc/Con] x Direction: p = {p_itr:.3f}\n'
                    f'Three-level [Inc/Neu/Con] x Direction: p = {p_reg:.3f} ')
            plt.suptitle(supt)

            # sns.move_legend(
            #     g, "lower center",
            #     bbox_to_anchor=(0.9, 0.63), ncol=1,
            #     title=None, frameon=False,
            # )
        g._legend.remove()

        #     pairs = [
        #         [('Inc', 'Between'), ('Con', 'Between')],
        #         [('Inc', 'Within'), ('Con', 'Within')],
        #     ]
        # else:
        pairs = [[('Inc', c), ('Con', c)] for c in cond_set]
        # pairs += [[('Neu', c), ('Con', c)] for c in cond_set]

        # print(f'{pairs=}')
        ax = plt.gca()
                # subset the table otherwise the stats were calculated on the whole dataset
        # annot = Annotator(ax, pairs, **plot_params,
        #                   data=df_set)
        # annot.configure(test='t-test_paired', text_format='simple',
        #                 show_test_name=False, verbose=2)
        # # annot.apply_test().annotate()
        # annot.apply_and_annotate()
        plt.plot([-.5, 2.5], [0, 0], 'k', linewidth=.5)
        plt.xlim(-.5, 2.5)
        plt.ylim(-.11, .11)
        g.set_xticklabels(['High PE', 'Med. PE', 'Low PE'])
        plt.gca().spines['bottom'].set_visible(False)
        plt.xlabel('')
        plt.ylabel('Mean connectivity')
        plt.tight_layout()
        fp = r'result_pics/other/inc_vendor_FC.png'
        plt.savefig(fp, dpi=600)
        plt.show()
        # quit()
    return p_reg

def plot_four(df_agg):
    plot_params = {
        # 'data': df_agg,
        'y': 'vals',
        'x': 'inc',

        'col': 'age',
        'color': 'r',
        'kind': 'bar'
    }

    side2str = {'Between': 'Between', 'Within': 'Within',
                'dd': 'Within: Dorsal', 'vv': 'Within: Ventral',
                'dv_ant': 'Between: Anterior', 'dv_pos': 'Between: Posterior',
                'dv_cross': 'Between: P. Dorsal - A. Ventral',
                'vd_cross': 'Between: P. Ventral - A. Dorsal'}
    betweens = {'Between', 'dv_ant', 'dv_pos'}

    # sides = ['Between', 'Within']
    # for side in sides:
    #     df_side = df_agg.groupby('within_between')[side]

    for side, df_side in df_agg.groupby('within_between'):
        print(f'{side=}')
        # print(df_side[['inc', 'vals', 'age']])
        # quit()
        if side in betweens:
            color = 'orange'
        else:
            color = 'dodgerblue'
        plot_params['color'] = color
        # edgecolor="black", **plot_params, errcolor="black", errwidth=1.5,
        # capsize=0.1, height=4, aspect=.7, alpha=0.5, ci="sd",
        g = sns.catplot(x=plot_params["x"], y=plot_params["y"],
                        hue=plot_params['col'],
                        data=df_side[['inc', 'vals', 'age']],
                        kind='box')
        # g.map(sns.stripplot, plot_params["x"], plot_params["y"],
        #       # plot_params["hue"],
        #       # hue_order=plot_params["hue_order"],  # order=plot_params["order"],
        #       # palette=sns.color_palette(),
        #       # color=color,
        #       dodge=True, alpha=0.6, #ec='k',
        #       linewidth=1)

        YA_match = df_side['age'] == 'YA'
        OA_match = df_side['age'] == 'OA'
        inc_match = df_side['inc'] == 'Inc'
        con_match = df_side['inc'] == 'Con'
        dif_YA = df_side.loc[YA_match & con_match, 'vals'].values - \
                 df_side.loc[YA_match & inc_match, 'vals'].values
        dif_OA = df_side.loc[OA_match & con_match, 'vals'].values - \
                 df_side.loc[OA_match & inc_match, 'vals'].values
        t, p = stats.ttest_ind(dif_YA, dif_OA)
        title_itr = f'Age x Congruency: {t=:.2f}, {p=:.3f}'


        pairs = [
            ['Inc', 'Con']
            # [('Inc', 'Between'), ('Con', 'Between')],
            # [('Inc', 'Within'), ('Con', 'Within')],
        ]

        # for name, ax in g.axes_dict.items():
        #     ax.set_ylabel('Connectivity')
        #     ax.set_xlabel('')
        #
        #     # subset the table otherwise the stats were calculated on the whole dataset
        #     annot = Annotator(ax, pairs, **plot_params,
        #                       data=df_side.loc[df_side['age'] == name, :])
        #     annot.configure(test='t-test_paired', text_format='simple',
        #                     show_test_name=False,
        #                     # loc='inside',
        #                     verbose=2)
        #     # annot.apply_test().annotate()
        #     annot.apply_and_annotate()

        plt.suptitle(f'{side2str[side]}\n{title_itr}')
        plt.tight_layout(rect=(0, 0, 0.85, 1.0))
        plt.show()
        # quit()

def activity_partition_4bar(fp='obj7_fMRI', anat=False, ):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = (
        get_vendor_partitions(age='healthy', anat=anat, flip=True, thr=.9,
                              scrub=False))

    ps = [p_d_ant, p_d_pos, p_v_ant, p_v_pos]
    names = ['DP', 'DA', 'VP', 'VA']

    ps = [p_d_pos + p_v_pos, p_d_ant + p_v_ant]
    names = ['Posterior', 'Anterior']
    vals = []
    partitions = []
    sn = []
    incs = []
    for i in range(sn_inc_activity.shape[1]):
        for p, name in zip(ps, names):
            print(sn_inc_activity.shape)
            vals += list(np.nanmean(sn_inc_activity[:, i, p, :], axis=(1, 2)))
            partitions += [name] * len(sn_inc_activity)
            incs += [i] * len(sn_inc_activity)
            sn += list(range(len(sn_inc_activity)))

    df = pd.DataFrame({'vals': vals, 'partition': partitions, 'sn': sn,
                       'inc': incs})
    # for name in names:
    #     df.loc[df['partition'] == name, 'vals'] = (
    #         stats.zscore(df.loc[df['partition'] == name, 'vals']))
    df['inc'] = df['inc'].map({0: 'Inc', 1: 'Neu', 2: 'Con'})
    df = df[df['inc'] != 'Neu']
    # print(df)
    # df_pivot = df.pivot_table(index=['sn'],
    #                           columns=['inc', 'partition'],
    #                           values='vals', aggfunc='mean')
    # print(df_pivot)

    plot_params = {
        # 'data': df_agg,
        'y': 'vals',
        'x': 'inc',
        'hue': 'partition',
        'hue_order': names,
        'kind': 'bar'
    }

    print(df)

    g = sns.catplot(x=plot_params["x"], y=plot_params["y"],
                    hue=plot_params['hue'],
                    data=df,
                    kind='bar',  # ci='sd',
                    errwidth=1.5, edgecolor='k',
                    capsize=0.1, height=4, alpha=0.5, linewidth=.7,
                    palette=sns.color_palette())

    # g.map(sns.stripplot, plot_params["x"], plot_params["y"],
    #       plot_params["hue"],
    #       hue_order=plot_params["hue_order"],  # order=plot_params["order"],
    #       palette=sns.color_palette(), dodge=True, alpha=0.15,
    #       linewidth=1)

    pairs = [[('Inc', c), ('Con', c)] for c in names]
    # pairs += [[('Neu', c), ('Con', c)] for c in cond_set]
    plt.ylim(-1, 1)

    print(f'{pairs=}')
    ax = plt.gca()
    # subset the table otherwise the stats were calculated on the whole dataset
    annot = Annotator(ax, pairs, **plot_params,
                      data=df)
    annot.configure(test='t-test_paired', text_format='simple',
                    show_test_name=False, verbose=2)
    annot.apply_and_annotate()
    plt.plot([-.5, 2.5], [0, 0], 'k', linewidth=.5)

    plt.show()


if __name__ == '__main__':
    conn_partition_3bar()
    # activity_partition_4bar()


