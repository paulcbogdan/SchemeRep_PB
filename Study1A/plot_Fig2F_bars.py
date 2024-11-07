import os
import pathlib

from Study1A.modularity_funcs import get_FC_between_ROIs

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from Study1A.load_Study1A_funcs import load_FC
from Utils.pickle_wrap_funcs import pickle_wrap
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions

pd.DataFrame.iteritems = pd.DataFrame.items  # fix: https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems

import statsmodels.formula.api as smf

from warnings import filterwarnings

filterwarnings('ignore', category=SettingWithCopyWarning, )

np.float = float
np.bool = bool
np.int = int


def prep_plot_Fig2F_bars():
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stats.zscore(sn_inc_activity, axis=3, nan_policy='omit')

    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]

    dd_flat = get_FC_between_ROIs(conn_trials, p_d_pos, p_d_ant, trialwise=False)
    vv_flat = get_FC_between_ROIs(conn_trials, p_v_pos, p_v_ant, trialwise=False)
    dv_ant = get_FC_between_ROIs(conn_trials, p_d_ant, p_v_ant, trialwise=False)
    dv_pos = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_pos, trialwise=False)
    dv_cross = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_ant, trialwise=False)
    vd_cross = get_FC_between_ROIs(conn_trials, p_v_pos, p_d_ant, trialwise=False)

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
        ages += (['YA'] * len(age2idxs[1]) + ['OA'] * len(age2idxs[2])) * 3
        wbs += [key] * (3 * n_sn)
        subj_nums += list(range(n_sn)) * 3
        vals += list(flat.T.reshape(-1))

    d = {'vals': vals, 'inc': incs, 'within_between': wbs,
         'age': ages, 'sn': subj_nums}

    df_agg = pd.DataFrame(d)
    plot_Fig2F_bars(df_agg)


def plot_Fig2F_bars(df_agg, mean_norm=True):
    plot_params = {
        'y': 'vals',
        'x': 'inc',
        'hue': 'within_between',
        'kind': 'bar'
    }

    cond_sets = [('Within', 'Between'), ]

    plt.rcParams.update({'font.size': 21,
                         'font.sans-serif': 'Arial'})
    print(df_agg['within_between'].unique())

    for wb, df_wb in df_agg.groupby('within_between'):
        df_inc = df_wb[df_wb['inc'] == 'Inc'].reset_index()
        df_con = df_wb[df_wb['inc'] == 'Con'].reset_index()
        t, p = stats.ttest_rel(df_inc['vals'], df_con['vals'])
        print(f'{wb}: {t=:.3f}, {p=:.4f}')

    if mean_norm:
        for cond_set in cond_sets:
            df_set = df_agg.loc[df_agg['within_between'].isin(cond_set)]
            for wb in ['Within', 'Between']:
                for sn in df_set['sn'].unique():
                    match = (df_set['sn'] == sn) & (df_set['within_between'] == wb)
                    df_set.loc[match, 'vals'] -= df_set.loc[match, 'vals'].mean()

        pd.set_option('display.max_rows', None)

        df_set['inc_num'] = df_set['inc'].map({'Inc': -1, 'Neu': 0, 'Con': 1})
        df_set['sn_str'] = df_set['sn'].astype(str)

        df_set['PE'] = df_set['inc'].map({'Inc': 'High PE', 'Neu': 'Med. PE',
                                          'Con': 'Low PE'})
        g = sns.catplot(x=plot_params["x"], y=plot_params["y"],
                        hue=plot_params['hue'],
                        data=df_set[['inc', 'vals', 'within_between']],
                        kind='bar', ci=68,
                        edgecolor='k',
                        alpha=0.7, linewidth=.7,
                        errwidth=1.2,
                        capsize=0.05,
                        palette=['dodgerblue', 'red'],
                        height=5, aspect=0.8
                        )

        df_set['inc_num'] = df_set['inc'].map({'Inc': -1, 'Neu': 0, 'Con': 1})
        df_set['wb_num'] = df_set['within_between'].map(
            {'Within': -0.5, 'Between': 0.5, })

        df_set['sn_str'] = df_set['sn'].astype(str)

        formula = 'vals ~ inc_num*within_between + within_between*sn_str'
        model = smf.ols(formula=formula, data=df_set)
        res = model.fit()
        p_reg = res.pvalues.iloc[-1]

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
        g._legend.remove()

        plt.plot([-.5, 2.5], [0, 0], 'k', linewidth=.5)
        plt.xlim(-.5, 2.5)
        plt.tick_params(axis='x', which='both', bottom=False, top=False)

        g.set_xticklabels(['High PE', 'Med. PE', 'Low PE'])
        plt.gca().spines['bottom'].set_visible(False)
        plt.xlabel('')
        plt.ylabel('Mean connectivity')
        plt.tight_layout()
        M_norm_str = '_no_M_norm' if not mean_norm else ''
        fp = fr'result_pics/Fig2/Fig2F_PE_x_Conn_bars{M_norm_str}.png'
        plt.savefig(fp, dpi=600)
        plt.show()


if __name__ == '__main__':
    prep_plot_Fig2F_bars()
