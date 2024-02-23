from analyze_rs import get_rs_vendor_df
from atlas_utils import get_atlas
from old.plot_gen import my_plot_surf
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
import scipy.stats as stats
import matplotlib.pyplot as plt

from vendor_partitioning import get_vendor_partitions
import statsmodels.formula.api as smf
from utils import get_formula_cols
import numpy as np
from autocorr import add_prev


def do_rs_t(fp='rs', base='dv_ant', exclude='va', seed='va'):

    # base='vv', exclude='va', seed='va'

    # if fp == 'rs':
    df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True,
                                                          'do_hemi': False,
                                'high_var_confounds': False},
                                easy_override=False)
    for sn, df_sn in df.groupby('sn'):
        t = list(range(206))
        df.loc[df_sn.index, 'rs_trial'] = t


    # print(df['sn'].value_counts())
    #
    # df['sn'] = df['sn'].astype(str)
    # formula = ('dd ~ 1 + da * dp + sn + FC_all ')
    # cols = get_formula_cols(df, formula)
    #
    # # model = smf.ols(formula=formula, data=df[cols])
    # # res = model.fit()
    # # print(res.summary())
    #
    # formula = ('dd ~ da*dp +'
    #            '(1 | sn)')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # plt.gca().scatter(df['dp'], df['da'], df['dd'], alpha=.1)
    # ax.set_zlabel('conn')
    # # plt.scatter(df['dd'], df['da'])
    #
    # plt.show()
    # quit()

    # from pymer4 import Lmer
    # cols = get_formula_cols(df, formula)
    # for col in cols:
    #     if col == 'sn': continue
    #     df[col] = stats.zscore(df[col])
    # df_vals = df[cols].dropna()
    # model = Lmer(formula, data=df_vals)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    # quit()

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    formula = 'dd_vv ~ 1 + FC_all' #  + (1 | sn)
    cols = get_formula_cols(df, formula)
    for col in cols:
        df[col] = stats.zscore(df[col])
    df_cols = df[cols + ['sn']].dropna()

    model = smf.ols(formula=formula, data=df_cols)
    res = model.fit()


    df.dropna(subset=['dd_vv', 'dv_dv'], inplace=True)

    df['dd_vv'] -= df['FC_all'] * res.params['FC_all']

    # formula = ('dv_dv ~ 1 + FC_all') #  + (1 | sn)
    # model = smf.ols(formula=formula, data=df_cols)
    # res = model.fit()
    # df['dv_dv'] -= df['FC_all'] * res.params['FC_all']

    # r, p = stats.pearsonr(df['dd_vv'], df['dv_dv'])

    keys = ['dd_vv', 'FC_all', 'dp']
    for key in keys:
        add_prev(df, key, 'rs')
        add_prev(df, f'{key}_prev', 'rs')
        add_prev(df, f'{key}_prev_prev', 'rs')

    add_prev(df, 'dv_dv', 'rs')

    df.dropna(subset=['dd_vv_prev_prev_prev'], inplace=True)
    print(len(df))
    # quit()

    formula = ('dd_vv ~ 1 + dd_vv_prev + dd_vv_prev_prev + dd_vv_prev_prev '
               '+ FC_all + FC_all_prev + FC_all_prev_prev')

    # formula = ('vv ~ 1 + vp * dp + vp * va')
    cols = get_formula_cols(df, formula)
    for col in cols:
        df[col] = stats.zscore(df[col])
    model = smf.ols(formula=formula, data=df[cols])
    res = model.fit()
    print(res.summary())

    # r, p = stats.pearsonr(df['dv_dv'], df['dv_dv_prev'])
    # print(f'Autocorrelation: {r=}, {p=}')


    # for sn, df_sn in df.groupby('sn'):
    #     t = list(range(206))
    #     df.loc[df_sn.index, 't'] = t
    #     plt.plot(t, df_sn['dd'])
    #     plt.plot(t, df_sn['vv'])
    #     # plt.plot(t, df_sn['dv_dv'])
    #     plt.show()
    #     quit()


if __name__ == '__main__':
    do_rs_t()