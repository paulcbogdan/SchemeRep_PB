import os

from fluctuations import get_df_networks

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

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

def regress_out_FC_all(df, key='dd_vv'):
    formula = f'{key} ~ 1 + FC_all' #  + (1 | sn)
    cols = get_formula_cols(df, formula)
    for col in cols:
        df[col] = stats.zscore(df[col])
    df_cols = df[cols + ['sn']].dropna()
    model = smf.ols(formula=formula, data=df_cols)
    res = model.fit()
    print(res.summary())
    df[key] -= df['FC_all'] * res.params['FC_all']
    return df

def do_rs_t(fp='rs', base='dv_ant', exclude='va', seed='va'):
    df, vndr_cols = pickle_wrap(get_rs_vendor_df,
                                kwargs={'roiwise': True, 'zscore': True,
                                        'norm_std': True, 'YA_only': False,
                                        'high_var_confounds': False},
                                easy_override=False)

    # df, vndr_cols = pickle_wrap(get_vendor_df, None,
    #                             kwargs={'fp': 'obj7_fMRI',
    #                                     'scrub': False,
    #                                     'anat': True,
    #                                     'roiwise': False},
    #                             easy_override=False, cache_dir='cache')

    # df, networks = pickle_wrap(get_df_networks, kwargs={'fp': 'obj7_fMRI',
    #                                                     'norm_std': False,
    #                                                     'zscore': False},
    #                            easy_override=False)
    # #
    conn_keys = ['dd', 'vv',
                 'dv_ant', 'dv_pos',
                 'dpva', 'vpda',
                 ]
    #
    # df['da_dp'] = df['da'] + df['dp']
    # df['va_vp'] = df['va'] + df['vp']
    # print(df[['da_dp', 'va_vp']].corr())
    # quit()

    # print(df)

    # # 'dp',
    # print(df[['da', 'dp', 'va', 'vp']].corr())
    # print(np.array(df[['da', 'dp', 'va', 'vp']].corr()))
    # #
    # # print(df[['dv_ant', 'vv']].corr())
    # # # print(df[['dv_pos', 'dd']].corr())
    # #
    # quit()
    # #
    #
    # df['dd_dv_pos'] = df['dd'] * df['dv_pos']
    # conn_keys += ['dd_dv_pos']
    #
    # for key in conn_keys:
    #
    #     M = df[key].mean()
    #     SD = df[key].std()
    #     # M /= SD
    #
    #     print(f'{key}: {M=:.4f} ({SD:.4f})')
    #
    # quit()


    # sn 1232 is correlated inversely at r = -.98??????

    # df = df[df['dd'].abs() < 5]
    # df = df[df['vv'].abs() < 5]

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    df['age'] = df['sn'].map(lambda x: int(x[0]))
    # df = df[df['age'] > 1.5]

    bad_sns = {'116', '117', '130', '232'}
    df = df[~df['sn'].isin(bad_sns)]

    for sn, df_sn in df.groupby('sn'):
        t = list(range(len(df_sn)))
        df.loc[df_sn.index, 'rs_trial'] = t
        r, p = stats.spearmanr(df_sn['dd_vv'], df_sn['dv_dv'])
    #     # print(f'{r=:.3f}')
        if -.7 < r < -.5:
            r_, _ = stats.spearmanr(df_sn['dd'], df_sn['vv'])
            plt.title(f'{r=:.2f} ({sn}) [{r_=:.2f}]')

            plt.plot(df_sn['da'])
            plt.plot(df_sn['dp'])
            plt.show()
            quit()
    quit()
    #     plt.title(sn)
    #     plt.show()
    # quit()
        # quit()

    keys = ['dd_vv', 'FC_all', 'dp', 'dv_dv', 'dd', 'vv']
    for key in keys:
        add_prev(df, key, 'rs')
        add_prev(df, f'{key}_prev', 'rs')
        add_prev(df, f'{key}_prev_prev', 'rs')
        add_prev(df, f'{key}_prev_prev_prev', 'rs')


    df.dropna(subset=['dd_vv_prev_prev_prev'], inplace=True)
    # print(len(df))
    # formula = ('dd_vv ~ 1 + '
    #            'dd_vv_prev + dd_vv_prev_prev + dd_vv_prev_prev_prev')
    # model = smf.ols(formula=formula, data=df[cols])
    # res = model.fit()
    # print(res.summary())
    formula = ('dd_vv ~ dd_vv_prev + dd_vv_prev_prev + '
               'dd_vv_prev_prev_prev + dd_vv_prev_prev_prev_prev + (1 | sn)')

    formula = ('dd ~ dv_pos + (1 | sn)')

    cols = get_formula_cols(df, formula)
    print(df[cols])
    # print('------')
    # for col in cols:
    #     if col == 'sn': continue
    #
    #     df[col] = stats.zscore(df[col], nan_policy='omit')
    #
    # print(df[cols])
    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())


if __name__ == '__main__':
    do_rs_t()