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

def regress_out_FC_all(df):
    formula = 'dd_vv ~ 1 + FC_all' #  + (1 | sn)
    cols = get_formula_cols(df, formula)
    for col in cols:
        df[col] = stats.zscore(df[col])
    df_cols = df[cols + ['sn']].dropna()
    model = smf.ols(formula=formula, data=df_cols)
    res = model.fit()
    df['dd_vv'] -= df['FC_all'] * res.params['FC_all']
    return df

def do_rs_t(fp='rs', base='dv_ant', exclude='va', seed='va'):
    df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True,
                                                          'do_hemi': False,
                                                          'zscore': False,
                                'high_var_confounds': True},
                                easy_override=False)
    for sn, df_sn in df.groupby('sn'):
        t = list(range(206))
        df.loc[df_sn.index, 'rs_trial'] = t

    df['dd_vv'] = df['dd'] + df['vv']
    df['age'] = df['sn'].apply(lambda sn: int(str(sn)[0]))
    df = regress_out_FC_all(df)

    df_grp = df.groupby(['age', 'sn'])['dd_vv'].mean()
    plt.hist(df_grp.loc[1, :], label='YA', alpha=.75)
    plt.hist(df_grp.loc[2, :], label='OA', alpha=.5)


    t, p = stats.ttest_ind(df_grp.loc[1, :], df_grp.loc[2, :])
    plt.legend()
    plt.title(f'YA vs. OA: {t=:.2f}')
    plt.show()
    quit()



    keys = ['dd_vv', 'FC_all', 'dp', 'dv_dv', 'dd', 'vv']
    for key in keys:
        add_prev(df, key, 'rs')
        add_prev(df, f'{key}_prev', 'rs')
        add_prev(df, f'{key}_prev_prev', 'rs')

    df.dropna(subset=['dd_vv_prev_prev_prev'], inplace=True)
    print(len(df))
    formula = ('FC_all ~ 1 + '
               'FC_all_prev + FC_all_prev_prev + FC_all_prev_prev_prev + '
               'FC_all_prev')
    # formula = ('dp ~ 1 + '
    #            'dp_prev + dp_prev_prev + dp_prev_prev_prev')
    cols = get_formula_cols(df, formula)
    for col in cols:
        df[col] = stats.zscore(df[col])
    model = smf.ols(formula=formula, data=df[cols])
    res = model.fit()
    print(res.summary())


if __name__ == '__main__':
    do_rs_t()