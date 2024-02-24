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

def do_OA_vs_YA_vendor(fp='rs', base='dv_ant', exclude='va', seed='va'):
    df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True,
                                                          'do_hemi': False,
                                                          'zscore': False,
                                'high_var_confounds': True},
                                easy_override=False)
    for sn, df_sn in df.groupby('sn'):
        t = list(range(206))
        df.loc[df_sn.index, 'rs_trial'] = t

    df['dd_vv'] = df['dd'] + df['vv']
    # df['dd_vv'] = df['da'] * df['dp']
    df['age'] = df['sn'].apply(lambda sn: int(str(sn)[0]))

    key = 'dd_vv'
    df = regress_out_FC_all(df, key=key)

    df_grp = df.groupby(['age', 'sn'])[key].mean()
    plt.hist(df_grp.loc[1, :], label='YA', alpha=.75)
    plt.hist(df_grp.loc[2, :], label='OA', alpha=.5)

    t, p = stats.ttest_ind(df_grp.loc[1, :], df_grp.loc[2, :])
    plt.legend()
    plt.title(f'YA vs. OA: {t=:.2f}, {p=:.3f}, {key}')
    plt.show()
    quit()

if __name__ == '__main__':
    do_OA_vs_YA_vendor()