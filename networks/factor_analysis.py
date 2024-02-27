import pandas as pd

from analyze_rs import get_rs_vendor_df
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
from sklearn import decomposition
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_FA(fp):
    if fp == 'rs':
        df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': False,
                                                              'do_hemi': False,
                                                              'zscore': True,
                                    'high_var_confounds': False},
                                    easy_override=True)
    else:
        df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                    kwargs={'fp': fp, 'scrub': False,
                                            'anat': True,
                                            'roiwise': True,
                                            'zscore': True},
                                    easy_override=True, cache_dir='cache')
    return df

def do_FA(fp='rs'):
    df = load_FA(fp)

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']

    # df['rotate'] = np.arctan(df['vert'] / df['horz'])

    # bad_sns = ['']
    # sns = df['sn'].unique()
    # df = df[df['sn'].isin(sns[:5])]

    df['rotate'] = np.degrees(np.arctan2(df['vert'], df['horz']))
    # df['rotate'] += 0.5

    df = df[(df['rotate'] > 44.5) & (df['rotate'] < 45.5)]
    df = df[df['sn'] == '138']
    # print(df['sn'].value_counts())
    print(df[['da', 'dp', 'va', 'vp', 'rotate']])
    plt.hist(df['da'])
    plt.show()

    quit()


    # df = df[[]]

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)



    N, bins, _ = plt.hist(df['rotate'], bins=360, range=(-180, 180))
    plt.show()
    print(df.groupby('sn')['rotate'].mean().sort_values())

    for n, bin in zip(N, bins):
        print(f'{bin=:.1f}, {n=:.0f}')

    quit()

    # df['vert'] = df['da'] + df['dp']
    # df['horz'] = df['va'] + df['vp']

    r_v_x_h, _ = stats.pearsonr(df['vert'], df['horz'])
    print(f'{r_v_x_h=:.2f}')

    cols = ['da', 'dp', 'va', 'vp']
    df = df.dropna(subset=cols)

    M_rsq = []
    for col in cols:
        formula = f'{col} ~ 0 + rotate'
        model = smf.ols(formula=formula, data=df)
        res = model.fit()
        # print(res.summary())
        M_rsq.append(res.rsquared)
    M_rsq = np.mean(M_rsq)
    print(f'{M_rsq=:.3f}')
    quit()

    pca = decomposition.PCA(n_components=2)
    X = df[cols].values
    pca.fit(X)
    first_two_components = sum(pca.explained_variance_ratio_[:2])
    print(f'{first_two_components=:.3f}')
    print(pca.explained_variance_ratio_)
    # print(pca.components_)

    df[['PCA1', 'PCA2']] = pca.transform(X)
    M_rsq_pca = []
    for col in cols:
        formula = f'{col} ~ 0 + PCA1 + PCA2'
        model = smf.ols(formula=formula, data=df)
        res = model.fit()
        # print(res.summary())
        M_rsq_pca.append(res.rsquared)
    M_rsq_pca = np.mean(M_rsq_pca)
    print(f'{M_rsq_pca=:.3f}')



    # print(trans.shape)

    # for sn, df_sn in df.groupby('sn'):
    #     pca = decomposition.PCA()
    #     X = df_sn[cols].dropna().values
    #     pca.fit(X)
    #     print(pca.explained_variance_ratio_)
    #     print(pca.components_)
    #
    #     pca.components_ = np.random.normal(size=pca.components_.shape)
    #     print(pca.explained_variance_ratio_)
    #
    #
    #     quit()
    #     # quit()






if __name__ == '__main__':
    do_FA()


