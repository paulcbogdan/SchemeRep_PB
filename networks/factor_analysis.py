import pandas as pd

from analyze_rs import get_rs_vendor_df
from autocorr import add_prev
from dFC_control_FC import get_all_task_vendor
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
                                    easy_override=False)
        # bad_sns = {'138'}
        # df = df[~df['sn'].isin(bad_sns)]
    else:
        df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                    kwargs={'fp': fp, 'scrub': False,
                                            'anat': True,
                                            'roiwise': False,
                                            'zscore': True},
                                    easy_override=False, cache_dir='cache')
    return df

def identify_extremely_low_variance_sn(df, key='da', limit=25):
    bad_sns = set()
    for sn, df_sn in df.groupby('sn'):
        vals = df_sn[key].values
        sd = np.std(vals)
        max_num_close = 0
        for val in vals:
            num_close = np.sum(np.abs(vals - val) < sd / 100)
            max_num_close = max(max_num_close, num_close)
        if max_num_close > limit:
            bad_sns.add(sn)
    print(f'Extremely low variance sns: {bad_sns}')
    return bad_sns

def do_FA(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor()
    # df = df[df['task'] != 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    df = df[~df['sn'].isin(bad_sns)]
    df.dropna(subset=['da', 'dp', 'va', 'vp'], inplace=True)

    # df['dd_vv'] = df['dd'] + df['vv']
    # df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']
    df['diag'] = df['da'] - df['dp'] - df['va'] + df['vp']

    df['vert'] /= np.std(df['vert'])
    df['horz'] /= np.std(df['horz'])

    cov = [[1., .09],
           [.09, 1.]]
    # df = pd.DataFrame()
    # n_samples = len(df)
    # n_samples = 100_000
    # df['vert'], df['horz'] = np.random.multivariate_normal([0, 0], cov,
    #                                                        n_samples).T

    # df['vert'] = np.random.normal(size=len(df['vert']))

    df['rotate'] = np.degrees(np.arctan2(df['vert'], df['horz']))

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)

    r_v_x_h, _ = stats.pearsonr(df['vert'], df['horz'])
    # print(f'{r_v_x_h=:.2f}')
    # quit()
    df = df[df['task'] == 'OBJ']
    add_prev(df, 'horz', 'RS')
    add_prev(df, 'vert', 'RS')
    add_prev(df, 'vert_prev', 'RS')


    # formula = f'horz ~ 1 + inc'
    # formula = f' ~ 1 + da + dp + va + vp'
    # formula = formula.replace(f' + {col}', '')
    # formula = col + formula
    # model = smf.ols(formula=formula, data=df)
    # res = model.fit()
    # print(res.summary())
    # quit()

    # df = df[df['horz'].abs() < 4]
    # df = df[df['vert'].abs() < 4]


    # for sn, df_sn in df.groupby('sn'):
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.plot(df_sn['horz'], df_sn['vert'], zs=list(range(len(df_sn))),
    #             )
    #     plt.show()
    #     print(f'{sn=}')

        # df_sn.reset_index(drop=True, inplace=True)
        # print(len(df_sn))
        # plt.plot(df_sn['rotate'])
        # plt.show()
        # quit()

    # N, bins, _ = plt.hist(df['rotate'], bins=36, range=(-180, 180))
    # plt.show()
    # quit()



    cols = ['da', 'dp', 'va', 'vp']
    df = df.dropna(subset=cols)

    M_rsq = []
    for col in cols:
        formula = f'{col} ~ 0 + horz + vert'
        # formula = f' ~ 1 + da + dp + va + vp'
        # formula = formula.replace(f' + {col}', '')
        # formula = col + formula
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


