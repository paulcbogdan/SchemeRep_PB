import pandas as pd

from analyze_rs import get_rs_vendor_df
from autocorr import add_prev, add_next
from dFC_control_FC import get_all_task_vendor
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
from sklearn import decomposition
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_FA(fp, anat=False):
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
                                    kwargs={'fp': fp, 'scrub': True,
                                            'anat': anat,
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
        # print(f'{sn}: {max_num_close}')
    print(f'Extremely low variance sns: {bad_sns}')
    return bad_sns

def plot_fluctuation(df, key='horz'):
    # add_prev(df, key, 'RS')
    np.set_printoptions(edgeitems=30, linewidth=100000)

    x = []
    y = []

    # df = df[df[key].abs() < center_bounding]
    # print(df.groupby('sn')[key].count().sort_values())
    # quit()



    # sns = df['sn'].unique()
    # df = df[df['sn'] == '217']
    # print(df[key])
    # quit()
    center_bounding = 0.1

    add_prev(df, key, 'RS')
    add_next(df, key, 'RS')

    df[f'dn_{key}'] = df[f'{key}_next'] - df[key]
    df[f'dp_{key}'] = df[key] - df[f'{key}_prev']
    add_prev(df, f'dp_{key}', 'RS')
    add_next(df, f'dp_{key}', 'RS')


    df.dropna(subset=[f'dn_{key}', f'dp_{key}_next',
                      f'dp_{key}_prev'], inplace=True)
    df = df[df[key].abs() < center_bounding]
    print(f'{len(df)=}')
    # print(df['sn'].value_counts())

    # r_test, p_test = stats.pearsonr(df[f'dp_{key}'], df[f'dp_{key}_prev'],)
    # print(f'{r_test=:.3f}, {p_test=:.3f}')
    # quit()

    # add_prev(df, f'd_{key}', 'RS')
    df.dropna(subset=[f'dp_{key}', f'dn_{key}'], inplace=True)
    h, _, _, _ = plt.hist2d(df[f'dp_{key}'], df[f'dn_{key}'],
                            range=((-3, 3), (-3, 3)), bins=21)


    r, p = stats.pearsonr(df[f'dp_{key}'], df[f'dn_{key}'])
    # print(h)
    plt.title(f'{r=:.3f}, {p=:.3f}')
    plt.xlabel('prev change')
    plt.ylabel('next change')
    plt.show()
    quit()


    prev_key = key
    for prev_depth in range(1, 11):
        add_prev(df, prev_key, 'RS')
        df0 = df[df[key].abs() < center_bounding]
        prev_key = f'{prev_key}_prev'
        x.extend([-prev_depth]*len(df0))
        y.extend(df0[prev_key].values)

        df1 = df[df[prev_key].abs() < center_bounding]
        x.extend([prev_depth]*len(df1))
        y.extend(df1[key].values)


    # print(len(x))
    # plt.hist(y, bins=20)
    # plt.show()
    h, _, _, _ = plt.hist2d(x, y, range=((-10, 11), (-3, 3)), bins=21)
    print(h)
    plt.show()
    quit()

    # print(x)
    # print(y)
    # quit()

    # df.dropna(subset=[f'{key}_prev', key], inplace=True)
    plt.hist2d(df[f'{key}_prev'], df[key], range=[(-3, 3), (-3, 3)], bins=20)



    plt.show()
    quit()



def do_FA(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor(anat=True, zscore=False)
    # df = df[df['task'] != 'RS']
    df = df[df['task'] == 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    bad_sns.add('217') # almost always near zero
    df = df[~df['sn'].isin(bad_sns)]
    df.dropna(subset=['da', 'dp', 'va', 'vp'], inplace=True)

    # df['dd_vv'] = df['dd'] + df['vv']
    # df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']
    df['diag_a'] = df['da'] - df['dp'] - df['va'] + df['vp']
    df['diag_p'] = -df['da'] + df['dp'] + df['va'] - df['vp']

    df['up'] = df['da'] + df['dp']
    df['down'] = df['va'] + df['vp']
    df['ant'] = df['da'] + df['va']
    df['pos'] = df['dp'] + df['vp']

    df['vert'] /= np.std(df['vert'])
    df['horz'] /= np.std(df['horz'])
    plot_fluctuation(df)
    quit()

    # cov = [[1., .09],
    #        [.09, 1.]]
    # df = pd.DataFrame()
    # n_samples = len(df)
    # n_samples = 100_000
    # df['vert'], df['horz'] = np.random.multivariate_normal([0, 0], cov,
    #                                                        n_samples).T

    # df['vert'] = np.random.normal(size=len(df['vert']))

    # df['rotate'] = np.degrees(np.arctan2(df['vert'], df['horz']))
    # plt.hist(df['rotate'], bins=36, range=(-180, 180))
    # plt.show()
    # quit()


    # print(f'{r_v_x_h=:.2f}')
    # quit()
    df = df[df['task'] == 'RS']




    cols = ['da', 'dp', 'va', 'vp']
    df = df.dropna(subset=cols)

    M_rsq = []
    for col in cols:
        formula = f'{col} ~ 0 + horz + vert'
        # formula = f'{col} ~ 1 + inc'

        # formula = f' ~ 1 + da + dp + va + vp'
        # formula = formula.replace(f' + {col}', '')
        # formula = col + formula
        model = smf.ols(formula=formula, data=df)
        res = model.fit()
        # print(res.summary())
        M_rsq.append(res.rsquared)
    M_rsq = np.mean(M_rsq)
    print(f'{M_rsq=:.3f}')


    for sn, df_sn in df.groupby('sn'):
        # ax = plt.figure().add_subplot(projection='3d')
        plt.plot(df_sn['horz'])
        # ax.plot(df_sn['horz'], df_sn['vert'])
        plt.show()
        print(f'{sn=}')

        df_sn.reset_index(drop=True, inplace=True)
        print(len(df_sn))
        plt.plot(df_sn['rotate'])
        plt.show()
        quit()

    N, bins, _ = plt.hist(df['rotate'], bins=36, range=(-180, 180))
    plt.show()
    quit()






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





if __name__ == '__main__':
    do_FA()


