import pandas as pd
from tqdm import tqdm

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
    df = df[df['task'] == 'RS']
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
    df[key] = stats.zscore(df[key])
    # plt.hist(df[key], bins=100)
    # plt.show()
    # quit()
    center_bounding = 0.25
    lower_bound = -.25
    upper_bound = .25

    add_prev(df, key, 'obj')
    add_next(df, key, 'obj')

    df[f'dn_{key}'] = df[f'{key}_next'] - df[key]
    df[f'dp_{key}'] = df[key] - df[f'{key}_prev']
    add_prev(df, f'dp_{key}', 'RS')
    add_next(df, f'dp_{key}', 'RS')


    df.dropna(subset=[f'dn_{key}', f'dp_{key}_next',
                      f'dp_{key}_prev'], inplace=True)
    # df = df[df[key].abs() < center_bounding]
    df = df[(lower_bound < df[key]) & (df[key] < upper_bound)]

    # print(f'{len(df)=}')
    # print(df['sn'].value_counts())
    # quit()

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

def is_valid_state(row):
    # if row['da'] > 0 and row['dp'] > 0 and row['vp'] < 0 and row['va'] < 0:
    #     return 1
    # elif row['da'] < 0 and row['dp'] < 0 and row['vp'] > 0 and row['va'] > 0:
    #     return 0
    # else:
    #     return 0

    if row['da'] > 0 and row['dp'] > 0 and row['vp'] < 0 and row['va'] < 0:
        return 'A'
    elif row['da'] < 0 and row['dp'] < 0 and row['vp'] > 0 and row['va'] > 0:
        return 'B'
    elif row['da'] < 0 and row['dp'] > 0 and row['va'] < 0 and row['vp'] > 0:
        return 'C'
    elif row['da'] > 0 and row['dp'] < 0 and row['va'] > 0 and row['vp'] < 0:
        return 'D'
    else:
        return 'E'

def state_test(df):
    df = df[df['task'] == 'OBJ']

    # cols = ['da', 'dp', 'vp', 'va']
    # for sn, df_sn in tqdm(df.groupby('sn'), desc='M_drop'):
    #     for col in cols:
    #         new_vals = []
    #         # df_sn.reset_index(inplace=True)
    #         # print(df.loc[df_sn.index, col])
    #         for i in range(len(df_sn)):
    #             idx = df_sn.index[i]
    #             vals_sans_i = df_sn[col].drop(idx)
    #             M = np.mean(vals_sans_i)
    #             SD = np.std(vals_sans_i)
    #             new_vals.append((df_sn[col].iloc[i] - M) / SD)
    #         df.loc[df_sn.index, col] = new_vals

    # df[cols] = np.random.normal(size=(len(df), len(cols)))
    # print(df[cols])
    # df['M'] = df[cols].mean(axis=1)
    # for col in cols:
    #     df[col] -= df['M']
    df['match'] = df.apply(is_valid_state, axis=1)
    add_prev(df, 'match', 'obj')
    add_prev(df, 'horz', 'obj')
    add_prev(df, 'vert', 'obj')
    # print(df['rs_trial'])
    # quit()
    df.dropna(subset=['horz_prev', 'match_prev'], inplace=True)


    # print(df['match'].value_counts(normalize=True))
    cnt = df['match_prev'].value_counts()
    print(cnt)
    cnt_pair = df[['match_prev', 'match']].value_counts()
    cnt_pair /= cnt
    # print(cnt_pair.sort_index())

    r_horz_x_prev, _ = stats.pearsonr(df['horz'], df['horz_prev'])
    print(f'{r_horz_x_prev=:.3f}')

    states = ['A', 'B', 'C', 'D', 'E']
    for state in states:
        # df = df[df['match'] != 'E']

        df[f'is_{state}'] = (df['match'] == state).astype(int)
        # print(df[f'is_{state}'].value_counts(normalize=True))
        # quit()
        formula = f'is_{state} ~ 1 + inc'

        df['abs_horz'] = np.abs(df['horz'])

        formula = f'dd_vv ~ 1 + horz'
        # df.dropna(subset=[f'is_{state}', 'inc'], inplace=True)
        # print(df[['inc', f'is_{state}']])
        # quit()
        model = smf.ols(formula=formula, data=df)
        res = model.fit()
        print(res.summary())
        print(f'{state=}')
        print('\n'*5)
        quit()


    # p_match = df['match'].mean()
    # print(f'{p_match=:.2f}')
    quit()

def do_FA(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor(anat=True, zscore=True)
    # df = df[df['task'] != 'RS']
    # df = df[df['task'] == 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    bad_sns.add('138')
    bad_sns.add('217') # almost always near zero for Horz
    bad_sns.add('110') # almost always nere zero for Vendor
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

    formula = fr'da ~ 1 + inc'
    model = smf.ols(formula=formula, data=df)
    res = model.fit()
    print(res.summary())
    quit()

    # df = df[df['task'] == 'OBJ']
    # print(df[['up', 'down', 'ant', 'pos']].corr())
    # print(df[['da', 'dp', 'va', 'vp']].corr())
    # quit()



    df['vert'] /= np.std(df['vert'])
    df['horz'] /= np.std(df['horz'])

    df['dd_vv'] = df['dd'] + df['vv']
    # df['dd_vv'] = stats.zscore(df['dd_vv'], nan_policy='omit')
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    # df['dv_dv'] = stats.zscore(df['dv_dv'], nan_policy='omit')
    df['vendor'] = df['dd_vv'] - df['dv_dv']
    df['alt'] = df['dd'] - df['vv'] + df['dv_ant'] - df['dv_pos']
    # plot_fluctuation(df)
    state_test(df)
    quit()

    # df = df[df['task'] == 'OBJ']
    r_ant_pos, _ = stats.pearsonr(df['ant'], df['pos'])
    print(f'{r_ant_pos=:.3f}')
    r_up_down, _ = stats.pearsonr(df['up'], df['down'])
    print(f'{r_up_down=:.3f}')




    cols = ['da', 'dp', 'va', 'vp']
    # cols = ['dd', 'vv', 'dv_ant', 'dv_pos']
    # cols = ['horz', 'vert']
    df = df.dropna(subset=cols)

    # flipper = [1, -1] * ((len(df) + 2) // 2)
    # df['flipper'] = flipper[:len(df)]

    M_rsq = []
    for col in cols:
        formula = f'{col} ~ 1 + horz + vert'
        # formula = f'{col} ~ 1 + inc'

        # formula = f' ~ 1 + da + dp + va + vp'
        # formula = formula.replace(f' + {col}', '')
        # formula = col + formula
        model = smf.ols(formula=formula, data=df)
        res = model.fit()
        print(res.summary())
        M_rsq.append(res.rsquared)
    M_rsq = np.mean(M_rsq)
    print(f'{M_rsq=:.3f}')
    quit()


    # for sn, df in df.groupby('sn'):
    #     # ax = plt.figure().add_subplot(projection='3d')
    #     plt.plot(df['horz'])
    #     # ax.plot(df_sn['horz'], df_sn['vert'])
    #     plt.show()
    #     print(f'{sn=}')
    #
    #     df.reset_index(drop=True, inplace=True)
    #     print(len(df))
    #     plt.plot(df['rotate'])
    #     plt.show()
    #     quit()
    #
    # N, bins, _ = plt.hist(df['rotate'], bins=36, range=(-180, 180))
    # plt.show()
    # quit()
    # quit()

    pca = decomposition.PCA(n_components=2)
    X = df[cols].values
    pca.fit(X)
    first_two_components = sum(pca.explained_variance_ratio_[:2])
    print(f'{first_two_components=:.3f}')
    print(pca.explained_variance_ratio_)
    print(pca.components_)

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


