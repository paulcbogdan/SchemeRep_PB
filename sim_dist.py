import numpy as np
import scipy.stats as stats

from old_Apr6.fluctuations import get_df_networks, partial_corr_df
from Utils.pickle_wrap_funcs import pickle_wrap
import os
import pandas as pd
os.chdir(r'H:\PycharmProjects_H\SchemeRep')


def sim_second_order_corr(r=-.5):
    xy = np.random.normal(0, 1, 1000)
    e = np.random.normal(0, 1, 1000)
    xz = r * xy + np.sqrt(1 - r**2) * e
    # print(xy)
    # quit()
    # plt.hist(xz)
    # plt.show()
    # quit()

    x = np.random.normal(0, 1, 1000)
    # e_y = np.random.normal(0, 1, 1000)
    y = xy / x

    # e_z = np.random.normal(0, 1, 1000)
    z = xz / x#x + np.sqrt(1 - r**2) * e_z

    # plt.scatter(y, z)
    # plt.show()

    r_yz, _ = stats.pearsonr(y, z)
    print(f'{r_yz=:.2f}')


def euc(a, b):
    # dist = distance.euclidean(a, b)
    dif = a - b
    dif = np.mean(np.abs(dif))
    # print()
    # print(f'{a=}')
    # print(f'{b=}')
    # print(dist)
    # quit()
    return dif

if __name__ == '__main__':
    # sim_second_order_corr()
    # quit()

    fp = 'obj7_fMRI'
    df, networks = pickle_wrap(get_df_networks, kwargs={'fp': fp,
                                                        'norm_std': False,
                                                        'zscore': True},
                               easy_override=False)

    # cov = [[1.0, -0.10, -0.28],
    #        [-0.10, 1.0, 0.21],
    #        [-0.28, -0.21, 1.0]]

    cov = np.array(df[['da', 'dp', 'va', 'vp']].corr())

    df['da_dp'] = df['da'] + df['dp']
    df['va_vp'] = df['va'] + df['vp']
    print(df[['da', 'dp', 'va', 'vp', ]].corr())
    # M = df['dv_pos'].mean()
    # print(f'{M=:.4f}')
    #
    # quit()



    # cov = [[1.0, -0.14, -0.48],
    #        [-0.14, 1.0, -0.21],
    #        [-0.48, -0.21, 1.0]]

    # cov[0][1] = 0
    # cov[0][2] = 0
    # cov[1][3] = 0
    # cov[2][3] = 0
    #
    # cov[1][0] = 0
    # cov[2][0] = 0
    # cov[3][1] = 0
    # cov[3][2] = 0
    # #
    # cov[2][1] = 0
    # cov[1][2] = 0

    # cov = np.zeros((4, 4))
    # # cov[0][3] = -0.5
    # # cov[3][0] = -0.5
    # # cov[1][2] = -0.5
    # # cov[2][1] = -0.5
    #
    # print(cov)

    # cov = np.zeros((4, 4))
    cov[np.diag_indices_from(cov)] = 1

    da, dp, va, vp = np.random.multivariate_normal([0, 0, 0, 0],
                                                   cov,
                                                   100_000).T

    # stats.pearsonr()

    dd = da * dp
    # M = np.mean(dd)
    vv = va * vp
    dv_ant = da * va
    dv_pos = dp * vp

    ar = np.array([dd, vv, dv_ant, dv_pos])
    print('-*- sim -*-')
    df_ar = pd.DataFrame(ar.T, columns=['dd', 'vv', 'dv_ant', 'dv_pos'])

    cov_ar = df_ar[['dd', 'vv', 'dv_ant', 'dv_pos']].corr()
    print(cov_ar)

    dd_vv = dd + vv
    dv_dv = dv_ant + dv_pos

    r, _ = stats.pearsonr(dd_vv, dv_dv)
    print(f'Sim: {r=:.3f}')
    print()
    print('----')

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    df_cols = df[['dd_vv', 'dv_dv']].dropna()
    r, _ = stats.pearsonr(df_cols['dd_vv'], df_cols['dv_dv'])
    print(f'Actual: {r=:.3f}')


    cov_network = df[['dd', 'vv', 'dv_ant', 'dv_pos']].corr()
    print(cov_network)
    print()
    print('-- control FC_all --')

    partial_corr_df(df, ['dd', 'vv', 'dv_ant', 'dv_pos',
                         'dd_vv', 'dv_dv'],
                    cov=['FC_all'])


    # partial_corr_df(df, ['dd', 'vv', 'dv_ant', 'dv_pos',
    #                      'dd_vv', 'dv_dv'],
    #                 cov=['FC_all', ]) # 'ad_no', 'pd_no',


    # print(x.shape)
    #
    # xy = x * y
    # yz = y * z
    # xz = x * z

    # r_xy_yz, _ = stats.pearsonr(xy, yz)
    # print(f'MAIN INTEREST: {r_xy_yz=:.2f}')
    # r_xy_xz, _ = stats.pearsonr(xy, xz)
    # print(f'{r_xy_xz=:.2f}')
    # r_xz_yz, _ = stats.pearsonr(xz, yz)
    # print(f'{r_xz_yz=:.2f}')


