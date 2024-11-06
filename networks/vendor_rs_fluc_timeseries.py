# see fluctuations.py as of Monday June 24, should produce r = -.18 for dv_dv vs. dd_dd

from old_Apr6.fluctuations import get_df_networks
from utils import pickle_wrap

from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg
from EEG_fMRI.EEG_fMRI_test import get_hrf
import matplotlib.pyplot as plt
import os
os.chdir(r'H:\PycharmProjects_H\SchemeRep')
import statsmodels.formula.api as smf

from numpy.fft import fft, ifft


def wiener_deconvolution(signal, kernel):
    lambd = .01
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
    return pd.Series(deconvolved)

def produce_Fig5A(fp='rs_medium', anat_ver=3, combine_regions=False):
    df, networks = pickle_wrap(get_df_networks,
                               kwargs={'fp': fp,
                                       'norm_std': False,
                                       'zscore': True,
                                       'anat_ver': anat_ver,
                                       'combine_regions': combine_regions},
                               easy_override=False)
    #
    # print(list(df.columns))
    # quit()

    # ignore all FutureWarnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)


    df['da_dp'] = df['da'] + df['dp']
    df['va_vp'] = df['va'] + df['vp']

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['abs_vert'] = np.abs(df['vert'])
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']
    df['abs_horz'] = np.abs(df['horz'])

    df['dd'] = df['da'] * df['da']
    df['vv'] = df['va'] * df['va']
    # df['dv_ant'] = df['da'] * df['va']
    # df['dv_pos'] = df['dp'] * df['vp']
    # df['pd_no'] = df['da'] * df['no']
    # df['ad_no'] = df['dp'] * df['no']
    # df['av_no'] = df['va'] * df['no']
    # df['pv_no'] = df['vp'] * df['no']

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    # df['dd_x_vv'] = df['dv_ant'] * df['dv_pos']
    # r, p = stats.spearmanr(df['dd_x_vv'], df['abs_horz'])
    # print(f'{r=:.2f}, {p=:.5f}')
    # quit()

    # print(df[networks].corr())

    # print(df[['da', 'dp', 'va', 'vp', 'no']].corr())
    # quit()

    # r, p = stats.spearmanr(df['dd_vv'], df['abs_vert'])
    # print(f'{r=:.2f}, {p=:.5f}')
    # r, p = stats.spearmanr(df['dv_dv'], df['abs_horz'])
    # print(f'{r=:.2f}, {p=:.5f}')
    # r, p = stats.spearmanr(df['dd_vv'], df['dv_dv'])
    # print(f'{r=:.2f}, {p=:.5f}')
    # quit()

    rs = []
    autos0 = []
    autos1 = []
    autos_dif = []
    min_r = 0
    max_crosses = 0
    HRF = get_hrf()[1:-7]

    for sn, df_sn in df.groupby('sn'):
        # if sn != '126': continue
        cols = ['dd_vv', 'dv_dv', 'pd_no', 'ad_no', 'av_no', 'pv_no',
                'dpva', 'vpda',
                'dd', 'vv', 'dv_ant', 'dv_pos',
                'horz', 'vert', 'abs_horz', 'abs_vert',
                'da', 'dp', 'va', 'vp', 'no_no']

        # for col in cols:
        #     df_sn[col] = wiener_deconvolution(df_sn[col], HRF).values
        #     df_sn[col] = stats.zscore(df_sn[col])
            # has_nan = df_sn[col].isnull().sum()
            # print(f'{col}, {has_nan}')
            # if has_nan:
            #     print(df_sn[col])
            #     quit()

        df_sn = df_sn[cols].dropna().reset_index()
        sim_80_power_implied_low(beta=1.8)

        mod1 = smf.ols(formula='dv_dv' + formula,
                       data=df_sn)
        res1 = mod1.fit()
        y1 = res1

        y0 = df_sn['dv_pos']
        y1 = df_sn['dv_ant']
        y0 = df_sn['dd_vv']
        y1 = df_sn['dv_dv']
        r, p = stats.pearsonr(y0, y1) # going pearsonr to spearmanr leads to a big drop
        rs.append(r)
        continue
        dif = np.abs(y0 - y1)


        # s = np.sum(HRF)
        # HRF /= s
        # y0_ = wiener_deconvolution(y0, HRF)
        # y0_ = stats.zscore(y0_)
        # y1_ = wiener_deconvolution(y1, HRF)
        # y1_ = stats.zscore(y1_)
        # dif_ = np.abs(y0_ - y1_)
        # r_, p_ = stats.spearmanr(y0_, y1_)
        # rs.append(r_)
        # y0 = np.abs(y0)
        # y1 = np.abs(y1)
        # y0 = df['dd_vv']#)#*df['dp'])# + np.abs(df['dp'])
        # y1 = df['dv_dv']#)#*df['dp'])# + np.abs(df['vp'])
        # y0 = np.abs(df['da'] - df['va'])
        # y1 = np.abs(df['dp'] + df['vp'] - df['da'] - df['va'])
        # y0 = y0 # - np.abs(y1)# + np.abs(y0 - y1)
        autocorr_y0 = pg.corr(y0, y0.shift(1))['r'][0]
        autos0.append(autocorr_y0)
        autocorr_y1 = pg.corr(y1, y1.shift(1))['r'][0]
        autos1.append(autocorr_y1)
        autocorr_dif = pg.corr(dif, dif.shift(1))['r'][0]
        # autos_dif.append(autocorr_dif)
        # print(f'{r=:.2f}, {r_=:.2f}')
        # print(f'\t{autocorr_dif=:.2f}')
        # continue

        # num_crosses = np.sum(np.diff(np.sign(y0 - y1)) != 0)

        # num_crosses = np.sum(np.diff(np.sign(df_sn['vp'])) != 0)
        # y0 = df_sn['vp']
        # y1 = df_sn['dp']

        # print(np.diff(np.sign(y0 - y1)))
        # print(y0 - y1)
        # print(np.sign(y0 - y1))
        # quit()
        # autos_dif.append(205 / num_crosses)
        continue

        plt.rcParams.update({'font.size': 40,
                             'font.sans-serif': 'Arial'})
        if r < -.3:
            min_r = r
        # if num_crosses > max_crosses:
        #     max_crosses = num_crosses
        #     plt.figure(figsize=(20, 12))
            plt.figure(figsize=(12, 6.9))

            linewidth = 2.5

            plt.plot([0, 100], [0, 0], color='black', linewidth=1.5)

            plt.plot(y0[:101], linewidth=linewidth, label='Posterior-Anterior',
                     color='dodgerblue')
            plt.plot(y1[:101], linewidth=linewidth, label='Ventral-Dorsal',
                     color='red')
            plt.plot(y0[:101], linewidth=linewidth, label='Posterior-Anterior',
                     color='dodgerblue', alpha=0.5)
            # plt.legend(frameon=False, ncol=22)

            # plt.title(f'{sn}: {r=:.2f} & {r_=:.2f} | '
            #           f'dd_vv: {autocorr_y0:.2f}, '
            #           f'dv_dv: {autocorr_y1:.2f}, '
            #           f'dif: {autocorr_dif:.2f}, '
            #           f'{num_crosses=}',)
            plt.gca().spines[['bottom', 'right', 'top']].set_visible(False)
            plt.gca().spines['left'].set_linewidth(2.5)
            # set yticks width to 2.5
            plt.gca().yaxis.set_tick_params(width=linewidth, length=10)
            plt.gca().xaxis.set_tick_params(width=linewidth, length=10)

            plt.ylabel('Time-varying\nconnectivity', fontsize=36, labelpad=15)
            plt.xlabel('Volume', labelpad=15)
            plt.yticks([-2, 0, 2])
            plt.xlim(0, 100)

            plt.tight_layout()
            plt.show()
            # quit()

            # plt.figure(figsize=(20, 12))
            # plt.plot(y0_, linewidth=1.5, label='dd_vv', color='green')
            # plt.plot(y1_, linewidth=1.5, label='dv_dv', color='purple')
            # plt.tight_layout()
            # plt.show()
            #
            # plt.plot(dif_)
            # plt.show()

    # plt.plot(sorted(rs)[::-1])
    # plt.show()

    M_rs = np.mean(rs)
    SE_rs = stats.sem(rs)
    M_low = M_rs - SE_rs * 1.96
    M_high = M_rs + SE_rs * 1.96
    print(f'{M_rs=:.3f} [{M_low:.3f}, {M_high:.3f}], {SE_rs=:.3f}')

    M_autos0 = np.mean(autos0)
    SE_autos0 = stats.sem(autos0)
    M_autos1 = np.mean(autos1)
    SE_autos1 = stats.sem(autos1)
    M_autos_dif = np.mean(autos_dif)
    SE_autos_dif = stats.sem(autos_dif)
    print(f'\t{M_autos0=:.3f} ± {SE_autos0:.3f}')
    print(f'\t{M_autos1=:.3f} ± {SE_autos1:.3f}')
    print(f'\t{M_autos_dif=:.3f} ± {SE_autos_dif:.3f}')

    t, p = stats.ttest_1samp(rs, 0)
    print(f'One sample: {t=:.3f}, {p=:.10f}')
    #
    # quit()


if __name__ == '__main__':
    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)

    produce_Fig5A()
    # get_dfs_conn_trials(fp='obj7_fMRI', single=False)
    # analyze_networks()

