# see fluctuations.py as of Monday June 24, should produce r = -.18 for dv_dv vs. dd_dd

from analyze_rs import prep_conn_ps
from atlas_utils import get_atlas
from load_more import load_a, get_module_cross_trialwise_z, get_dfs_conn_trials, load_resting_data, load_act_conn
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import get_df_networks, partial_corr_df
from utils import pickle_wrap, stdize
from collections import defaultdict

from old_Apr6.vendor_lmers import get_module_trialwise_z
from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg
from EEG_fMRI.EEG_fMRI_test import get_hrf
from scipy import signal
import matplotlib.pyplot as plt
from vendor_partitioning import get_vendor_partitions
from functools import partial
import os
os.chdir(r'H:\PycharmProjects_H\SchemeRep')
import statsmodels.formula.api as smf

from numpy.fft import fft, ifft, ifftshift

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

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

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

    for sn, df_sn in df.groupby('sn'):
        if sn != '126': continue
        cols = ['dd_vv', 'dv_dv', 'pd_no', 'ad_no', 'av_no', 'pv_no',
                'dpva', 'vpda',
                'dd', 'vv', 'dv_ant', 'dv_pos',
                'horz', 'vert', 'abs_horz', 'abs_vert']
        df_sn = df_sn[cols].dropna().reset_index()

        formula = '~ 1 + pd_no + ad_no + av_no + pv_no'
        # formula += ' + dpva + vpda'
        # formula += '+ abs_horz + abs_vert'
        # df['dd_vv'] = df['abs_vert']
        # df['dv_dv'] = df['abs_horz']


        mod0 = smf.ols(formula='dd_vv' + formula,
                       data=df_sn)
        res0 = mod0.fit()
        # print(res0.summary())
        # quit()
        y0 = res0.resid

        mod1 = smf.ols(formula='dv_dv' + formula,
                       data=df_sn)
        res1 = mod1.fit()
        y1 = res1.resid

        # mod0 = smf.ols(formula='dv_ant' + formula,
        #                data=df_sn)
        # res0 = mod0.fit()
        # y0 = res0.resid
        #
        # mod1 = smf.ols(formula='dv_pos' + formula,
        #                data=df_sn)
        # res1 = mod1.fit()
        # y1 = res1.resid


        # y0 = df_sn['pd_no']
        # y1 = df_sn['av_no']



        r, p = stats.spearmanr(y0, y1) # going pearsonr to spearmanr leads to a big drop
        rs.append(r)
        # continue

        # y0, y1 = df_sn['dd_vv'], df_sn['dv_dv']
        #
        # y0 = stats.rankdata(y0)
        # y1 = stats.rankdata(y1)

        dif = np.abs(y0 - y1)
        #
        # plt.plot(y0)
        # plt.show()

        HRF = get_hrf()[1:-7]
        # s = np.sum(HRF)
        # HRF /= s
        y0_ = wiener_deconvolution(y0, HRF)
        y0_ = stats.zscore(y0_)
        y1_ = wiener_deconvolution(y1, HRF)
        y1_ = stats.zscore(y1_)
        dif_ = np.abs(y0_ - y1_)


        # y0_ = signal.deconvolve(y0, HRF)[0]
        # print(y0_)
        # plt.plot(y0_)
        # plt.show()
        #
        # quit()

        # print(len(y0))
        # print(len(y0_))

        r_, p_ = stats.spearmanr(y0_, y1_)
        # rs.append(r_)

        autocorr_y0 = pg.corr(y0, y0.shift(1))['r'][0]
        autos0.append(autocorr_y0)
        autocorr_y1 = pg.corr(y1, y1.shift(1))['r'][0]
        autos1.append(autocorr_y1)
        # continue
        autocorr_dif = pg.corr(dif_, dif_.shift(1))['r'][0]
        autos_dif.append(autocorr_dif)
        print(f'{r=:.2f}, {r_=:.2f}')
        print(f'\t{autocorr_dif=:.2f}')
        # continue

        num_crosses = np.sum(np.diff(np.sign(y0 - y1)) != 0)

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
            quit()

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

