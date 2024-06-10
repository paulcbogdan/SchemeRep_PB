import os

from connRSA.old.conn_ISPC import run_settings_ISPC
from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.old.conn_plot import plot_pie_chart
from old_Apr6.corr_RSA_x_vendor import get_plain_df_sn

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from conn_report import report_results
from utils import pickle_wrap
import pandas as pd
import numpy as np

import scipy.stats as stats
import statsmodels.formula.api as smf


def pie_charts_ISPC(df, title):
    df = df[df['fp_idx'] == '0']

    df = df.groupby('img')[['conn_score', 'BOLD_score']].mean()

    df['conn_score'] /= df['conn_score'].std()
    df['BOLD_score'] /= df['BOLD_score'].std()

    mod = smf.ols(formula='conn_score ~ 1', data=df)
    res = mod.fit()
    conn_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{conn_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    # df_ = df.copy()
    # df['BOLD_score'] = stats.zscore(df['BOLD_score'])
    # df['BOLD_score'] -= 10
    mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    res = mod.fit()
    # print(res.summary())
    conn_itr_r = res.params['Intercept']
    # slope_bold = res.params['BOLD_score']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress out BOLD: {conn_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    print('-')

    # df['conn_score'] = df['conn_score'] - slope_bold * df['BOLD_score']
    # mod = smf.ols(formula='conn_score ~ 1', data=df)
    # res = mod.fit()
    # print(res.summary())
    # conn_itr_r = res.params['Intercept']
    # p = res.pvalues['Intercept']
    # z = res.tvalues['Intercept']
    # print(f'\tRegress out BOLD 2: {conn_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    # print('-')
    # quit()

    mod = smf.ols(formula='BOLD_score ~ 1', data=df)
    res = mod.fit()
    bold_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{bold_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='BOLD_score ~ 1 + conn_score', data=df)
    res = mod.fit()
    bold_itr_r = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress conn: {bold_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    plot_pie_chart([conn_itr_r, bold_itr_r], title)
    return

def lmer_ISPC(df):
    for fp_idx in range(4):
        df_ = df[df['fp_idx'] == str(fp_idx)]
        df_['mem'] = stats.zscore(df_['mem'])
        mod = smf.ols(formula='BOLD_score ~ 1 + mem', data=df_)
        res = mod.fit()
        print(res.summary())
        # quit()
    quit()
    mod = smf.ols(formula='conn_score ~ 0 + mem * fp_idx', data=df)
    res = mod.fit()
    print(res.summary())

    # df = df.groupby('img')[['conn_score', 'BOLD_score']].mean()
    # mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    # res = mod.fit()
    # print(res.summary())
    quit()

    formula = 'conn_score ~ 1 + BOLD_score + (1|fp_idx)'
    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())

    # scores_conn = results_conn['scores_by_ROI']
    # print(f'{scores_conn.shape=}')
    # scores_conn = np.array(scores_conn)
    # # if not ISPC:
    # scores_sep = results_bold_sep['scores_by_ROI']
    # scores_sep = np.array(scores_sep)
    # scores_comb = results_bold_com['scores_by_ROI']


    # print(df)
    # print(results_conn['scores_by_ROI'].shape)
    quit()

    # scores_by_ROI = np.array(results['scores_by_ROI'])
    # scores_by_fp = np.transpose(scores_by_ROI, (0, 2, 1))

    # df, ROI_cols = organize_df(results_conn, results_bold_sep,
    #                            results_bold_comb, do_networks, target_ROI,
    #                            ISPC=True)
    #
    # lmer_stats(df, ROI_cols)

def load_for_lmer_ISPC(four_tasks='8', conn='euc', combine_regions=False,
                       split=False, do_networks=0, trial_similarity='corr',
                       age=1):
    settings = locals().copy()
    dir_results = r'cache/conn_RSA'
    results_conn = pickle_wrap(run_settings_ISPC, None, kwargs=settings,
                               easy_override=False, cache_dir=dir_results)
    report_results(results_conn, do_lmer=True, ISPC=True)

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        settings['RDM_method'] = 'within_nan'

    settings['conn'] = 'BOLD'
    print('BOLD ' * 10)
    results_bold_comb = pickle_wrap(run_settings_ISPC, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results,
                                )
    report_results(results_bold_comb, do_lmer=True, ISPC=True)

    print('SEP ' * 10)
    settings['do_networks'] = False
    results_bold_sep = pickle_wrap(run_settings_ISPC, None,
                                   kwargs=settings, easy_override=True,
                                   verbose=1, cache_dir=dir_results)
    # report_results(results_bold_sep, do_lmer=True, ISPC=True)
    # quit()


    results_bold_sep = {}
    return results_conn, results_bold_comb, results_bold_sep


def run_lmer_PFC_ISPC():
    df_bhv, _ = get_plain_df_sn()
    df_bhv = df_bhv.groupby('obj')['hit_hit'].mean()
    mem_scores = list(df_bhv.values) * 4

    # else_cortical:
    # numpy.core._exceptions._ArrayMemoryError: Unable to allocate 49.8 GiB
    #   for an array with shape (56, 114, 114, 9180) and data type float64
    target_ROI = 'Occipital'
    target_ROI = 'Hipp'
    target_ROI = 'SFG'
    # target_ROI = 'MTL'



    do_networks = ROI2NETWORK[target_ROI]
    results_conn, results_bold_comb, results_bold_sep = (
        load_for_lmer_ISPC(four_tasks='7', conn='prod', combine_regions=False,
                           split=False, do_networks=do_networks,
                           trial_similarity='corr',
                           age='healthy'))

    ROI2network_idx = {'Occipital': (1, 0), 'Ventral': (1, 1), 'Dorsal': (1, 1),
                   'else_cortical': (2, 1),
                   'PFC': (16, 0),
                   'PFC_ACC': (14, 1), 'FP': (14, 2),
                   'perceptual': (17, 0),
                   'full_frontal': 11, 'full_frontal_CG': 11,
                   'MTL': (9, 0)}

    idx = ROI2network_idx[target_ROI][1]
    results_conn['scores_by_ROI'] = results_conn['scores_by_ROI'][:, idx, :]
    # plt.hist(results_conn['scores_by_ROI'].flatten())
    # plt.show()
    # quit()
    # print(results_conn['scores_by_ROI'][:, :].shape)
    # quit()
    results_bold_comb['scores_by_ROI'] \
        = results_bold_comb['scores_by_ROI'][:, idx, :]

    # networks = {'PFC': ['SFG', 'MFG', 'IFG', 'OrG'],
    #             'PFC_ACC': ['SFG', 'MFG', 'IFG', 'OrG', 'ACC'],
    #             'FP': ['MFG', 'IFG', 'IPL', 'SPL']}



    d = {'conn_score': np.reshape(results_conn['scores_by_ROI'], -1),
         'BOLD_score': np.reshape(results_bold_comb['scores_by_ROI'], -1),
         'fp_idx': list(range(4)) * 114,
         'img': list(range(114)) * 4,
         'mem': mem_scores}

    df = pd.DataFrame(d)
    df['fp_idx'] = df['fp_idx'].astype(str)

    title = f'ISPC: {target_ROI}'
    title = title.replace('PFC_ACC', 'Prefrontal')
    title = title.replace('MTL', 'Medial Temporal Lobe')
    pie_charts_ISPC(df, title)
    # df = df.groupby('img').mean()
    # df['BOLD_score'] = stats.zscore(df['BOLD_score'])
    # df['conn_score'] = stats.zscore(df['conn_score'])



if __name__ == '__main__':
    run_lmer_PFC_ISPC()
