import os

from colorama import Fore

os.environ['R_HOME'] = r'C:\Users\Paul\anaconda3\envs\py312\Lib\R'

import warnings

import numpy as np
import pandas as pd
from scipy import stats as stats

from conn_utils import get_BNA_ROIs



def report_results(results, do_lmer=False, ISPC=False):
    # print_settings(results['settings'])
    if 'plotting' in results['settings'] and \
            results['settings']['plotting'] in ['regions', 'edges']:
        setting_plotting = results['settings']['plotting']
        print(f'Can\'t report_results for plotting={setting_plotting}')
        return
    scores = np.array(results['scores'])
    sizes = np.array(results['sizes'])
    # if scores.shape[1] == 246:
    #     print('\t Results printing for BOLD is not yet implemented')
    #     return

    if len(results['keys']) != scores.shape[1]:
        if results['settings']['conn'] in ['BOLD', 'BOLD_avg']:
            assert scores.shape[1] == 246, f'BAD BOLD {scores.shape=}'
            results['keys'] = get_BNA_ROIs()
            warnings.warn('For BOLD, setting keys to 246 BNA ROIs')
        else:
            raise ValueError(f'{len(results["keys"])=} != {scores.shape=}')

    if ISPC:
        scores_by_ROI = np.array(results['scores_by_ROI'])
        scores_by_fp = np.transpose(scores_by_ROI, (0, 2, 1))

    else:
        scores_by_ROI = np.array(results['scores_by_ROI'])
        scores_by_fp = np.nanmean(scores_by_ROI, axis=3)

    # quit()
    ts = np.full(scores_by_fp.shape[2], np.nan)
    ts_by_fp = np.full(scores_by_fp.shape[1:], np.nan)
    for j, ROI in enumerate(results['keys']):
        ROI_scores_by_fp = scores_by_fp[:, :, j]
        M_by_fp = np.nanmean(ROI_scores_by_fp, axis=0)

        # print(f'{M_by_fp=}')
        SD_by_fp = np.nanstd(ROI_scores_by_fp, axis=0)
        # print(ROI_scores_by_fp)
        # quit()
        N_by_fp = [len(ROI_scores_by_fp[~np.isnan(ROI_scores_by_fp[:, k])]
                    ) for k in range(ROI_scores_by_fp.shape[1])]
        # N_by_fp = len(ROI_scores_by_fp[~np.isnan(ROI_scores_by_fp)])
        SE_by_fp = SD_by_fp / np.sqrt(N_by_fp)
        # print(f'{SD_by_fp=}')
        # print(f'{N_by_fp=}')
        # print(f'{SE_by_fp=}')
        t_by_fp = M_by_fp / SE_by_fp
        t_by_fp_str = '['
        for k, t in enumerate(t_by_fp):
            t_by_fp_str += f'{t:.2f}, '
            ts_by_fp[k, j] = t
        t_by_fp_str = t_by_fp_str[:-2]
        t_by_fp_str += ']'

        ROI_scores = scores[:, j]
        M = np.nanmean(ROI_scores)
        SD = np.nanstd(ROI_scores)
        N = len(ROI_scores[~np.isnan(ROI_scores)])
        SE = SD / np.sqrt(N)
        t = M / SE
        if not do_lmer: ts[j] = t
        p = stats.t.sf(np.abs(t), N - 1)
        M_size = np.nanmean(sizes[:, j])

        # if 'FuG' not in ROI:
        #     ts[j] = 0
        # ts[j] = 0
        # if 'FuG_L_3_1'  in ROI:
        #     ts[j] = 2.5
        # if 'FuG_R_3_1'  in ROI:
        #     ts[j] = 2.5

        if do_lmer and not ISPC:
            df_ROI_as_d = {'conn_score': [], 'sn': [], 'fp': []}
            scores_fp_all = scores_by_ROI[:, :, j, :]
            for sn_i in range(scores_fp_all.shape[0]):
                for fp_k in range(scores_fp_all.shape[1]):
                    for d in scores_fp_all[sn_i, fp_k, :]:
                        df_ROI_as_d['conn_score'].append(d)
                        df_ROI_as_d['sn'].append(sn_i)
                        df_ROI_as_d['fp'].append(fp_k)
            df_ROI_as_d = pd.DataFrame(df_ROI_as_d)
            df_ROI_as_d.dropna(inplace=True)
            if not len(df_ROI_as_d):
                lmer_result_str = '\tLmer: all NaN'
            else:
                from pymer4.models import Lmer
                formula = 'conn_score ~ 1 + (1|sn) + (1|fp)'
                model = Lmer(formula, data=df_ROI_as_d)
                try:
                    model.fit(REML=True, verbose=False, summary=False)
                    summary = model.coefs
                    lmer_t = summary['T-stat'].loc['(Intercept)']
                    ts[j] = lmer_t
                    lmer_p = summary['P-val'].loc['(Intercept)']
                    lmer_result_str = f'\tLmer: t={lmer_t:.2f}, p={lmer_p:.3f}'
                except:
                    lmer_result_str = '\tLmer: fit failed'
        else:
            lmer_result_str = ''

        # if p < .05:
        print(f'{ROI} ({M_size:.1f}), t[{N - 1}]={t:.2f}, p={p:.3f}, '
              f'{t_by_fp_str} {lmer_result_str}')

    return ts, ts_by_fp


def lmer_stats(df, ROI_cols):
    from pymer4.models import Lmer
    # conn_sess_ERS = ['2', ] # '4', '5'
    # df = df[df['fp_idx'].isin(conn_sess_ERS)]
    # df = df[df['fp_idx'].isin(['1'])]
    #
    # n_sn = df['sn'].nunique()
    #
    # df_M = df.groupby('sn')['conn_score'].mean()
    # M_score = df_M.mean()
    # SD_score = df_M.std()
    # # print(np.sum(~np.isnan(df_M)))
    # # quit()
    # N_score = np.sum(~np.isnan(df_M))
    # SE_score = SD_score / np.sqrt(N_score)
    # t_score = M_score / SE_score
    #
    # print(f'Single FP: {M_score=:.3f}, {SD_score=:.3f}, {SE_score=:.3f}, '
    #       f'{t_score=:.3f}')
    #
    # cols = ['ROI', 'sn', 'conn_score', 'hit_hit', 'inc', 'vis_hit', 'con_hit',
    #         'inc_str', 'per_inc_str', 'fp_idx', 'BOLD_score'] + \
    #          ROI_cols
    # df = df[cols]
    # df.dropna(inplace=True)
    # df['vis_hit'] = df['vis_hit'].astype(int)
    # df['con_hit'] = df['con_hit'].astype(int)
    #
    #
    #
    # if len(df['fp_idx'].unique()) > 1:
    #     formula = 'conn_score ~ 1 + BOLD_score + (1|sn) + (1|fp_idx)'
    #     print('tteet')
    # else:
    #     formula = 'conn_score ~ 1 + BOLD_score + (1|sn)'
    #     print('toast')
    # # quit()
    #
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    #
    # print('-' * 120)
    # formula = ('conn_score ~ 1 + BOLD_score + ' +
    #            ' + '.join(ROI_cols) + '+  (1|sn)')# + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    #
    # formula = ('BOLD_score ~ 1 + ' +
    #            ' + '.join(ROI_cols) + '+  (1|sn)')# + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())

    # df['BOLD_score'] = df['conn_score']
    cols_keep = ['ROI_M', 'BOLD_score', 'conn_score', 'sn',
                 'fp_idx', 'hit_hit', 'con_hit', 'vis_hit']



    # plt.hist(df['ROI_M'])



    # df['ROI_M'] = df[ROI_cols].apply(rsum, axis=1)

    df['ROI_M'] = df[ROI_cols].mean(axis=1)





    mem_cols = ['hit_hit', 'con_hit', 'vis_hit']
    df[mem_cols] = df[mem_cols].astype(int)

    formula = ('BOLD_score ~ 1 + ROI_M + (1 + ROI_M |sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('conn_score ~ 1 + BOLD_score + ROI_M + '
               '(1 + BOLD_score + ROI_M | sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())
    quit()

    formula = ('hit_hit ~ 1 + BOLD_score*fp_idx +  (1 |sn)')# + (1|fp_idx)' ROI_M*fp_idx  +
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('con_hit ~ 1 + BOLD_score*fp_idx + (1 |sn)')# + (1|fp_idx)' + ROI_M*fp_idx
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    formula = ('vis_hit ~ 1 + BOLD_score*fp_idx + (1 |sn)')# + (1|fp_idx)'
    print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    model = Lmer(formula, data=df, family='binomial')
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())

    # print('-' * 120)
    # formula = 'conn_score ~ 1 + hit_hit + (1|sn) + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=True, summary=True)
    # print(model.summary())
    #
    # print('-' * 120)
    # formula = 'conn_score ~ 1 + con_hit + (1|sn) + (1|fp_idx)'
    # print(f'{Fore.LIGHTYELLOW_EX}{formula=}{Fore.RESET}')
    # model = Lmer(formula, data=df)
    # model.fit(REML=True, verbose=True, summary=True)
    # print(model.summary())
