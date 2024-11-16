import os

os.environ['R_HOME'] = r'C:\Users\Paul\anaconda3\envs\py312\Lib\R'

import warnings

import numpy as np
import pandas as pd
from scipy import stats as stats

# from conn_utils import get_BNA_ROIs



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


