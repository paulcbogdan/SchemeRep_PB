import warnings

import numpy as np
import pandas as pd
from scipy import stats as stats

from conn_utils import get_BNA_ROIs


def print_settings(settings):
    print(f'{settings=}')


def report_results(results, do_lmer=False, ISPC=False):
    print_settings(results['settings'])
    scores = np.array(results['scores'])
    sizes = np.array(results['sizes'])
    # if scores.shape[1] == 246:
    #     print('\t Results printing for BOLD is not yet implemented')
    #     return

    if len(results['keys']) != scores.shape[1]:
        if results['settings']['conn'] == 'BOLD':
            assert scores.shape[1] == 246, f'BAD BOLD {scores.shape=}'
            results['keys'] = get_BNA_ROIs()
            warnings.warn('For Bold, setting keys to 246 BNA ROIs')
        else:
            raise ValueError(f'{len(results["keys"])=} != {scores.shape=}')

    # assert len(results['keys']) == scores.shape[1], \
    #     f'{len(results["keys"])=}, {scores.shape=}'
    if ISPC:
        scores_by_ROI = np.array(results['scores_by_ROI'])
        scores_by_fp = np.transpose(scores_by_ROI, (0, 2, 1))
    else:
        scores_by_ROI = np.array(results['scores_by_ROI'])
        scores_by_fp = np.nanmean(scores_by_ROI, axis=3)

    for j, ROI in enumerate(results['keys']):
        ROI_scores_by_fp = scores_by_fp[:, :, j]
        M_by_fp = np.nanmean(ROI_scores_by_fp, axis=0)
        SD_by_fp = np.nanstd(ROI_scores_by_fp, axis=0)
        N_by_fp = len(ROI_scores_by_fp[~np.isnan(ROI_scores_by_fp)])
        SE_by_fp = SD_by_fp / np.sqrt(N_by_fp)
        t_by_fp = M_by_fp / SE_by_fp
        t_by_fp_str = '['
        for t in t_by_fp:
            t_by_fp_str += f'{t:.2f}, '
        t_by_fp_str = t_by_fp_str[:-2]
        t_by_fp_str += ']'

        ROI_scores = scores[:, j]
        M = np.nanmean(ROI_scores)
        SD = np.nanstd(ROI_scores)
        N = len(ROI_scores[~np.isnan(ROI_scores)])
        SE = SD / np.sqrt(N)
        t = M / SE
        p = stats.t.sf(np.abs(t), N - 1)
        M_size = np.nanmean(sizes[:, j])

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
                # print('\tLmer: all NaN')
                continue
            formula = 'conn_score ~ 1 + (1|sn) + (1|fp)'

            from pymer4.models import Lmer
            model = Lmer(formula, data=df_ROI_as_d)
            model.fit(REML=True, verbose=False, summary=False)
            # print(model.summary())
            # print('-'*100)
            summary = model.coefs
            lmer_t = summary['T-stat'].loc['(Intercept)']
            lmer_p = summary['P-val'].loc['(Intercept)']
            lmer_result_str = f'\tLmer: t={lmer_t:.2f}, p={lmer_p:.3f}'
        else:
            lmer_result_str = ''

        print(f'{ROI} ({M_size:.1f}), t[{N - 1}]={t:.2f}, p={p:.3f}, '
              f'{t_by_fp_str} {lmer_result_str}')
