import warnings

import numpy as np
import pandas as pd
from scipy import stats as stats
from tqdm import tqdm

from atlas_utils import get_atlas
from conn_utils import get_BNA_ROIs
from old.plot_gen import plot_connectivity, my_plot_surf

from utils import get_default_fp, pickle_wrap, make_title_str


def print_settings(settings):
    print(f'{settings=}')


def report_results(results, do_lmer=False, ISPC=False):
    print_settings(results['settings'])
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

    # quit()
    ts = np.full(scores_by_fp.shape[2], np.nan)
    ts_by_fp = np.full(scores_by_fp.shape[1:], np.nan)
    for j, ROI in enumerate(results['keys']):
        ROI_scores_by_fp = scores_by_fp[:, :, j]
        M_by_fp = np.nanmean(ROI_scores_by_fp, axis=0)
        SD_by_fp = np.nanstd(ROI_scores_by_fp, axis=0)
        N_by_fp = len(ROI_scores_by_fp[~np.isnan(ROI_scores_by_fp)])
        SE_by_fp = SD_by_fp / np.sqrt(N_by_fp)
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

        print(f'{ROI} ({M_size:.1f}), t[{N - 1}]={t:.2f}, p={p:.3f}, '
              f'{t_by_fp_str} {lmer_result_str}')





    return ts, ts_by_fp

def get_lmer_matrix(results):
    print(f'{results["scores_by_ROI"].shape=}')
    n_regions = results['scores_by_ROI'].shape[2]
    print(f'{n_regions=}')
    lmer_ar = np.full((n_regions, n_regions), np.nan)
    for i in tqdm(range(n_regions), desc='running lmers'):
        for j in range(n_regions):
            if i > j:
                continue
            else:
                pair_scores = results['scores_by_ROI'][:, :, i, j, :]
                # print(f'{i} | {j} ')
                pair_scores_raveled = pair_scores.ravel()
                # print(f'{pair_scores_raveled.shape=}')
                df = pd.DataFrame({'IRAF': pair_scores_raveled})
                # print(f'{pair_scores.shape=}')
                idxs = np.ndindex(pair_scores.shape)
                idxs = np.array(list(idxs))

                df[['sn', 'fp', 'stim']] = idxs
                for key in ['sn', 'fp', 'stim']:
                    df[key] = df[key].astype(str)
                # print(df)
                df.dropna(inplace=True)
                if len(df) < 10000:
                    print(f'{i}, {j} | many na drops {len(df)=}')
                    lmer_ar[i, j] = np.nan
                    lmer_ar[j, i] = np.nan
                    continue

                # print(df)
                from pymer4.models import Lmer
                # st = time()
                formula = 'IRAF ~ 1 + (1|sn) + (1|fp)'
                model = Lmer(formula, data=df)
                # model.fit()
                # print(model.summary())

                model.fit(REML=True, verbose=False, summary=False)
                summary = model.coefs
                lmer_t = summary['T-stat'].loc['(Intercept)']
                lmer_ar[i, j] = lmer_t
                lmer_ar[j, i] = lmer_t
    return lmer_ar

def visualize_region_matrix(results, plot_lmer=False):
    M_all = np.nanmean(results['scores'], axis=0)
    SD_all = np.nanstd(results['scores'], axis=0)
    N_all = np.sum(~np.isnan(results['scores']), axis=0)
    N = np.max(N_all)
    SE_all = SD_all / np.sqrt(N_all)
    t_all = M_all / SE_all

    # ticks = results['ticks']
    # tick_labels = results['tick_labels']
    # tick_lows = results['tick_lows']

    # print(f'{ticks=}')
    # print(f'{tick_lows=}')

    conn_str = f'conn={results["settings"]["conn"]}'
    second_order_str = f'second_order={results["settings"]["second_order"]}'
    trial_similarity_str = f'trial_similarity={results["settings"]["trial_similarity"]}'
    analysis_str = f'RSA={results["settings"]["RSA"]}, ' \
                   f'arg={results["settings"]["semantic"]}'
    title_str = f'n = {N}, {analysis_str}, \n' \
                f'{conn_str}, {second_order_str}, {trial_similarity_str}'

    plot_connectivity(t_all,
                      # ticks,
                      # tick_labels,
                      # tick_lows,
                      results['ticks'],
                      results['tick_labels'],
                      results['tick_lows'],
                      no_avg=True,
                      title=f't-test: {title_str}',
                      cbar_label='t-value',
                      vmin=-4, vmax=4)

    if N == 24 or N >= 30:
        results['scores_by_ROI'] = np.array(results['scores_by_ROI'])
        scores_trialwise = results['scores_by_ROI']
        for fp in range(scores_trialwise.shape[1]):
            scores_fp = np.mean(scores_trialwise[:, fp, :, :, :],
                                axis=-1)
            M_scores_fp = np.nanmean(scores_fp, axis=0)
            SD_scores_fp = np.nanstd(scores_fp, axis=0)
            N_scores_fp = np.sum(~np.isnan(scores_fp), axis=0)
            SE_scores_fp = SD_scores_fp / np.sqrt(N_scores_fp)
            t_scores_fp = M_scores_fp / SE_scores_fp
            plot_connectivity(t_scores_fp,
                              # ticks,
                              # tick_labels,
                              # tick_lows,
                              results['ticks'],
                              results['tick_labels'],
                              results['tick_lows'],
                              no_avg=True,
                              title=f't-test (fp={fp}): {title_str}',
                              cbar_label='t-value',
                              vmin=-4, vmax=4)
        # quit()

    if not plot_lmer:
        return

    settings = results['settings']
    lmer_fp = get_default_fp(None, settings, get_lmer_matrix,
                             r'../cache/lmer_ar', False)

    lmer_ar = pickle_wrap(lmer_fp, lambda: get_lmer_matrix(results))

    plot_connectivity(lmer_ar,
                      # ticks,
                      # tick_labels,
                      # tick_lows,
                      results['ticks'],
                      results['tick_labels'],
                      results['tick_lows'],
                      no_avg=True,
                      title=f'lmer: {title_str}',
                      cbar_label='t-value',
                      vmin=-4, vmax=4)

def visualize_ROIs(results, do_lmer=False):
    from connsearch.report import plot_ROI_scores
    if 'atlas' in results['settings'] and results['settings']['atlas'] == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=results['settings']['combine_regions'],
                          combine_bilateral=False,
                          split=results['settings']['split'], split_code='xyz')
    # print(atlas['ROIs'])
    # quit()
    ROI2coord = atlas['ROI2coord']
    # print(ROI2coord)
    # print(f'{len(ROI2coord)=}')
    # print(list(atlas['ROI2coord']))
    # quit()
    results['keys'] = [ROI.replace('LH_', 'L_').replace('RH_', 'R_')
                       for ROI in results['keys']]
    # print(list(ROI2coord))
    # print(results['keys'])
    results_coords = [ROI2coord[ROI] for ROI in results['keys']]
    ts, ts_by_fp = report_results(results, do_lmer=do_lmer)
    print(f'{do_lmer=}')



    # print(f'{len(atlas["ROIs"])=}')
    # quit()

    ROI2atlas_idx = {ROI: i for i, ROI in enumerate(atlas['ROIs'])}
    scores = np.full((len(atlas['ROIs']),), np.nan)
    for i, ROI in enumerate(results['keys']):
        scores[ROI2atlas_idx[ROI]] = ts[i]
    if results['settings']['RSA']:
        title_short = make_title_str('', 'obj', 1, False,
                                     results['settings']['semantic'],
                                     short=True)
    else:
        title_short = 'Object IPS. YA.'
    my_plot_surf(scores, atlas, title_short)
    quit()

    # ROI2score = {}
    # for i, ROI in enumerate(results['keys']):
    #     ROI2score[ROI] = ts[i]
    # scores_w_NaNs = []
    # print(f'{len(ROI2score)=}')
    # for ROI in atlas['ROIs']:
    #     if ROI in ROI2score:
    #         scores_w_NaNs.append(ROI2score[ROI])
    #     else:
    #         scores_w_NaNs.append(np.nan)
    # print(f'{len(scores_w_NaNs)=}')
    # print(ts)
    # quit()
    plot_ROI_scores(ts, results_coords, fp_out='trash.png', show=True,
                    vmin=0, vmax=2, title='all')

    for k in range(ts_by_fp.shape[0]):
        plot_ROI_scores(ts_by_fp[k, :], results_coords, fp_out='trash.png',
                        show=True, vmin=0, vmax=3, title=f'fp: {k}')

    quit()
