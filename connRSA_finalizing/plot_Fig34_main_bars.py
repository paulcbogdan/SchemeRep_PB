import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.single_trial_conn import prep_fps
# from old.networks import prep_networks
from org_sns import get_sns

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)


#
# os.chdir(r'/')


def get_stars(p, no_cross=True):
    if p < .0001:
        stars = '****'
    elif p < .001:
        stars = '***'
    elif p < .01:
        stars = '**'
    elif p < .05:
        stars = '*'
    elif p < .1 and not no_cross:
        stars = '†'
    else:
        stars = 'NS'
    return stars


def plot_scatter_overlay(cond2betas, big_voxelwise):
    all_betas = np.array([cond2betas[cond_other] for cond_other in cond2betas])
    height = np.max(all_betas)
    floor = np.min(all_betas)

    sns_out = np.any(all_betas > height, axis=0) | np.any(all_betas < floor, axis=0)
    cond2betas_ = {}
    for cond, betas in cond2betas.items():
        cond2betas_[cond] = betas[~sns_out]
    cond2betas = cond2betas_

    n_sn = len(cond2betas[(False, 'Local')])

    cond2spot = {(False, 'Local'): 0, (False, 'Distributed'): 1,
                 (True, 'Local'): 2, (True, 'Distributed'): 3}
    cond2color = {(False, 'Local'): 'dodgerblue',
                  (False, 'Distributed'): 'orange' if big_voxelwise else 'red',
                  (True, 'Local'): 'dodgerblue',
                  (True, 'Distributed'): 'orange' if big_voxelwise else 'red'}
    cond2jitter = {}

    for cond, spot in cond2spot.items():
        v = cond2betas[cond]
        jitter = np.random.uniform(-0.17, 0.17, len(v))
        dist = np.abs((v - np.mean(v)) / np.std(v))
        dist = np.clip(dist, 0, 3)
        dist = (3 - dist) / 3
        jitter *= dist
        cond2jitter[cond] = jitter
        plt.scatter(np.full(len(v), spot) + jitter, v, color=cond2color[cond],
                    s=25, alpha=0.25, linewidth=1, )
        plt.scatter(np.full(len(v), spot) + jitter, v, color=cond2color[cond],
                    s=25, alpha=0.5, facecolors='none', linewidth=1, )
    for i in range(n_sn):
        for bl in [False, True]:
            plt.plot([cond2spot[(bl, 'Local')] +
                      cond2jitter[(bl, 'Local')][i],
                      cond2spot[(bl, 'Distributed')] +
                      cond2jitter[(bl, 'Distributed')][i], ],
                     [cond2betas[(bl, 'Local')][i],
                      cond2betas[(bl, 'Distributed')][i]],
                     color='k', alpha=0.25,
                     linestyle=(5, (10, 3)),
                     linewidth=0.35, zorder=-1)

    plt.xticks([0.1, 0.9, 2.1, 2.9],
               ['Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)',
                'Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)'
                ],
               fontsize=19)
    plt.plot([-.5, 3.5], [0, 0], linewidth=0.5, color='k')
    plt.xlim(-.5, 3.5)
    pad = (height - floor) * .03
    return height, floor, pad


def plot_DistRep_bars(region='Occipital', big_voxelwise=True,
                      get_betas=False, no_lines_stars=False,
                      fp=None, VGG_semantic=False):
    trial_similarity = 'corr'
    second_order = 'spear'
    RDM_method = 'within_nan'
    stdize_by_run = False
    regress_row = False

    one_fp = fp is not None

    kwargs = {'fp': None,
              'trial_similarity': trial_similarity,
              'second_order': second_order,
              'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,
              'regress_row': regress_row,
              }

    if fp is None:
        four_tasks = '7'
        fps = prep_fps(four_tasks)
    else:
        fps = [fp]
    sns = get_sns('all')['healthy']

    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    conds = [(False, 'Local'), (False, 'Distributed'),
             (True, 'Local'), (True, 'Distributed')]

    if fp is not None:
        for tup in bad_tups:
            sns = [sn for sn in sns if (sn, fp) != tup]

    cond2betas = {}
    for cond in conds:
        cond2betas[cond] = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            for cond in conds:
                kwargs['semantic'] = cond[0]
                if VGG_semantic and cond[0]:
                    kwargs['semantic'] = (False, -1)
                # print(kwargs['semantic'])
                # print(f'{kwargs["semantic"]=}')
                if cond[1] == 'Local':
                    kwargs['ROI_focus'] = f'{region}_M_{trial_similarity}'
                    if big_voxelwise:
                        kwargs['ROIs_ctrl'] = []
                    else:
                        kwargs['ROIs_ctrl'] = [f'{region}_BOLD']
                else:
                    if big_voxelwise:
                        kwargs['ROI_focus'] = f'{region}_BOLD_cmb'
                        kwargs['ROIs_ctrl'] = []
                    else:
                        kwargs['ROI_focus'] = f'{region}_BOLD'
                        kwargs['ROIs_ctrl'] = [f'{region}_M_{trial_similarity}']

                if kwargs['semantic'] == False:
                    kwargs['semantic'] = (False, 0)

                beta1, test, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                             verbose=-1, easy_override=True)

                cond2betas[cond][i, j] = beta1 * 1000

    cond2betas = {cond: np.nanmean(betas, axis=1) for cond, betas
                  in cond2betas.items()}
    if get_betas:
        return cond2betas
    plt.figure(figsize=(5.3, 7.5))

    cond2Ms = {cond: np.nanmean(betas) for cond, betas in cond2betas.items()}
    cond2SEs = {cond: np.nanstd(betas) / np.sqrt(len(betas))
                for cond, betas in cond2betas.items()}

    itr = (cond2betas[conds[0]] - cond2betas[conds[1]] -
           cond2betas[conds[2]] + cond2betas[conds[3]])
    t, p = stats.ttest_1samp(itr, 0)

    per_ef = cond2betas[conds[0]] - cond2betas[conds[1]]
    per_t, per_p = stats.ttest_1samp(per_ef, 0)
    per_d = per_t / np.sqrt(len(per_ef))
    sem_ef = cond2betas[conds[2]] - cond2betas[conds[3]]
    sem_t, sem_p = stats.ttest_1samp(sem_ef, 0)
    sem_d = sem_t / np.sqrt(len(sem_ef))

    # for cond in conds:
    #     print(F'{cond} | {cond2Ms[cond]=:.3f}, {cond2SEs[cond]=:.3f}')
    # itr_M = np.nanmean(itr)
    # print(f'{itr_M=:.3f}')
    print(f'Interaction: {t=:.3f}, {p=:.5f}')
    print(f'\tPerception effect: {per_t=:.3f}, {per_p=:.5f}, {per_d=:.3f}')
    print(f'\tSemantic effect: {sem_t=:.3f}, {sem_p=:.5f}, {sem_d=:.3f}')

    Ms = np.array([cond2Ms[cond] for cond in conds])
    SEs = np.array([cond2SEs[cond] for cond in conds])
    ts = Ms / SEs
    print(f'{ts=}')

    plt.rcParams.update({'font.sans-serif': 'Arial'})
    colors = ['dodgerblue' if (isinstance(cond[1], bool) or 'Local' in cond[1])
              else ('orange' if big_voxelwise else 'red')
              for cond in conds]

    plt.bar([0, 1, 2, 3], Ms, yerr=SEs, color=colors,
            capsize=5, linewidth=0., edgecolor='k',
            alpha=.75, )

    for x in [0, 1, 2, 3]:
        plt.plot([x - .4, x - .4],
                 [0, Ms[x]], color='k', linewidth=0.75, alpha=1.0)
        plt.plot([x + .4, x + .4],
                 [0, Ms[x]], color='k', linewidth=0.75, alpha=1.0)
        plt.plot([x - .4, x + .4],
                 [Ms[x], Ms[x]], color='k', linewidth=0.75, alpha=1.0)

    height, floor, pad = plot_scatter_overlay(cond2betas, big_voxelwise)

    plt.xticks([0, 1, 2, 3],
               ['Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Avgs.)',
                'Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Avgs.)'],
               fontsize=19)

    lower_signif_line = (height - floor + 2 * pad) * .88 + floor - pad
    upper_signif_line = (height - floor + 2 * pad) * .93 + floor - pad
    shift_down = (height - floor + 2 * pad) * .06
    if not no_lines_stars:
        plt.plot([0, 1], [lower_signif_line, lower_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([0.5, 0.5], [lower_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([2, 3], [lower_signif_line, lower_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([2.5, 2.5], [lower_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([0.5, 2.5], [upper_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)

        # stars_height = height * .92 if stars != 'NS' else height * 1.02
        itr_stars = get_stars(p)
        itr_stars = itr_stars.replace('NS', ' ')
        stars_fs = 36 if itr_stars not in ['NS', '†'] else 24
        # if 'NS' not in itr_stars:
        plt.text(1.5,
                 upper_signif_line - shift_down if itr_stars not in ['NS', '†'] else
                 upper_signif_line - shift_down,
                 itr_stars, ha='center', va='center',
                 fontsize=stars_fs, )

        oc_stars = get_stars(per_p)
        if 'NS' not in oc_stars:
            plt.text(0.5, lower_signif_line - shift_down,
                     oc_stars, ha='center', va='center',
                     fontsize=stars_fs, )
        itl_stars = get_stars(sem_p)
        if 'NS' not in itl_stars:
            plt.text(2.5, lower_signif_line - shift_down,
                     itl_stars, ha='center', va='center',
                     fontsize=stars_fs, )

    ax = plt.gca()
    if height > 0 and np.max(Ms) > 0:
        ytick_low = (floor // 10 * 10)
        ytick_high = (height // 10 * 10)
        ticks = list(range(int(ytick_low), int(ytick_high) + 1, 10))
        ticks_corr = [tick / 1000 for tick in ticks]
        plt.yticks(ticks, ticks_corr,
                   fontsize=20)

        ax.tick_params(axis='y', which='major', pad=5, )

        for label in ax.get_yticklabels():
            label.set_ha('right')

    plt.ylabel('Mean correlation' if big_voxelwise else 'Mean beta',
               fontsize=24, labelpad=0)
    ax.yaxis.set_label_coords(-0.25, 0.5)

    plt.gca().spines[['top', 'right', 'bottom']].set_visible(False)
    plt.ylim(floor - pad, height + pad)
    plt.text(0.5, floor - pad * 12, 'Perceptual',
             fontsize=22, ha='center')
    plt.text(2.5, floor - pad * 12, 'Semantic',
             fontsize=22, ha='center')

    plt.subplots_adjust(bottom=0.26, left=0.29, right=.98, top=.98)

    no_stars_str = 'no_stars_' if no_lines_stars else ''
    one_fp_str = f'one_fp/{fp}_' if one_fp else ''
    if one_fp:
        fp2title = {'bl7_fMRI': 'Passive naming',
                    'obj7_fMRI': 'Object-scene comparison',
                    'vis7_fMRI': 'Visual recognition',
                    'con7_fMRI': 'Conceptual recognition', }
        plt.title(fp2title[fp], fontsize=20)
        plt.subplots_adjust(top=.95)

    if big_voxelwise:
        fp = (f'result_pics/DistRep_bars/{one_fp_str}'
              f'{no_stars_str}yellow_voxelwise_{region}.png')
    else:
        fp = (f'result_pics/DistRep_bars/{one_fp_str}'
              f'{no_stars_str}red_averages_{region}.png')
    print(f'Figure out: {fp=}')
    plt.savefig(fp, dpi=300)
    plt.show()

def do_three_way_itr(big_voxelwise=False):
    occ_betas = plot_DistRep_bars('Occipital', big_voxelwise=big_voxelwise,
                                  get_betas=True)
    # print(occ_betas)
    # return
    # quit()

    # for key, v in occ_betas.items():
    #     print(f'{key=}, {np.nanmean(v)=:.3f}')

    ITL_betas = plot_DistRep_bars('ITL', big_voxelwise=big_voxelwise,
                                  get_betas=True)

    per_itr = (occ_betas[(False, 'Local')] - occ_betas[(False, 'Distributed')] -
               ITL_betas[(False, 'Local')] + ITL_betas[(False, 'Distributed')])
    # for key, v in ITL_betas.items():
    #     print(f'{key=}, {np.nanmean(v)=:.3f}')
    print('-------------------')

    # print(per_itr)
    t, p = stats.ttest_1samp(per_itr, 0)
    F = t ** 2
    print(f'Perceptual: F[{len(per_itr) - 1}] = {F:.2f}, {p=:.4f}')
    # return
    sem_itr = (occ_betas[(True, 'Local')] - occ_betas[(True, 'Distributed')] -
                ITL_betas[(True, 'Local')] + ITL_betas[(True, 'Distributed')])
    t, p = stats.ttest_1samp(sem_itr, 0)
    F = t ** 2
    print(f'Semantic: F[{len(sem_itr) - 1}] = {F:.2f}, {p=:.4f}')

def plot_DistRep_Fig3_bars():
    plot_DistRep_bars('Occipital', big_voxelwise=True)
    plot_DistRep_bars('ITL', big_voxelwise=True)
    # plot_DistRep_bars('Parietal', big_voxelwise=True)
    # plot_DistRep_bars('PFC', big_voxelwise=True)


def plot_DistRep_Fig4_bars():
    plot_DistRep_bars('Occipital', big_voxelwise=False)
    plot_DistRep_bars('ITL', big_voxelwise=False)
    # plot_DistRep_bars('Parietal', big_voxelwise=False)
    # plot_DistRep_bars('PFC', big_voxelwise=False)


def plot_FigureS1_bars():
    for fp in ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']:
        plot_DistRep_bars('ITL', big_voxelwise=True, fp=fp)
        plot_DistRep_bars('ITL', big_voxelwise=False, fp=fp)


import sys

sys.setrecursionlimit(10000)

if __name__ == '__main__':
    # do_three_way_itr(big_voxelwise=True)
    # do_three_way_itr(big_voxelwise=False)
    # quit()

    plot_DistRep_Fig3_bars()
    # plot_DistRep_Fig4_bars()
    # quit()
    # plot_FigureS1_bars()
