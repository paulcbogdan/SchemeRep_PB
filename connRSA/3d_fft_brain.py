import numpy as np
import pandas as pd
from tqdm import tqdm

from connRSA.fft_rsm import run_FFT_RSA2
from org_sns import get_sns
import utils

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# suppress RuntimeWarning: invalid value encountered in divide
np.seterr(divide='ignore', invalid='ignore')


def run_all_sns(region='IT', fz=(1, 5), semantic=True, layer=None,
                do_reconstruct=False, eight_corners=False,
                wide_fz=False):
    if wide_fz:
        fz = (fz[0], fz[1], 'all')
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI', 'obj7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI']
    rs_all = np.full((len(sns), len(fps)), np.nan)
    print(f'GO: {region}, {fz}')
    # sns = sns[::-1]
    for i, sn in tqdm(enumerate(sns)):
        for j, fp in enumerate(fps):
            r = utils.pickle_wrap(run_FFT_RSA2, None,
                                  kwargs={'sn': sn, 'region': region,
                                          'fp': fp, 'fz': fz,
                                          'semantic': semantic,
                                          'layer': layer,
                                          'do_mag': False,
                                          'do_reconstruct': do_reconstruct,
                                          'uniform_size': True,
                                          'eight_corners': eight_corners},
                                  easy_override=False,
                                  verbose=-1, dir_branches=100)
            rs_all[i, j] = r
    assert np.sum(np.isnan(rs_all)) == 0
    subj_rs = np.mean(rs_all, axis=1)
    t, p = stats.ttest_1samp(subj_rs, 0)
    M = np.mean(subj_rs) * 100
    SE = stats.sem(subj_rs) * 100
    print(f'{region} : {fz} | {M=:.2f}, {SE=:.2f}, {t=:.2f}, {p=:.3f}')
    # subj_rs /= np.std(subj_rs)
    return subj_rs

def analyze_multi_fz(region='IT', eight_corners=False,
                     do_reconstruct=False, wide_fz=False):
    print(f'Semantic')

    # fzs = [(i, i+3) for i in range(1, 24, 3)]
    fzs = [(i, i+1) for i in range(1, 12, 1)]
    fzs = [(0, 49)]


    for fz in fzs:
        run_all_sns(region, fz, True, None,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners,
                    wide_fz=wide_fz)


    for DNN_layer in range(0, 1):
        print(f'Perceptual: DNN = {DNN_layer} ({eight_corners=}, {do_reconstruct=})')
        for fz in fzs:
            run_all_sns(region, fz, False, DNN_layer,
                        do_reconstruct=do_reconstruct,
                        eight_corners=eight_corners,
                        wide_fz=wide_fz)


def plot_fz_interaction(region='IT_L', eight_corners=False, do_reconstruct=True,
                        semantic=True, wide_fz=False, std=False):
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    if len(fps) > 1:
        title = 'All tasks: '
    else:
        mapper = {'bl7_fMRI': 'Baseline task', 'obj7_fMRI': 'Encoding task',
                  'con7_fMRI': 'Conceptual retrieval', 'vis7_fMRI': 'Visual retrieval'}
        title = f'{mapper[fps[0]]}:\n'
    if isinstance(semantic, bool) and semantic:
        title += 'Semantic'
    elif isinstance(semantic, bool) and not semantic:
        title += 'DNN layer 2'
    elif isinstance(semantic, tuple):
        assert not semantic[0]
        title += f'DNN layer {semantic[1]}'
    title += f'\n{region}'


    fzs = [(i, i+1) for i in range(1, 12, 1)]
    # fzs = [(i, i+2) for i in range(1, 15, 2)]

    Ms_semantic = []
    SEs_semantic = []
    subj_rs_l_semantic = []
    for fz in fzs:
        subj_rs = run_all_sns(region, fz, semantic=True, layer=None,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners, wide_fz=wide_fz)
        subj_rs_l_semantic.append(subj_rs)
        if std:
            subj_rs /= np.std(subj_rs)
            subj_rs *= np.sqrt(len(subj_rs))
        M = np.mean(subj_rs)
        SE = stats.sem(subj_rs)
        Ms_semantic.append(M)
        SEs_semantic.append(SE)
    fzs_first = [fz[0] for fz in fzs]
    plt.errorbar(fzs_first, Ms_semantic, yerr=SEs_semantic,
                 label='Semantic', color='green', marker='.')


    Ms_per = []
    SEs_per = []
    for i, fz in enumerate(fzs):
        subj_rs = run_all_sns(region, fz, semantic=False, layer=0,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners, wide_fz=wide_fz)
        subj_rs_semantic = subj_rs_l_semantic[i]
        if std:
            subj_rs /= np.std(subj_rs)
            subj_rs *= np.sqrt(len(subj_rs))
        M = np.mean(subj_rs)
        SE = stats.sem(subj_rs)
        Ms_per.append(M)
        SEs_per.append(SE)
        t, p = stats.ttest_rel(subj_rs, subj_rs_semantic)
        if p < .05:
            plt.text(fzs_first[i],
                     (Ms_semantic[i] + M + np.min([SEs_semantic[i], SE])) / 2,
                     '*', fontsize=18,
                     ha='center', va='top', color='red')
    plt.errorbar(fzs_first, Ms_per, yerr=SEs_per,
                 label='Perceptual', color='darkviolet',
                 marker='.')
    if std:
        plt.ylabel('t-value', fontsize=16)
        plt.ylim(0, 10)
        plt.yticks([0, 2, 4, 6, 8, 10])
    else:
        plt.ylabel('Mean correlation', fontsize=16)
        # plt.yticks([0, 0.01, 0.02])
        # plt.ylim(0, 0.02)
    plt.xlabel('Frequency neural signal (Hz)', fontsize=16)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    # plt.legend(frameon=False)
    plt.title(region)
    plt.gca().spines[['right', 'top']].set_visible(False)

    # 1 Hz = 49 voxels for full wavelength
    # 1 Hz = 24.5 voxels for half wavelength
    # 6 Hz = 4 voxels for half wavelength


def plot_fz_by_region(eight_corners=False, do_reconstruct=False,
                      wide_fz=False):
    plt.rcParams.update({'font.size': 16})
    # fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))


    # region1 = 'ITL'
    # region2 = 'Occipital'

    region1 = 'LPFC'
    region2 = 'Parietal'

    plt.sca(axs[0, 0])
    plot_fz_interaction(f'{region1}', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    # print('TWOOOOOOOOOOO')
    plt.sca(axs[0, 1])
    plot_fz_interaction(f'{region1}_L', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)

    # fig.subplots_adjust(bottom=0.18, left=0.15, right=0.95, top=0.85, wspace=0.3)
    plt.sca(axs[0, 2])
    plot_fz_interaction(f'{region1}_R', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    # print('TWOOOOOOOOOOO')
    plt.sca(axs[1, 0])
    plot_fz_interaction(f'{region2}', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    plt.sca(axs[1, 1])
    # 'OC_T_L'
    plot_fz_interaction(f'{region2}_L', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    plt.sca(axs[1, 2])
    plot_fz_interaction(f'{region2}_R', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    leg = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=16,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5)
    plt.subplots_adjust(left=0.115,
                        bottom=0.17,
                        right=0.975,
                        top=0.915,
                        wspace=0.3,
                        hspace=.7
                        )
    # plt.tight_layout()
    plt.show()
    quit()

def fz_interaction(fzA=4, fzB=6, do_reconstruct=True,
                   eight_corners=False, wide_fz=False):
    region1 = 'ITL'
    region2 = 'Occipital'

    kw = {'region': region1, 'fz': (fzA, fzA + 1),
          'semantic': True, 'layer': None,
          'do_reconstruct': do_reconstruct,
          'eight_corners': eight_corners,
          'wide_fz': wide_fz}
    vals1A_semantic = run_all_sns(**kw)
    subjs = list(range(len(vals1A_semantic)))
    df_1As = pd.DataFrame({'val': vals1A_semantic, 'sn': subjs,
                           'region': region1, 'fz': fzA,
                           'semantic': True,})
    vals1A_perception = run_all_sns(**(kw | {'semantic': False, 'layer': 0}))
    df_1Ap = pd.DataFrame({'val': vals1A_perception, 'sn': subjs,
                           'region': region1, 'fz': fzA,
                           'semantic': False,})

    kw = kw | {'fz': (fzB, fzB + 1)}
    vals1B_semantic = run_all_sns(**kw)
    df_1Bs = pd.DataFrame({'val': vals1B_semantic, 'sn': subjs,
                           'region': region1, 'fz': fzB,
                           'semantic': True,})
    vals1B_perception = run_all_sns(**(kw | {'semantic': False, 'layer': 0}))
    df_1Bp = pd.DataFrame({'val': vals1B_perception, 'sn': subjs,
                           'region': region1, 'fz': fzB,
                           'semantic': False,})

    kw = kw | {'region': region2, 'fz': (fzA, fzA + 1)}
    vals2A_semantic = run_all_sns(**kw)
    df_2As = pd.DataFrame({'val': vals2A_semantic, 'sn': subjs,
                           'region': region2, 'fz': fzA,
                           'semantic': True,})
    vals2A_perception = run_all_sns(**(kw | {'semantic': False, 'layer': 0}))
    df_2Ap = pd.DataFrame({'val': vals2A_perception, 'sn': subjs,
                           'region': region2, 'fz': fzA,
                           'semantic': False,})

    kw = kw | {'fz': (fzB, fzB + 1)}
    vals2B_semantic = run_all_sns(**kw)
    df_2Bs = pd.DataFrame({'val': vals2B_semantic, 'sn': subjs,
                           'region': region2, 'fz': fzB,
                           'semantic': True,})
    vals2B_perception = run_all_sns(**(kw | {'semantic': False, 'layer': 0}))
    df_2Bp = pd.DataFrame({'val': vals2B_perception, 'sn': subjs,
                           'region': region2, 'fz': fzB,
                           'semantic': False,})

    df = pd.concat([df_1As, df_1Ap, df_1Bs, df_1Bp,
                    df_2As, df_2Ap, df_2Bs, df_2Bp])


    g = sns.catplot(x='semantic', y='val', col='region',
                    hue='fz',
                    data=df,
                    kind='boxen',
                    linecolor='k',
                    saturation=0.9,
                    palette=['red', 'dodgerblue', ],
                    height=3.75, aspect=1.2,
                    flier_kws={'edgecolor': ['k'],
                               'linewidth': 0.8}
                    )
    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], ['Perceptual', 'Semantic'])

    for ax in g.axes.flat:
        for line in ax.lines:
            if line.get_linestyle() == '-':
                line.set_color('white')
                line.set_linewidth(1.5)

    axes_dict = list(g.axes_dict.items())
    # print(axes_dict[0][1])
    plt.sca(axes_dict[0][1])
    plt.xlabel('')
    plt.plot([-0.5, 1.5], [0, 0], color='k', linewidth=1)

    plt.sca(axes_dict[1][1])
    plt.xlabel('')
    plt.plot([-0.5, 1.5], [0, 0], color='k', linewidth=1)

    # print(g.axes_dict.items()['ITL'])
    # quit()
    plt.show()



if __name__ == '__main__':
    EIGHT_CORNERS = False
    DO_RECONSTRUCT = True
    WIDE_FZ = False
    # TODO: Try ITL
    # fz_interaction()

    # TODO: ADD     img_box_g = stats.zscore(img_box_g, axis=-1) # NEEDED FOR PERFECT SIMILARITY?

    plot_fz_by_region()

    analyze_multi_fz('Parietal', do_reconstruct=DO_RECONSTRUCT,
                     eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)
    # analyze_multi_fz('PL_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)

    # analyze_multi_fz('Occipital_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)
    # analyze_multi_fz('Occipital_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)

    # analyze_multi_fz('Occipital', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_T_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_IT_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_IT_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_IT', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)


    # analyze_multi_fz('OC_IT_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS) # (35, 67, 49, 114)
    # analyze_multi_fz('OC_IT_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS) # (35, 65, 53, 114)
    # IT_R: (29, 45, 27, 114)
    # Occipital_R: (29, 33, 37, 114)
