import numpy as np
from tqdm import tqdm

from connRSA.fft_funcs import get_ffts, extract_fz, ffts_reconstruct, get_uniform_size_ffts
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from org_sns import get_sns
import utils

import scipy.stats as stats
import matplotlib.pyplot as plt

# suppress RuntimeWarning: invalid value encountered in divide
np.seterr(divide='ignore', invalid='ignore')

def get_fz_neural_RSM(sn, region, fp, fz, uniform_size, eight_corners, do_mag,
                      do_reconstruct):

    if '_L' not in region and '_R' not in region:
        neural_RSM_L = utils.pickle_wrap(get_fz_neural_RSM, None,
                                         kwargs={'sn': sn, 'region': f'{region}_L',
                                                 'fp': fp, 'fz': fz,
                                                 'uniform_size': uniform_size,
                                                 'eight_corners': eight_corners,
                                                 'do_mag': do_mag,
                                                 'do_reconstruct': do_reconstruct},
                                         verbose=-1, easy_override=False)
        neural_RSM_R = utils.pickle_wrap(get_fz_neural_RSM, None,
                                         kwargs={'sn': sn, 'region': f'{region}_R',
                                                 'fp': fp, 'fz': fz,
                                                 'uniform_size': uniform_size,
                                                 'eight_corners': eight_corners,
                                                 'do_mag': do_mag,
                                                 'do_reconstruct': do_reconstruct},
                                         verbose=-1, easy_override=False)
        neural_RSM = (neural_RSM_L + neural_RSM_R) / 2
        return neural_RSM

    if uniform_size:
        ffts, mask, orig_size = utils.pickle_wrap(get_uniform_size_ffts, None,
                                                  kwargs={'sn': sn, 'region': region,
                                                          'fp': fp, },
                                                  verbose=-1, easy_override=False,
                                                  )
    else:
        ffts, mask = utils.pickle_wrap(get_ffts, None,
                                       kwargs={'sn': sn, 'region': region,
                                               'fp': fp,},
                                       verbose=-1, easy_override=False)
        orig_size = ffts.shape

    if do_mag:
        ffts = np.abs(ffts)


    if do_reconstruct:
        # fp_step2 = fr'cache/fft_reconstruct/{sn}_{region}_{fp}_{fz}_{uniform_size}.pkl'
        # # eight_corners = False
        # kw = {'ffts': ffts, 'mask': mask, 'fz': fz,
        #       'eight_corners': eight_corners}
        # if uniform_size:
        #     kw['resize'] = orig_size
        # vals = utils.pickle_wrap(ffts_reconstruct, fp_step2,
        #                          kwargs=kw,
        #                          verbose=-1, easy_override=False)
        vals = ffts_reconstruct(ffts, mask, fz, resize=orig_size,
                                eight_corners=eight_corners)
        neural_RSM = np.corrcoef(vals)
    else:
        vals = extract_fz(ffts, fz, eight_corners=eight_corners)
        neural_RSM = np.corrcoef(vals)
        neural_RSM = np.real(neural_RSM)
    return neural_RSM

def run_FFT_RSA(sn, region, fp, fz=(1, 5), semantic=True, layer=None,
                do_mag=False, do_reconstruct=False, uniform_size=True,
                eight_corners=False, ):
    assert not (do_mag and do_reconstruct)
    # print('-')

    neural_RSM = utils.pickle_wrap(get_fz_neural_RSM, None,
                                   kwargs={'sn': sn, 'region': region,
                                           'fp': fp, 'fz': fz,
                                           'uniform_size': uniform_size,
                                           'eight_corners': eight_corners,
                                           'do_mag': do_mag,
                                           'do_reconstruct': do_reconstruct},
                                   verbose=-1, easy_override=False)
    # t_st = time()


        # t_end = time()
        # print(f'{t_end - t_st=:.2f}')
    # t_end = time()
    # print(f'{t_end - t_st=:.2f}')
    # t_st = time()

    neural_RSM[np.diag_indices_from(neural_RSM)] = np.nan
    trils = np.tril_indices_from(neural_RSM, k=-1)
    neural_RSM = neural_RSM[trils]

    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_RSM = stim_RSM[trils]
    # t_end = time()
    # print(f'{t_end - t_st=:.2f}')
    # quit()

    # t_st = time()
    r, _ = stats.spearmanr(neural_RSM, stim_RSM, nan_policy='omit')
    # t_end = time()
    # print(f'{t_end - t_st=:.2f}')
    return r

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
    sns = sns[::-1]
    for i, sn in tqdm(enumerate(sns)):
        for j, fp in enumerate(fps):
            r = utils.pickle_wrap(run_FFT_RSA, None,
                                  kwargs={'sn': sn, 'region': region,
                                          'fp': fp, 'fz': fz,
                                          'semantic': semantic,
                                          'layer': layer,
                                          'do_mag': False,
                                          'do_reconstruct': do_reconstruct,
                                          'uniform_size': True,
                                          'eight_corners': eight_corners},
                                  easy_override=False,
                                  verbose=0,)
            rs_all[i, j] = r
    assert np.sum(np.isnan(rs_all)) == 0
    subj_rs = np.mean(rs_all, axis=1)
    t, p = stats.ttest_1samp(subj_rs, 0)
    M = np.mean(subj_rs) * 100
    SE = stats.sem(subj_rs) * 100
    print(f'{region} : {fz} | {M=:.2f}, {SE=:.2f}, {t=:.2f}, {p=:.3f}')
    return subj_rs

def analyze_multi_fz(region='IT', eight_corners=False,
                     do_reconstruct=False, wide_fz=False):
    print(f'Semantic')

    # fzs = [(i, i+3) for i in range(1, 24, 3)]
    fzs = [(i, i+1) for i in range(1, 15, 1)]
    for fz in fzs:
        run_all_sns(region, fz, True, None,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners,
                    wide_fz=wide_fz)
    #
    # print('Perceptual: DNN = 2')
    # # fzs = [(1, 4), (4, 8), (8, 12), ]
    # for fz in fzs:
    #     run_all_sns(region, fz, False, 2)

    # print('Perceptual: DNN = 6')
    # fzs = [(1, 4), (4, 8), (8, 12), ]
    # for fz in fzs:
    #     run_all_sns(region, fz, False, 6)

    for DNN_layer in range(0, 16):
        print(f'Perceptual: DNN = {DNN_layer} ({eight_corners=}, {do_reconstruct=})')
        for fz in fzs:
            run_all_sns(region, fz, False, DNN_layer,
                        do_reconstruct=do_reconstruct,
                        eight_corners=eight_corners,
                        wide_fz=wide_fz)

def plot_fz_interaction(region='IT_L', eight_corners=False, do_reconstruct=True,
                        semantic=True, wide_fz=False):
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


    fzs = [(i, i+1) for i in range(1, 15, 1)]
    Ms_semantic = []
    SEs_semantic = []
    subj_rs_l_semantic = []
    for fz in fzs:
        subj_rs = run_all_sns(region, fz, semantic=True, layer=None,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners, wide_fz=wide_fz)
        subj_rs_l_semantic.append(subj_rs)
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
    plt.ylabel('Mean correlation', fontsize=16)
    plt.yticks([0, 0.01, 0.02])
    plt.ylim(0, 0.02)
    plt.xlabel('Frequency neural signal (Hz)', fontsize=16)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    # plt.legend(frameon=False)
    plt.title(region)
    plt.gca().spines[['right', 'top']].set_visible(False)

    # 1 Hz = 49 voxels for full wavelength
    # 1 Hz = 24.5 voxels for half wavelength
    # 6 Hz = 4 voxels for half wavelength


def plot_fz_region_interaction(eight_corners=False, do_reconstruct=True,
                               wide_fz=False):
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    # fig.subplots_adjust(bottom=0.18, left=0.15, right=0.95, top=0.85, wspace=0.3)
    plt.sca(axs[0, 0])
    plot_fz_interaction('ITL_L', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    # print('TWOOOOOOOOOOO')
    plt.sca(axs[0, 1])
    plot_fz_interaction('ITL_R', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    plt.sca(axs[1, 0])
    plot_fz_interaction('Occipital_L', eight_corners=eight_corners,
                        do_reconstruct=do_reconstruct,
                        wide_fz=wide_fz)
    plt.sca(axs[1, 1])
    plot_fz_interaction('Occipital_R', eight_corners=eight_corners,
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


if __name__ == '__main__':
    EIGHT_CORNERS = False
    DO_RECONSTRUCT = True
    WIDE_FZ = True
    # TODO: Try ITL
    plot_fz_region_interaction()

    # analyze_multi_fz('ITL_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)
    # analyze_multi_fz('ITL_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)
    # analyze_multi_fz('Occipital_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)
    analyze_multi_fz('Occipital_L', do_reconstruct=DO_RECONSTRUCT,
                     eight_corners=EIGHT_CORNERS, wide_fz=WIDE_FZ)

    # analyze_multi_fz('Occipital', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_T_L', do_reconstruct=DO_RECONSTRUCT,
                     # eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_T_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('OC_IT', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)


    # analyze_multi_fz('OC_IT_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS) # (35, 67, 49, 114)
    # analyze_multi_fz('OC_IT_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS) # (35, 65, 53, 114)
    # IT_R: (29, 45, 27, 114)
    # Occipital_R: (29, 33, 37, 114)
