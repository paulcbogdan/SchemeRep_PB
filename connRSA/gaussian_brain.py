import numpy as np
import pandas as pd
from numba.cuda import set_memory_manager
from tqdm import tqdm

from connRSA.conn_utils import get_trial_x_trial
from connRSA.fft_funcs import get_img_box
from connRSA.fft_rsm import run_FFT_RSA2
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
import utils

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def get_gaus_neural_RSM(region, fp, sn, sigma, subtract_sigmas=tuple(),
                        nan_thresh=1.01):
    if '_all' in region:
        region = region.replace('_all', '')
    elif '_L' not in region and '_R' not in region:
        neural_RSM_L = utils.pickle_wrap(get_gaus_neural_RSM, None,
                                            kwargs={'region': f'{region}_L',
                                                    'fp': fp, 'sn': sn,
                                                    'sigma': sigma,
                                                    'subtract_sigmas': subtract_sigmas},
                                            verbose=-1, easy_override=False,
                                            dir_branches=100)
        neural_RSM_R = utils.pickle_wrap(get_gaus_neural_RSM, None,
                                            kwargs={'region': f'{region}_R',
                                                    'fp': fp, 'sn': sn,
                                                    'sigma': sigma,
                                                    'subtract_sigmas': subtract_sigmas},
                                            verbose=-1, easy_override=False,
                                            dir_branches=100)
        neural_RSM = (neural_RSM_L + neural_RSM_R) / 2
        return neural_RSM

    img_box = utils.pickle_wrap(get_img_box, None,
                                kwargs={'region': region, 'fp': fp, 'sn': sn},
                                verbose=-1, easy_override=False,
                                RAM_cache=False)

    cnt_nan = np.sum(np.isnan(img_box))
    # print(f'{cnt_nan=}')

    p_nan = np.mean(np.isnan(img_box), axis=-1)
    mask = p_nan < nan_thresh

    if sigma is None:
        img_box_g = img_box
    else:
        img_box_g = np.full_like(img_box, np.nan)
        for trial in range(img_box.shape[-1]):
            img_box_g[..., trial] = gaussian_filter_ignore_nan(img_box[..., trial], sigma)
            if len(subtract_sigmas):
                for sig_subtract in subtract_sigmas:
                    img_box_g[..., trial] -= gaussian_filter_ignore_nan(img_box[..., trial],
                                                                        sig_subtract)

    img_box_g = stats.zscore(img_box_g, axis=-1) # NEEDED FOR PERFECT SIMILARITY?

    img_box_g_flat = img_box_g[mask, :].T

    neural_RSM = get_trial_x_trial(img_box_g_flat)
    return neural_RSM

def gaussian_filter_ignore_nan(data, sigma, d1=False, axis=0):
    # from Google AI
    """Apply a Gaussian filter to an array, ignoring NaN values."""

    # Create a mask of non-NaN values

    # Disable RuntimeWarning: invalid value encountered in divide
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    mask = ~np.isnan(data)

    # Replace NaNs with zeros
    data_zeroed = np.where(mask, data, 0)

    # Apply Gaussian filter
    if d1:
        filtered_data = gaussian_filter1d(data_zeroed, sigma, axis=axis)
    else:
        filtered_data = gaussian_filter(data_zeroed, sigma)

    # Compute the sum of weights for each pixel
    if d1:
        weights = gaussian_filter1d(mask.astype(float), sigma, axis=axis)
    else:
        weights = gaussian_filter(mask.astype(float), sigma)

    # Normalize the filtered data by the weights
    filtered_data /= weights
    # mask = mask | (weights > 0)

    # Replace values where the original data was NaN
    filtered_data[~mask] = np.nan
    filtered_data[np.isinf(filtered_data)] = np.nan
    # print(np.nanmax(filtered_data))
    # print(filtered_data)
    # quit()

    return filtered_data


def run_gaus_RSA(sn, region, fp, sigma, semantic, layer):
    neural_RSM = utils.pickle_wrap(get_gaus_neural_RSM, None,
                                   kwargs={'sn': sn, 'region': region,
                                           'fp': fp, 'sigma': sigma},
                                   verbose=-1, easy_override=True)

    neural_RSM = within_run_to_nan(neural_RSM)
    # print(neural_RSM.shape)
    # quit()

    # dir_in = fr'cache/conn_RSA/ars/RSA'
    # RDM_method_ = 'within_nan'
    # dir_focus1 = (f'{dir_in}/{fp}_{"corr"}_'
    #              f'{"spear"}_{RDM_method_}_{False}')
    # region = region.replace('_all', '')
    # fp_focus1 = f'{dir_focus1}/{sn}_{region}_BOLD_cmb.npy'
    # neural_RSM_old = np.load(fp_focus1)
    # neural_RSM_old = within_run_to_nan(neural_RSM_old)
    # plt.imshow(neural_RSM)
    # plt.show()
    # plt.imshow(neural_RSM_old)
    # plt.show()

    # neural_RSM = neural_RSM[np.tril_indices_from(neural_RSM, k=-1)]
    # neural_RSM_old = neural_RSM_old[np.tril_indices_from(neural_RSM_old, k=-1)]
    # r, p = stats.spearmanr(neural_RSM, neural_RSM_old, nan_policy='omit')
    # print(f'{r=:.3f}')
    # quit()

    neural_RSM[np.diag_indices_from(neural_RSM)] = np.nan
    trils = np.tril_indices_from(neural_RSM, k=-1)
    neural_RSM = neural_RSM[trils]

    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    # plt.imshow(stim_RSM)
    # plt.colorbar()
    # plt.show()
    # quit()
    stim_RSM = stim_RSM[trils]
    r, _ = stats.spearmanr(neural_RSM, stim_RSM, nan_policy='omit')
    # print(f'{r=}')
    # quit()
    return r

def run_all_sns_gaus(region='IT', sigma=10,
                     semantic=True, layer=None,):

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI', 'obj7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI']

    rs_all = np.full((len(sns), len(fps)), np.nan)
    # print(f'GO: {region}, {sigma}')
    print(f'Cooking: {region}, {sigma=}')
    sns = sns[::-1]
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            # for sigma in range(20):
            r = utils.pickle_wrap(run_gaus_RSA, None,
                                  kwargs={'sn': sn, 'region': region,
                                          'fp': fp, 'sigma': sigma,
                                          'semantic': semantic,
                                          'layer': layer,},
                                  easy_override=False,
                                  verbose=-1, dir_branches=100)
            rs_all[i, j] = r
            # print(r)
            # quit()

    # assert np.sum(np.isnan(rs_all)) == 0
    subj_rs = np.nanmean(rs_all, axis=1)
    t, p = stats.ttest_1samp(subj_rs, 0, nan_policy='omit')
    M = np.nanmean(subj_rs) * 100
    SE = stats.sem(subj_rs, nan_policy='omit') * 100
    print(f'{region} : {sigma} | {M=:.2f}, {SE=:.2f}, {t=:.2f}, {p=:.3f}')
    # subj_rs /= np.std(subj_rs)
    return subj_rs

def analyze_multi_gaus(region='ITL_L'):
    # print(f'Semantic')

    sigmas = list(range(0, 20))
    # sigmas = [None]

    for sigma in sigmas:
        print('Semantic')
        run_all_sns_gaus(region=region, sigma=sigma, semantic=True,
                         layer=None)

        print('Perception DNN = 0')
    # for sigma in sigmas:
        run_all_sns_gaus(region=region, sigma=sigma, semantic=False,
                         layer=0)

def plot_gaus_by_region():
    plt.rcParams.update({'font.size': 16})
    # fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))

    # region1 = 'Occipital'
    # region2 = 'ITL'

    # region1 = 'cortical'
    # region2 = 'PFC'

    region1 = 'LPFC'
    region2 = 'Parietal'

    plt.sca(axs[0, 0])
    plot_gaus_region(f'{region1}_L', )
    plt.sca(axs[0, 1])
    plot_gaus_region(f'{region1}_L', )
    plt.sca(axs[0, 2])
    plot_gaus_region(f'{region1}_R', )

    plt.sca(axs[1, 0])
    plot_gaus_region(f'{region2}_L', )
    plt.sca(axs[1, 1])
    plot_gaus_region(f'{region2}_L', )
    plt.sca(axs[1, 2])
    plot_gaus_region(f'{region2}_L', )

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


def plot_gaus_region(region='IT_L', semantic=True, std=False):
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    print(f'{region=}')
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


    # fzs = [(i, i+1) for i in range(1, 12, 1)]
    sigmas = list(range(8))
    # fzs = [(i, i+2) for i in range(1, 15, 2)]

    Ms_semantic = []
    SEs_semantic = []
    subj_rs_l_semantic = []
    for sigma in sigmas:
        # subj_rs = run_all_sns_gaus(region, sigma, semantic=False, layer=0)
        subj_rs = run_all_sns_gaus(region, sigma, semantic=True, layer=None)
        subj_rs_l_semantic.append(subj_rs)
        if std:
            subj_rs /= np.std(subj_rs)
            subj_rs *= np.sqrt(len(subj_rs))
        M = np.mean(subj_rs)
        SE = stats.sem(subj_rs)
        Ms_semantic.append(M)
        SEs_semantic.append(SE)
    # fzs_first = [fz[0] for fz in fzs]
    plt.errorbar(sigmas, Ms_semantic, yerr=SEs_semantic,
                 label='Semantic', color='green', marker='.')

    Ms_per = []
    SEs_per = []
    for i, sigma in enumerate(sigmas):
        subj_rs = run_all_sns_gaus(region, sigma, semantic=False, layer=0)
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
            plt.text(sigmas[i],
                     (Ms_semantic[i] + M + np.min([SEs_semantic[i], SE])) / 2,
                     '*', fontsize=18,
                     ha='center', va='top', color='red')
    plt.errorbar(sigmas, Ms_per, yerr=SEs_per,
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




if __name__ == '__main__':
    plot_gaus_by_region()
    # analyze_multi_gaus()
