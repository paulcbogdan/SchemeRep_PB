import utils
from Study1A.load_Study1A_funcs import get_ROI_vecs
from Utils.atlas_funcs import get_atlas
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from time import time
from tqdm import tqdm

@njit(fastmath=True, nopython=True, cache=True)
def get_ROI_synergy(vecs, stim_RSM, nan_mask, stim_is_nan):
    n_trials = vecs.shape[0]
    n_voxels = vecs.shape[1]

    I_both_all = 0
    I_same0_all = 0
    I_same1_all = 0
    skips = 0

    for v0 in range(n_voxels):
        for v1 in range(v0):
            n_both = 0
            n_same0 = 0
            n_same1 = 0

            I_both = 0
            I_same0 = 0
            I_same1 = 0
            for t0 in range(n_trials):
                if nan_mask[t0, v0] or nan_mask[t0, v1]:
                    continue
                for t1 in range(t0):
                    if nan_mask[t1, v0] or nan_mask[t1, v1]:
                        continue
                    if stim_is_nan[t0, t1]:
                        break

                    if vecs[t0, v0] == vecs[t1, v0]:
                        n_same0 += 1
                        I_same0 += stim_RSM[t0, t1]
                        if vecs[t0, v1] == vecs[t1, v1]:
                            n_same1 += 1
                            I_same1 += stim_RSM[t0, t1]
                            n_both += 1
                            I_both += stim_RSM[t0, t1]
                    else:
                        if vecs[t0, v1] == vecs[t1, v1]:
                            n_same1 += 1
                            I_same1 += stim_RSM[t0, t1]

            if n_same0 == 0 or n_same1 == 0:
                skips += 1
                continue

            I_same0 /= n_same0
            I_same1 /= n_same1
            I_both /= n_both
            I_both_all += I_both
            I_same0_all += I_same0
            I_same1_all += I_same1

    return I_both_all, I_same0_all, I_same1_all, skips

def do_ROI_synergy_sn(sn, fp, just_ROIs=None,
                      semantic=True, layer=None):
    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_RSM = stats.zscore(stim_RSM, axis=None, nan_policy='omit')
    stim_is_nan = np.isnan(stim_RSM)

    atlas = get_atlas(combine_regions=False)
    df_sn = get_trial_info(sn, easy_override=True)
    df_sn, sess = sort_df_sn(df_sn, fp)
    ROI2vecs = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.0)

    ROIs = atlas['ROIs']
    synergies = []
    for ROI in tqdm(ROIs):
        if just_ROIs:
            if ROI not in just_ROIs:
                continue
        vecs = ROI2vecs[ROI]
        vecs_bool = vecs > np.nanmedian(vecs, axis=0)

        nan_mask = np.isnan(vecs)
        I_both_all, I_same0_all, I_same1_all, skips = (
            get_ROI_synergy(vecs_bool, stim_RSM, nan_mask, stim_is_nan))
        synergy = I_both_all - I_same0_all - I_same1_all
        # print(f'{ROI}: {synergy=:.3f}, {skips=}')
        synergies.append(synergy)
    return synergies


def do_ROI_synergy():
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI']#, 'con7_fMRI', 'vis7_fMRI']
    sns = sns[9::10]
    atlas = get_atlas(combine_regions=False)
    all_synergies = []
    # for fp in fps:
    for sn in sns:
        sn_synergies = []
        for fp in fps:
            print(f'Onto: {sn}, {fp}')
            t_st = time()
            synergies = utils.pickle_wrap(do_ROI_synergy_sn,
                                          kwargs={'sn': sn, 'fp': fp},)
            t_en = time()
            print(f'{sn}, {fp}: {t_en - t_st:.2f}s')
            sn_synergies.append(synergies)
        sn_synergies = np.array(sn_synergies)
        sn_synergies = np.nanmean(sn_synergies, axis=0)
        print(sn_synergies)
        all_synergies.append(sn_synergies)

    for ROI_i, ROI in enumerate(atlas['ROIs']):
        ROI_synergies = [all_synergies[sn][ROI_i] for sn in range(len(sns))]
        M = np.nanmean(ROI_synergies)
        N = np.sum(~np.isnan(ROI_synergies))
        SE = np.nanstd(ROI_synergies) / np.sqrt(len(sns))
        t = stats.ttest_1samp(ROI_synergies, 0)
        p = t.pvalue
        t = t.statistic
        print(f'{ROI}: {M=:.3f} [{SE:.3f}], t[{N-1}]={t:.3f}, {p=:.3f}')


if __name__ == '__main__':
    do_ROI_synergy()
