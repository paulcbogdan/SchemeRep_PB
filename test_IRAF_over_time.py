import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from PFC_analysis import get_stim_RDMs
from atlas_utils import get_BN_and_resample
from fMRI_proc import get_ROI_vecs, within_run_to_nan
from organize_bhv import get_trial_info
from utils import stdize


def sanity_test(sn='102'):
    atlas = get_BN_and_resample(combine_bilateral=False)
    df_sn = get_trial_info(sn)
    # ROI2vecs, _ = load_and_get_ROI_vecs(sn, atlas, fp_fMRI_col='obj_fMRI')
    # img = image.load_img(df_sn['obj_fMRI']).get_fdata()
    # ROI2vecs, _ = get_ROI_vecs_(atlas['ROIs'], atlas['ROI_nums'], atlas,
    #                             img, atlas['ROI_regions'])

    ROI2vecs = get_ROI_vecs(sn, atlas, 'obj_fMRI', df_sn,
                            nan_thresh=.25, org_by_region=False, inc=None)

    RDM_stims = get_stim_RDMs(df_sn, semantic=False, DNN_layer=2)
    RDM_stim = RDM_stims['obj']
    # RDM_stim = get_stim_RDM_lifu(df_sn)
    # RDM_stim = np.random.normal(size=RDM_stim.shape)
    # RDM_stim_flat = RDM_stim[np.tril_indices_from(RDM_stim, k=-1)]
    all_data = []
    for ROI, vecs in tqdm(ROI2vecs.items(), desc='looping ROIs'):
        fMRI_RDM = np.corrcoef(vecs)
        fMRI_RDM = within_run_to_nan(fMRI_RDM)

        # if ROI == '199 LOC_L_4_1':
        # fMRI_RDM = regress_out_within_across(fMRI_RDM)
        #     title = f'fMRI RDM, subject: {sn}, ROI: {ROI}\n' \
        #             f'Only examine between-run'
        #     plt.title(title)
        #     plt.imshow(fMRI_RDM)
        #     plt.xlabel('trial x')
        #     plt.ylabel('trial y')
        #     plt.colorbar()
        #     plt.tight_layout()
        #     plt.show()
        #     quit()
        # else:
        #     continue

        # trial_per_run = fMRI_RDM.shape[0] // 3
        # for run in range(3):
        #     low = run * trial_per_run
        #     high = (run + 1) * trial_per_run
        #     fMRI_RDM[low:high, low:high] = np.nan

        trial_rs = []
        for i in range(fMRI_RDM.shape[0]):
            fMRI_vec_std = np.delete(fMRI_RDM[i, :], i)
            stim_vec_std = np.delete(RDM_stim[i, :], i)
            fMRI_vec_std = stdize(fMRI_vec_std, nans=True)
            stim_vec_std = stdize(stim_vec_std, nans=True)

            fMRI_nans = np.isnan(fMRI_vec_std)
            stim_nans = np.isnan(stim_vec_std)
            either_nan = fMRI_nans | stim_nans
            fMRI_vec_std = fMRI_vec_std[~either_nan]
            stim_vec_std = stim_vec_std[~either_nan]
            r = (fMRI_vec_std @ stim_vec_std) / len(fMRI_vec_std)
            trial_rs.append(r)
        all_data.append(trial_rs)
    all_data = np.vstack(all_data)
    plt.imshow(all_data)
    # find_temporal_correlation(all_data)
    plt.xlabel('Item')
    plt.ylabel('ROI')
    # plt.title(f'IRAF. subject: {sn}, 2nd layer DNN\nAnalysis of all trials')
    # plt.title(f'IRAF. subject: {sn}, 2nd layer DNN\nRegress out within vs. between')
    plt.title(f'IRAF. subject: {sn}, 2nd layer DNN\nOnly examine between-run')

    # plt.title('IRAF based on only within-run\nIRAF for 2nd layer DNN, participant 102')
    # plt.title('Shuffled trials\nIRAF for 2nd layer DNN, participant 102')
    plt.colorbar()
    plt.show()
    quit()
