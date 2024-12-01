import os.path

import utils
from Study1A.load_Study1A_funcs import get_ROI_vecs
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import get_default_fp
from connRSA.jit_funcs import do_int_downsample
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from connRSA.map_size import GLOBAL_NAN_VALUE
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from time import time
from tqdm import tqdm
from nilearn import plotting, image
import nibabel as nib

def downsample_brain_image(input_image, downsample_rate=2, target_affine=None):
    """
    Downsample a brain image by changing the voxel size.

    Parameters:
    -----------
    input_image : nibabel.Nifti1Image or str
        Input brain image (either Nifti image or path to Nifti file)
    target_affine : numpy.ndarray, optional
        Target affine matrix. If None, will be automatically calculated.

    Returns:
    --------
    nibabel.Nifti1Image
        Downsampled brain image
    """
    # If input is a string (file path), load the image
    if isinstance(input_image, str):
        input_image = nib.load(input_image)

    # If target_affine is not provided, create it by doubling voxel sizes
    if target_affine is None:
        original_affine = input_image.affine
        # Create a new affine matrix with doubled voxel sizes
        target_affine = original_affine.copy()
        target_affine[0, 0] *= downsample_rate  # x-dimension
        target_affine[1, 1] *= downsample_rate  # y-dimension
        target_affine[2, 2] *= downsample_rate  # z-dimension

    # Resample the image
    downsampled_image = image.resample_img(
        input_image,
        target_affine=target_affine,
        interpolation='nearest'  # Use mean interpolation for downsampling
    )

    return downsampled_image

@njit(fastmath=True, nopython=True, cache=True)
def get_ROI_synergy_median(vecs_bool, stim_RSM, nan_mask, stim_is_nan,
                           voxels_mat=None):

    n_trials = vecs_bool.shape[0]
    n_voxels = vecs_bool.shape[1]

    I_both_all = 0
    I_same0_all = 0
    I_same1_all = 0
    skips = 0

    I_both_all_mat = np.zeros((n_voxels, n_voxels), dtype=np.float32)
    I_same0_all_mat = np.zeros((n_voxels, n_voxels), dtype=np.float32)
    I_same1_all_mat = np.zeros((n_voxels, n_voxels), dtype=np.float32)

    for v0 in range(n_voxels):
        t2vox0 = np.empty(n_trials, dtype=np.bool_)
        for t in range(n_trials):
            t2vox0[t] = vecs_bool[t, v0]

        # print(t2vox0)
        # print(vecs_bool[:, v0])
        # quit()
        center_is_zero = np.argwhere(~vecs_bool[:, v0])
        # print(center_is_zero)
        cnt_zero = center_is_zero.shape[0]
        # print(cnt_zero)
        # print(len(center_is_zero))
        # quit()
        center_is_one = np.argwhere(vecs_bool[:, v0])
        cnt_one = center_is_one.shape[0]
        # voxel_Ms = np.empty((2, n_voxels))

        # TODO: Change this to not be a symmetric triangle
        for v1 in range(n_voxels):
            # print(1)

            t_at_is_zero = np.full(n_trials, 0, dtype=np.float32)
            for i in range(cnt_zero):
                t = center_is_zero[i][0]
                t_at_is_zero[i] = voxels_mat[t, v1]
            zero_median = np.median(t_at_is_zero[:cnt_zero])
            # print(2)
            t_at_is_one = np.full(n_trials, 0, dtype=np.float32)
            for i in range(cnt_one):
                t = center_is_one[i][0]
                t_at_is_one[i] = voxels_mat[t, v1]
            # print(t_at_is_one)
            one_median = np.median(t_at_is_one[:cnt_one])
            # print(3)

            t2vox1 = np.empty(n_trials, dtype=np.int8)
            for t in range(n_trials):
                if t2vox0[t] == 0:
                    t2vox1[t] = voxels_mat[t, v1] > zero_median
                else:
                    t2vox1[t] = voxels_mat[t, v1] > one_median

                # print(1111)
                # return
                # vox0 = t2vox0[t]
                # print(vox0)
                # quit()
                # # print('toast')
                # # print(vox0)
                # # print(v1)
                # # print(voxel_Ms[vox0][v1].shape)
                # # # print(voxel_Ms[vox0, v1])
                # # print('test')
                # # print(voxels_mat[t, v1])
                # t2vox1[t] = voxels_mat[t, v1] > voxel_Ms[vox0, v1]
            # print(t2vox0)
            # print(np.mean(t2vox0))
            # print(t2vox1)
            # print(np.sum(t2vox1))
            # print('-')
            # print(np.mean(t2vox1))
            # quit()
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
                    # print(f'{t0=}, {t1=}')
                    if stim_is_nan[t0, t1]:
                        break

                    if t2vox0[t0] == t2vox0[t1]:
                        n_same0 += 1
                        I_same0 += stim_RSM[t0, t1]
                        if t2vox1[t0] == t2vox1[t1]:
                            n_same1 += 1
                            I_same1 += stim_RSM[t0, t1]
                            n_both += 1
                            I_both += stim_RSM[t0, t1]
                    else:
                        if t2vox1[t0] == t2vox1[t1]:
                            n_same1 += 1
                            I_same1 += stim_RSM[t0, t1]

            if n_same0 == 0 or n_same1 == 0:
                skips += 1
                continue
            if n_both == 0:
                skips += 1
                continue

            I_same0 /= n_same0
            I_same1 /= n_same1
            I_both /= n_both
            I_both_all += I_both
            I_same0_all += I_same0
            I_same1_all += I_same1

            I_both_all_mat[v0, v1] = I_both
            I_same0_all_mat[v0, v1] = I_same0
            I_same1_all_mat[v0, v1] = I_same1

            I_both_all_mat[v1, v0] = I_both
            I_same0_all_mat[v1, v0] = I_same0
            I_same1_all_mat[v1, v0] = I_same1

    return (I_both_all, I_same0_all, I_same1_all,
            I_both_all_mat, I_same0_all_mat, I_same1_all_mat,
            skips)




@njit(fastmath=True, nopython=True, cache=True)
def get_ROI_synergy(vecs_bool, stim_RSM, nan_mask, stim_is_nan):
    n_trials = vecs_bool.shape[0]
    n_voxels = vecs_bool.shape[1]

    I_both_all = 0
    I_same0_all = 0
    I_same1_all = 0
    skips = 0

    for v0 in range(n_voxels):
        for v1 in range(v0):
            # n_both = 0
            # n_same0 = 0
            # n_same1 = 0

            n_both_equal = 0
            n_both_unequal = 0
            n_same1_equal = 0
            n_same1_unequal = 0
            n_same0_equal = 0
            n_same0_unequal = 0

            # I_both = 0
            # I_same0 = 0
            # I_same1 = 0

            I_both_equal = 0
            I_both_unequal = 0
            I_same1_equal = 0
            I_same1_unequal = 0
            I_same0_equal = 0
            I_same0_unequal = 0
            for t0 in range(n_trials):
                if nan_mask[t0, v0] or nan_mask[t0, v1]:
                    continue
                for t1 in range(t0):
                    if nan_mask[t1, v0] or nan_mask[t1, v1]:
                        continue
                    if stim_is_nan[t0, t1]:
                        break

                    if vecs_bool[t0, v0] == vecs_bool[t1, v0]:
                        # n_same0 += 1
                        # I_same0 += stim_RSM[t0, t1]

                        if vecs_bool[t0, v0] == vecs_bool[t0, v1]:
                            n_same0_equal += 1
                            I_same0_equal += stim_RSM[t0, t1]
                        else:
                            n_same0_unequal += 1
                            I_same0_unequal += stim_RSM[t0, t1]

                        if vecs_bool[t0, v1] == vecs_bool[t1, v1]:
                            # n_same1 += 1
                            # I_same1 += stim_RSM[t0, t1]
                            # n_both += 1
                            # I_both += stim_RSM[t0, t1]

                            if vecs_bool[t0, v0] == vecs_bool[t0, v1]:
                                n_same1_equal += 1
                                I_same1_equal += stim_RSM[t0, t1]
                                n_both_equal += 1
                                I_both_equal += stim_RSM[t0, t1]
                            else:
                                n_same1_unequal += 1
                                I_same1_unequal += stim_RSM[t0, t1]
                                n_both_unequal += 1
                                I_both_unequal += stim_RSM[t0, t1]

                    else:
                        if vecs_bool[t0, v1] == vecs_bool[t1, v1]:
                            # n_same1 += 1
                            # I_same1 += stim_RSM[t0, t1]

                            if vecs_bool[t0, v0] == vecs_bool[t0, v1]:
                                n_same1_equal += 1
                                I_same1_equal += stim_RSM[t0, t1]
                            else:
                                n_same1_unequal += 1
                                I_same1_unequal += stim_RSM[t0, t1]

            # if n_same0 == 0 or n_same1 == 0:
            #     skips += 1
            #     continue
            # if n_both == 0:
            #     skips += 1
            #     continue

            if n_both_equal == 0 or n_both_unequal == 0 or n_same1_equal == 0 or n_same1_unequal == 0 \
                    or n_same0_equal == 0 or n_same0_unequal == 0:
                skips += 1
                continue
            # if n_same1_equal == 0 or n_same1_unequal == 0:
            #     skips += 1
            #     continue

            I_same0 = I_same0_equal / n_same0_equal + I_same0_unequal / n_same0_unequal
            I_same1 = I_same1_equal / n_same1_equal + I_same1_unequal / n_same1_unequal
            I_both = I_both_equal / n_both_equal + I_both_unequal / n_both_unequal

            # I_same0 /= n_same0
            # I_same1 /= n_same1
            # I_both /= n_both

            I_same0_all += I_same0
            I_same1_all += I_same1
            I_both_all += I_both

    return I_both_all, I_same0_all, I_same1_all, skips

# def load_and_mask(img, ROI, atlas, downsample_rate=2, stdize_vol=True):
#     if downsample_rate:
#         nan_val = GLOBAL_NAN_VALUE
#         mask = atlas['maps'].get_fdata() == atlas['ROIs'].index(ROI) + 1
#         mask = mask[..., None]
#         print(mask.shape)
#         quit()
#         img[np.isnan(img)] = nan_val
#
#         img = do_int_downsample(img, downsample_rate, mask,
#                                 nan_val=nan_val, fourD_mask=False)
#         img[img == nan_val] = np.nan
#         num_not_nan = np.sum(~np.isnan(img))
#         if stdize_vol:
#             img = utils.stdize(img, stdize_by_run=False, axis=(0, 1, 2),
#                                nans=True)
#         print(img.shape)
#         quit()
#     else:
#         mask = atlas['maps'].get_fdata() == atlas['ROIs'].index(ROI) + 1
#         img = img[mask]
#         num_not_nan = np.sum(~np.isnan(img))
#         if stdize_vol:
#             img = utils.stdize(img, stdize_by_run=False, axis=0, nans=True)

def do_ROI_synergy_sn(sn, fp, just_ROIs=None, semantic=True, layer=None,
                      double_median=False, downsample_rate=2,
                      stdize_vol=True, combine_regions=True):
    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_RSM = stats.zscore(stim_RSM, axis=None, nan_policy='omit')
    stim_is_nan = np.isnan(stim_RSM)

    atlas = get_atlas(combine_regions=combine_regions)
    atlas_map = atlas['maps']
    df_sn = get_trial_info(sn, easy_override=True)
    df_sn, sess = sort_df_sn(df_sn, fp)
    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp])

    if downsample_rate:
        mask = np.isnan(img)
        img[mask] = GLOBAL_NAN_VALUE
        img = image.new_img_like(atlas_map, img)
        img = downsample_brain_image(img, downsample_rate=downsample_rate)
        img = img.get_fdata()
        img[img == GLOBAL_NAN_VALUE] = np.nan

    if downsample_rate:
        atlas_map = downsample_brain_image(atlas_map, downsample_rate=downsample_rate)


    # ROI2vecs = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.0)

    ROIs = atlas['ROIs']
    synergies = []
    I_both_all_all = []
    I_same0_all_all = []
    I_same1_all_all = []

    I_both_mat_all = []
    I_same0_mat_all = []
    I_same1_mat_all = []
    for i, ROI in tqdm(enumerate(ROIs)):
        # print(ROIs)
        # quit()
        # if i < 244: continue
        vecs = img[atlas_map.get_fdata() == i + 1].T
        if stdize_vol:
            vecs = utils.stdize(vecs, axis=1, nans=True)

        # print(vecs.shape)
        # quit()
        if just_ROIs:
            if ROI not in just_ROIs:
                continue

        vecs = utils.stdize(vecs, stdize_by_run=True, axis=0, nans=True)
        # print(vecs.shape)
        # quit()
        # print(vecs[:, 80])
        all_nan = np.all(np.isnan(vecs), axis=0)
        vecs = vecs[:, ~all_nan]
        vecs_bool = vecs > np.nanmedian(vecs, axis=0)
        all_one = np.all(vecs_bool, axis=0)
        all_zero = np.all(~vecs_bool, axis=0)
        all_same = all_one | all_zero
        vecs = vecs[:, ~all_same]
        # print(vecs.shape)
        # quit()
        if vecs.shape[1] < 10:
            I_both_mat_all.append(np.nan)
            I_same0_mat_all.append(np.nan)
            I_same1_mat_all.append(np.nan)
            I_both_all_all.append(None)
            I_same0_all_all.append(None)
            I_same1_all_all.append(None)
            synergies.append(np.nan)
            continue


        # plt.imshow(vecs, aspect='auto')
        # plt.show()
        # plt.imshow(vecs_bool, aspect='auto')
        # plt.show()
        # quit()
        # print(vecs_bool.shape)
        #
        # print(np.mean(vecs_bool, axis=0))
        # print(np.mean(vecs_bool, axis=1))
        # quit()


        # print(np.sum(vecs_bool))
        # quit()

        nan_mask = np.isnan(vecs)
        # print(vecs_bool)
        # plt.imshow(vecs_bool)
        # plt.show()
        # quit()

        if double_median:
            (I_both_all, I_same0_all, I_same1_all,
             I_both_all_mat, I_same0_all_mat, I_same1_all_mat,
             skips) = get_ROI_synergy_median(vecs_bool, stim_RSM, nan_mask,
                                             stim_is_nan, voxels_mat=vecs,)
            I_both_mat_all.append(I_both_all_mat)
            I_same0_mat_all.append(I_same0_all_mat)
            I_same1_mat_all.append(I_same1_all_mat)
        else:
            I_both_all, I_same0_all, I_same1_all, skips = (
                get_ROI_synergy(vecs_bool, stim_RSM, nan_mask, stim_is_nan))

        synergy = I_both_all - I_same0_all - I_same1_all
        synergies.append(synergy)
        I_both_all_all.append(I_both_all)
        I_same0_all_all.append(I_same0_all)
        I_same1_all_all.append(I_same1_all)

    return (synergies,
            I_both_all_all, I_same0_all_all, I_same1_all_all,
            I_both_mat_all, I_same0_mat_all, I_same1_mat_all)

def get_synergy_score(sn, fp, median, semantic, layer,
                      stdize_vol=False, downsample_rate=2,
                      combine_regions=True
                      ):
    (synergies,
     I_both_all_all, I_same0_all_all, I_same1_all_all,
     I_both_mat_all, I_same0_mat_all, I_same1_mat_all) = (
        utils.pickle_wrap(do_ROI_synergy_sn,
                          kwargs={'sn': sn, 'fp': fp,
                                  'double_median': median,
                                  'semantic': semantic,
                                  'layer': layer,
                                  'stdize_vol': stdize_vol,
                                  'downsample_rate': downsample_rate,
                                  'combine_regions': combine_regions
                                  },
                          easy_override=False))
    return synergies

def do_ROI_synergy(median=True, semantic=True, layer=None,
                   stdize_vol=True, downsample_rate=4,
                   combine_regions=True):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI'] #
    # sns = sns[::-1]
    # sns = sns[4::5]
    # print(sns)
    # quit()
    atlas = get_atlas(combine_regions=combine_regions)
    all_synergies = []
    # for fp in fps:
    for sn in sns:
        # if sn != '108':
        #     continue
        sn_synergies = []
        for fp in fps:
            # if fp != 'obj7_fMRI':
            #     continue
            print(f'Onto: {sn}, {fp}')
            t_st = time()

            fp_pkl = get_default_fp(None, {'sn': sn, 'fp': fp,
                                           'median': median,
                                           'semantic': semantic,
                                           'layer': layer,
                                           'stdize_vol': stdize_vol,
                                           'downsample_rate': downsample_rate,
                                           'combine_regions': combine_regions
                                           },
                                get_synergy_score, 'cache', 0)
            # if not os.path.exists(fp_pkl):
            #     print('Skip')
            #     continue
            synergies = utils.pickle_wrap(get_synergy_score,
                                          kwargs={'sn': sn, 'fp': fp,
                                                  'median': median,
                                                  'semantic': semantic,
                                                  'layer': layer,
                                                  'stdize_vol': stdize_vol,
                                                  'downsample_rate': downsample_rate,
                                                  'combine_regions': combine_regions
                                                  },
                                          easy_override=False)
            if len(synergies) < 10 or None in synergies:
                synergies = utils.pickle_wrap(get_synergy_score,
                                              kwargs={'sn': sn, 'fp': fp,
                                                      'median': median,
                                                      'semantic': semantic,
                                                      'layer': layer,
                                                      'stdize_vol': stdize_vol,
                                                      'downsample_rate': downsample_rate
                                                      },
                                              easy_override=True)
            # quit()

            t_en = time()
            print(f'{sn}, {fp}: {t_en - t_st:.2f}s')
            sn_synergies.append(synergies)
        sn_synergies = np.array(sn_synergies)
        sn_synergies = np.nanmean(sn_synergies, axis=0)
        sn_synergies -= np.nanmean(sn_synergies)

        if len(sn_synergies.shape) == 0:
            continue
        if len(sn_synergies) == 0:
            print('OOOOOOP')
            continue
        all_synergies.append(sn_synergies)


    ts = []
    ps = []
    for ROI_i, ROI in enumerate(atlas['ROIs']):
        ROI_synergies = [all_synergies[sn][ROI_i] for
                         sn in range(len(all_synergies))]
        # print(ROI_synergies)
        # quit()
        M = np.nanmean(ROI_synergies)
        # if np.isnan(M):
        #     print(ROI_synergies)
        #     quit()
        # quit()
        N = np.sum(~np.isnan(ROI_synergies))
        SE = np.nanstd(ROI_synergies) / np.sqrt(len(sns))
        t = stats.ttest_1samp(ROI_synergies, 0, nan_policy='omit')
        p = t.pvalue
        t = t.statistic
        if not np.isnan(t):
            print(f'{ROI}: {M=:.3f} [{SE:.3f}], t[{N-1}]={t:.3f}, {p=:.3f}')
        else:
            print(f'{ROI}: Just NaN test')
        ts.append(t)
        ps.append(p)

    print('----')
    img_data = np.zeros(atlas['maps'].shape)
    for i, (t, p) in enumerate(zip(ts, ps)):
        if t > 2:
            print(f'{atlas["ROIs"][i]}: t = {t:.3f}, p = {p * len(atlas["ROIs"]):.3f}')
        img_data[atlas['maps'].get_fdata() == i + 1] = t

    # vmin = np.nanquantile(img_data, 0.05)
    # vmax = np.nanquantile(img_data, 0.95)

    img = image.new_img_like(atlas['maps'], img_data)


    fig, axs = plotting.plot_img_on_surf(img, threshold=0,
                                         # vmin=vmin,
                                         # vmax=vmax,
                                         inflate=False,
                                         surf_mesh='fsaverage5',
                                         avg_method='median')
    plt.show()


if __name__ == '__main__':
    do_ROI_synergy()
    # do_ROI_synergy(semantic=False, layer=0)

    # do_ROI_synergy(median=False)
    # do_ROI_synergy(semantic=False, layer=0, median=False)
