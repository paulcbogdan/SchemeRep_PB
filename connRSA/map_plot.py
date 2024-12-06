import os.path

import numpy as np
from nilearn.experimental.surface import load_fsaverage

from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import get_default_fp
from connRSA.gaussian_brain import gaussian_filter_ignore_nan
from connRSA.jit_funcs import do_int_upsample
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from connRSA.map_size import get_dist2MI, generate_RSA_size_map
from org_sns import get_sns
import utils
from nilearn import plotting
from scipy import ndimage
from nilearn import image, surface
import matplotlib.pyplot as plt
import scipy.stats as stats
from time import time
# from nilearn.surface import SurfaceImage

def idxs2img(vox_dist2MI, vox_idxs, max_vox=None, min_vox=None,
             downsample_rate=2):
    # for i in range(vox_dist2MI.shape[0]):
    # vox_dist2MI = gaussian_filter_ignore_nan(vox_dist2MI, 2, d1=True, axis=1)
    # print('Filtered')

    if max_vox is not None:
        vox_dist2MI = vox_dist2MI[:, :max_vox]
    if min_vox is not None:
        vox_dist2MI = vox_dist2MI[:, min_vox:]
    #
    peak_dists = (np.nanmean(vox_dist2MI[:, :5], axis=-1) -
                  np.nanmean(vox_dist2MI[:, 5:10], axis=-1))

    # peak_dists = np.nanmean(vox_dist2MI[:, :], axis=-1)
    # peak_dists = (np.nanmean(vox_dist2MI[:, :3], axis=-1) -
    #               np.nanmean(vox_dist2MI[:, 3:], axis=-1))
    # peak_dists = np.nanmean(vox_dist2MI[:, :4], axis=-1)
    peak_dists = np.nanmean(vox_dist2MI[:, :5], axis=-1)

    # peak_dists = np.nanmean(vox_dist2MI[:, 3:], axis=-1)
    # print(vox_dist2MI.shape)
    # quit()
    # print(vox_dist2MI)
    # quit()
    # peak_dists = np.nanmean(vox_dist2MI[:, 0:15], axis=-1)
    # print(peak_dists)
    # quit()
    # peak_dists -= np.nanmean(peak_dists)
    # print(peak_dists.shape)
    # quit()


    # img = np.full((97, 115, 97), np.nan)
    # if downsample_rate == 3:
    #     img_data = np.zeros((32, 38, 32))
    # elif downsample_rate == 2:
    #     img_data = np.zeros((48, 57, 48))
    # else:
    #     img_data = np.zeros((97, 115, 97))

    if downsample_rate == 5:
        img_data = np.full((97 // 5, 115 // 5, 97 // 5), np.nan)
    elif downsample_rate == 4:
        img_data = np.full((97 // 4, 115 // 4, 97 // 4), np.nan)
    elif downsample_rate == 3:
        img_data = np.full((32, 38, 32), np.nan)
    elif downsample_rate == 2:
        img_data = np.full((48, 57, 48), np.nan)
    else:
        img_data = np.full((97, 115, 97), np.nan)
    for i, (peak_dist, vox_idx) in enumerate(zip(peak_dists, vox_idxs)):
        # if peak_dist > 6:
        #
        #     # print(vox_dist2MI[i])
        #     # print(vox_dist2MI.shape)
        #     # quit()
        #     plt.plot(vox_dist2MI[i])
        #     plt.show()
            # print(vox_idx)
            # quit()
        # print(vox_idxs)
        img_data[*vox_idx] = peak_dist
        # print(peak_dist)
    # img_data
    # print(f'{img_data.shape=}')
    if downsample_rate:

        img_data = do_int_upsample(img_data, downsample_rate, None)

        # num_not_nans = np.sum(~np.isnan(img_data) & (img_data > 0))
        # print(f'Highest: {num_not_nans=}')
        # print('test')
        # print(img_data)
        # quit()
        img_data_ = np.full((97, 115, 97), np.nan)
        img_data_[:img_data.shape[0], :img_data.shape[1], :img_data.shape[2]] = img_data
        img_data = img_data_


    return img_data


    # img = image.new_img_like(get_atlas()['maps'], img_data)
    #
    #
    # view = plotting.view_img(img, symmetric_cmap=False)
    # view.open_in_browser()
    #
    # quit()

def report_text(dist2MI_all, max_vox):
    M = np.nanmean(dist2MI_all, axis=0)
    super_M = np.nanmean(M)
    print(f'{super_M=}')
    SE = np.nanstd(dist2MI_all, axis=0) / np.sqrt(len(dist2MI_all))
    t = M / SE
    for dist in range(max_vox):
        print(f'{dist}: {M[dist]=:.2f}, {SE[dist]=:.2f}, {t[dist]=:.2f}')

def plot_map_region(region='OC_IT_L', semantic=True, layer=None,
                    max_vox=15, subtract_vox=True, subtract_other=False,
                    min_vox=1, downsample_rate=3, median_cond2=False,
                    stdize_vol=True, cross_only=False):
    if 'OC_IT' in region:
        max_vox_ = 50
    else:
        max_vox_ = 30

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI'] #
    # sns = sns[5::6]
    dist2MI_all = []
    img_data_l = []

    for i, sn in enumerate(sns):
        # if sn in ['108', '109', '112', '113', '117', '118']: continue
        if sn == '211': break
        sn_img_datas = []
        sn_dist2MI = []

        for j, fp in enumerate(fps):
            kw = {'sn': sn, 'fp': fp, 'semantic': semantic, 'layer': layer,
                  'region': region, 'subtract_vox': subtract_vox, 'max_vox': max_vox_,
                  'median_cond2': median_cond2, 'stdize_by_run': True,
                  'subtract_other': subtract_other, 'downsample_rate': downsample_rate,
                  'stdize_vol': stdize_vol, 'cross_only': cross_only,
                  'do_itr': False, 'dist_downplay': False,
                  'xor': False, 'RSM_RSM': True,
                  'RSM_RSM_override': True
                  }
            # kw['RSM_RSM']
            # kw['vox0_RSA'] = True
            print(kw)

            fp_out = get_default_fp(None, kw, get_dist2MI, 'cache', 0)
            if not os.path.exists(fp_out):
                print(f'Missing: {fp_out}')
                continue
            print('Go')
            # kw['RSM_RSM_override'] = False

            vox_dist2MI, _, vox_idxs, cnter = (
                utils.pickle_wrap(get_dist2MI, None, kwargs=kw,
                                  verbose=-1, easy_override=False))
            # # vox_dist2MI[cnter[:, 1] < 10] = np.nan
            vox_dist2MI[vox_dist2MI == -999_999] = np.nan
            cnter = np.array(cnter, dtype=np.float32)
            if 'cortical' in region:
                vox_dist2MI[np.sum(cnter[:, :3], axis=1) < 20] = np.nan
            # print(vox_dist2MI)
            # quit()
            # cnter[np.sum(cnter[:, :3], axis=1) < 70] = np.nan

            # vox_dist2MI = np.array(cnter, dtype=np.float32)

            # plt.imshow(cnter, aspect='auto', interpolation='none')
            # plt.show()

            # plt.hist(np.sum(cnter[:, :3], axis=1), bins=50)
            # plt.show()
            # quit()

            # Shape: (9607, 30)

            # Subtracting axis 1 = find voxels with strongest low > high effect

            # vox_dist2MI -= np.nanmean(vox_dist2MI[:, min_vox:max_vox],
            #                           axis=1)[..., None]

            # Subtracting 0 = find voxels that have the strongest effect across brain
            # vox_dist2MI -= np.nanmean(vox_dist2MI, axis=0)

            vox_dist2MI = stats.zscore(vox_dist2MI, axis=0, nan_policy='omit')

            dist2MI = np.nanmean(vox_dist2MI, axis=0)

            stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                          layer=layer)
            sn_dist2MI.append(dist2MI - np.nanmean(stim_RSM))


            img_data = idxs2img(vox_dist2MI, vox_idxs,
                                max_vox=max_vox, min_vox=min_vox,
                                downsample_rate=downsample_rate)
            sn_img_datas.append(img_data)

        if not len(sn_dist2MI):
            continue

        sn_dist2MI = np.nanmean(np.array(sn_dist2MI), axis=0)
        dist2MI_all.append(sn_dist2MI)

        img_data = np.nanmean(sn_img_datas, axis=0)
        img_data_l.append(img_data)
        # break

    img_data_l = np.array(img_data_l)
    num_subj = img_data_l.shape[0]
    # print(f'Number of subjects: {num_subj}')
    report_text(dist2MI_all, max_vox)

    img_data_l[img_data_l == 0] = np.nan
    # img_data = np.nanmean(img_data_l, axis=0)


    print(f'Number of subjects: {num_subj}')

    print(f'{img_data_l.shape=}')
    print(f'{num_subj=}')

    img_data = (np.nanmean(img_data_l, axis=0)
                / np.nanstd(img_data_l, axis=0)
                * np.sqrt(num_subj))
    # num_not_nan = np.sum(~np.isnan(img_data))
    # print(f'{num_not_nan=}')
    # quit()
    # img_data = np.nanmean(img_data_l, axis=0)

    # img_data = stats.zscore(img_data, axis=None, nan_policy='omit')
    # print(img_data.shape)
    # img_data = gaussian_filter_ignore_nan(img_data, 1)


    # num_not_nans = np.sum(~np.isnan(img_data))
    # print(f'{num_not_nans=}')

    # nan_mean = np.nanmean(img_data)
    # nan_std = np.nanstd(img_data)
    # print(f'{nan_mean=}, {nan_std=}')
    # quit()
    # quit()

    # img_data[np.nanquantile(img_data, 0.05) > img_data] = np.nanquantile(img_data, 0.05)
    # img_data[np.nanquantile(img_data, 0.95) < img_data] = np.nanquantile(img_data, 0.95)
    # print(np.sum(~np.isnan(img_data)))
    # print(img_data.shape)
    # quit()

    img_data[np.isinf(img_data)] = np.nan
    img_data[(img_data < .00000001) & (img_data > -.00000001)] = np.nan
    # print(img_data.shape)
    # quit()
    # img_data -= np.nanmean(img_data)
    if 'cortical' in region:
        vmin = np.nanquantile(img_data, 0.01)
        vmax = np.nanquantile(img_data, 0.99)
    else:
        vmin = np.nanquantile(img_data, 0.01)
        vmax = np.nanquantile(img_data, 0.99)
    # vmin = -5
    # vmax = 10
    # print(np.nanmin(img_data))

    # quit()
    print(f'{vmin=:.3f} {vmax=:.3f}')
    # vmin = -3
    # vmax = 3
    # plt.hist(img_data.flatten())
    # plt.show()
    # quit()
    img_data[np.isnan(img_data)] = 0
    img = image.new_img_like(get_atlas()['maps'], img_data)

    view = plotting.view_img(img, threshold=0, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             vmin=vmin, vmax=vmax,
                             cmap='cold_hot_r'
                             )
    view.open_in_browser()
    # quit()

    # fsaverage_meshes = load_fsaverage()

    # img = SurfaceImage.from_volume(
    #     mesh='fsaverage5',
    #     volume_img=img,
    # )

    # view = plotting.plot_surf_roi(
    #     # surf_mesh=fsaverage_meshes["inflated"],
    #     surf_mesh='fsaverage5',
    #     roi_map=img,
    #     threshold="90%",
    #     # bg_map=fsaverage_sulcal,
    #     # hemi="right",
    #     # title="3D visualization in a web browser",
    # )



    #
    fig, axs = plotting.plot_img_on_surf(img,
                                         # threshold=thresh,
                                         # cmap=cmap, title=title,
                                         # vmin=0 if only_positive else -vmax,
                                         vmax=vmax, vmin=vmin,
                                         # vmin=-10, vmax=10,
                                         inflate=False,
                                         surf_mesh='fsaverage5',
                                         avg_method='median',
                                         hemispheres=['right' if '_R' in region else 'left'],
                                         cmap='cold_hot_r',
                                         # threshold=2,
                                         )
    plt.show()
    # view.open_in_browser()

if __name__ == '__main__':

    # plot_map_region(region='cortical_L', semantic=False, layer=0,
    #                 downsample_rate=3)

    # plot_map_region(region='PFC_L', semantic=False, layer=0,
    #                 downsample_rate=2)

    plot_map_region(region='cortical_L', semantic=False, layer=0,
                    downsample_rate=2, median_cond2=False)

    # plot_map_region(region='OC_IT_L', semantic=False, layer=0,
    #                 downsample_rate=2)

    # plot_map_region(region='cortical_L', semantic=True, layer=None,
    #                 downsample_rate=3)

    # For median_cond2, if two RSMs are very independent, then a positive correlation reflects
    #   just an attention/salience/focus effect?
    #   Maybe test vox0_RSA and vox1_RSA given median_cond=True

    # TODO: Test (both - vox1):vox2 ratio? Is vox2 helping encode the same items vox1 is informing?



