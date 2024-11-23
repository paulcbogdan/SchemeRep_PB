import os.path

import numpy as np

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
from nilearn import image
import matplotlib.pyplot as plt
import scipy.stats as stats
from time import time

def idxs2img(vox_dist2MI, vox_idxs, max_vox=None, min_vox=None,
             downsample_rate=2):
    # vox_dist2MI = ndimage.gaussian_filter1d(vox_dist2MI, sigma=2, axis=-1)
    # print(vox_dist2MI.shape)
    # # print(vox_dist2MI.shape)
    # # quit()
    #
    # print(f'{np.nanmean(vox_dist2MI)=:.3f}')
    # plt.imshow(vox_dist2MI, aspect='auto')
    # plt.show()
    # quit()
    if max_vox is not None:
        vox_dist2MI = vox_dist2MI[:, :max_vox]
    if min_vox is not None:
        vox_dist2MI = vox_dist2MI[:, min_vox:]
    #
    # peak_dists = np.nanmean(vox_dist2MI[:, :], axis=-1)
    # peak_dists = (np.nanmean(vox_dist2MI[:, :3], axis=-1) -
    #               np.nanmean(vox_dist2MI[:, 3:10], axis=-1))

    peak_dists = np.argmax(vox_dist2MI, axis=-1) + min_vox
    # plt.hist(peak_dists)
    # plt.show()
    # quit()

    # img = np.full((97, 115, 97), np.nan)
    if downsample_rate == 3:
        img_data = np.zeros((32, 38, 32))
    elif downsample_rate == 2:
        img_data = np.zeros((48, 57, 48))
    else:
        img_data = np.zeros((97, 115, 97))
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
        img_data[*vox_idx] = peak_dist
    # img_data
    if downsample_rate:
        img_data = do_int_upsample(img_data, downsample_rate, None)
        # num_not_nans = np.sum(~np.isnan(img_data) & (img_data > 0))
        # print(f'Highest: {num_not_nans=}')
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

def plot_voxs(region='OC_IT_L', semantic=True, layer=None,
              max_vox=20, subtract_vox=False, subtract_other=False,
              min_vox=4):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'vis7_fMRI']  # 'con7_fMRI',
    # sns = sns[5::6]
    dist2MI_all = []
    img_data_l = []

    for i, sn in enumerate(sns):
        # if sn in ['135', '136', '203', '204']:
        #     continue
        # if sn == '205': break
        # if sn == '104': break
        sn_img_datas = []
        sn_dist2MI = []
        for j, fp in enumerate(fps):
            # stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
            #                               layer=layer)
            # stim_M = np.nanmean(stim_RSM) * 10_000
            # print(f'{sn}, {fp}: {stim_M=:.5f}')

            # fp = get_default_fp(get_sn_fp_stim_RSM, {'sn': sn, 'fp': fp,
            #                                          'semantic': semantic,
            #                                          'layer': layer})

            fp_out = get_default_fp(None, {'sn': sn, 'fp': fp, 'semantic': semantic, 'layer': layer,
                                       'region': region},
                                generate_RSA_size_map, 'cache', 0)
            if not os.path.exists(fp_out):
                print(f'Missing: {fp_out}')
                continue

            vox_Ms, valid_voxel_idxs2 = utils.pickle_wrap(generate_RSA_vox_map, None,
                                                          kwargs={'sn': sn, 'fp': fp,
                                                                  'region': region,
                                                                  'semantic': semantic,
                                                                  'layer': layer},
                                                          verbose=-1, easy_override=False)
            vox_Ms *= 10_000

            same_Ms, valid_voxel_idxs, has_nans = utils.pickle_wrap(generate_RSA_size_map, None,
                                                          kwargs={'sn': sn, 'fp': fp,
                                                                  'region': region,
                                                                  'semantic': semantic,
                                                                  'layer': layer},
                                                          verbose=0, easy_override=False)
            same_Ms *= 10_000

            same_Ms = np.nanmean(same_Ms, axis=0)

            vox_Ms = same_Ms - vox_Ms
            # print(f'{sn}, {fp}: {np.nanmean(vox_Ms)=:.5f}')
            synergy = np.nanmean(vox_Ms, axis=0)
            print(f'{sn}, {fp}: {np.nanmean(synergy)=:.5f}')
            # r, p = stats.spearmanr(vox_Ms, same_Ms, nan_policy='omit')
            # print(f'{sn}, {fp}: {r=:.5f}')


            # cnt_nans = np.sum(np.isnan(vox_Ms))
            # print(F'{sn}, {fp}, {cnt_nans=}, {vox_Ms.shape=}')



            # vox_Ms *= 10_000
            # vox_Ms -= stim_M

            # cnt_zero = np.sum(vox_Ms == 0)
            # print(F'\t{cnt_zero=}')

            # lowest = np.nanmin(vox_Ms)
            # plt.hist(vox_Ms)
            # plt.show()
            # print(f'{sn}, {fp}, {lowest=}')

            # img_data = np.zeros((97, 115, 97))
            img_data = np.full((97, 115, 97), np.nan)
            for vox_M, vox_idx in zip(vox_Ms, valid_voxel_idxs2):
                img_data[*vox_idx] = vox_M
            sn_img_datas.append(img_data)

        if not len(sn_img_datas):
            continue
        img_data = np.nanmean(sn_img_datas, axis=0)
        img_data_l.append(img_data)
        num_non_nans = np.sum(~np.isnan(img_data))# & (img_data > 0))
        print(f'{num_non_nans=}')

        # if len(img_data_l) > 2:
        #     break


    img_data_l = np.array(img_data_l)

    n_subjects = img_data_l.shape[0]
    print(n_subjects)
    cnt_non_nan = np.sum(~np.isnan(img_data_l), axis=0)
    print(f'{cnt_non_nan.shape=}')

    # view = plotting.view_img(image.new_img_like(get_atlas()['maps'], cnt_non_nan),
    #                          threshold=0)#, symmetric_cmap=False, cmap='Reds')
    # view.open_in_browser()

    # print(cnt_non_nan.shape)
    # quit()
    # img_data_l[:, cnt_non_nan < (n_subjects // 2)] = np.nan

    num_non_nans = np.sum(~np.isnan(img_data_l))
    print(f'{num_non_nans=}')

    img_data = np.nanmedian(img_data_l, axis=0)

    img_data = stats.zscore(img_data, axis=None, nan_policy='omit')

    num_non_nans = np.sum(~np.isnan(img_data))
    print(f'{num_non_nans=}')

    img_data = gaussian_filter_ignore_nan(img_data, 3)

    img = image.new_img_like(get_atlas()['maps'], img_data)

    num_non_nans = np.sum(~np.isnan(img.get_fdata()) & (img.get_fdata() > 0))
    print(f'{num_non_nans=}')
    # num_non_nans = np.sum(~np.isnan(img.get_fdata()))
    # print(f'{num_non_nans=}')

    # M = np.nanmean(img_data_l)
    # SD = np.nanstd(img_data_l)
    # plt.title(f'{M=:.1f}, {SD=:.1f}')
    # print(img_data_l.shape)
    # plt.hist(img_data_l.flatten())
    # plt.show()
    # quit()
    view = plotting.view_img(img, threshold=0)#, symmetric_cmap=False, cmap='Reds')
    view.open_in_browser()

def report_text(dist2MI_all, max_vox):
    M = np.nanmean(dist2MI_all, axis=0)
    super_M = np.nanmean(M)
    print(f'{super_M=}')
    SE = np.nanstd(dist2MI_all, axis=0) / np.sqrt(len(dist2MI_all))
    t = M / SE
    for dist in range(max_vox):
        print(f'{dist}: {M[dist]=:.2f}, {SE[dist]=:.2f}, {t[dist]=:.2f}')

def plot_map_region(region='OC_IT_L', semantic=True, layer=None,
                    max_vox=10, subtract_vox=True, subtract_other=False,
                    min_vox=1, downsample_rate=2, median_cond2=False,
                    stdize_vol=True, cross_only=False):
    if 'OC_IT' in region:
        max_vox_ = 50
    else:
        max_vox_ = 30

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # sns = sns[5::6]
    dist2MI_all = []
    img_data_l = []

    for i, sn in enumerate(sns):
        if sn in ['214']: break
        sn_img_datas = []
        sn_dist2MI = []

        for j, fp in enumerate(fps):
            kw = {'sn': sn, 'fp': fp, 'semantic': semantic, 'layer': layer,
                  'region': region, 'subtract_vox': subtract_vox, 'max_vox': max_vox_,
                  'median_cond2': median_cond2, 'stdize_by_run': True,
                  'subtract_other': subtract_other, 'downsample_rate': downsample_rate,
                  'stdize_vol': stdize_vol, 'cross_only': cross_only,
                  'do_itr': True}

            # fp_out = get_default_fp(None, kw, get_dist2MI, 'cache', 0)
            # if not os.path.exists(fp_out):
            #     print(f'Missing: {fp_out}')
            #     continue

            vox_dist2MI, _, vox_idxs, cnter = (
                utils.pickle_wrap(get_dist2MI, None, kwargs=kw,
                                  verbose=-1, easy_override=False))


            dist2MI = np.nanmean(vox_dist2MI, axis=0)

            stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                          layer=layer)
            sn_dist2MI.append(dist2MI - np.nanmean(stim_RSM))


            img_data = idxs2img(vox_dist2MI, vox_idxs, max_vox=max_vox, min_vox=min_vox,
                                downsample_rate=downsample_rate)
            sn_img_datas.append(img_data)

        if not len(sn_dist2MI):
            continue

        sn_dist2MI = np.nanmean(np.array(sn_dist2MI), axis=0)
        dist2MI_all.append(sn_dist2MI)



        img_data = np.nanmean(sn_img_datas, axis=0)
        # num_not_nans = np.sum(~np.isnan(img_data))
        # print(f'Second: {num_not_nans=}')
        sn_img_datas.append(img_data)
        img_data_l.append(img_data)

    img_data_l = np.array(img_data_l)
    num_subj = img_data_l.shape[0]
    # print(f'Number of subjects: {num_subj}')
    report_text(dist2MI_all, max_vox)

    img_data_l[img_data_l == 0] = np.nan
    img_data = np.nanmean(img_data_l, axis=0)


    print(f'Number of subjects: {num_subj}')

    print(f'{img_data_l.shape=}')
    print(f'{num_subj=}')

    # img_data = np.nanmean(img_data_l, axis=0) / np.nanstd(img_data_l, axis=0) * np.sqrt(num_subj)


    # img_data = stats.zscore(img_data, axis=None, nan_policy='omit')
    # img_data = gaussian_filter_ignore_nan(img_data, 2)

    # img_data[np.nanquantile(img_data, 0.05) > img_data] = np.nanquantile(img_data, 0.05)
    # img_data[np.nanquantile(img_data, 0.95) < img_data] = np.nanquantile(img_data, 0.95)
    # print(np.sum(~np.isnan(img_data)))

    vmin = np.nanquantile(img_data, 0.01)
    vmax = np.nanquantile(img_data, 0.99)
    # print(np.nanmin(img_data))
    # quit()
    print(f'{vmin=:.3f} {vmax=:.3f}')

    img_data[np.isnan(img_data)] = 0

    img = image.new_img_like(get_atlas()['maps'], img_data)

    view = plotting.view_img(img, threshold=0, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             vmin=vmin, vmax=vmax)
    view.open_in_browser()

if __name__ == '__main__':

    plot_map_region(region='OC_IT_L', semantic=False, layer=0)
    # plot_map_region(region='OC_IT_L', semantic=True, layer=None,)
    # plot_map_region(region='Occipital_L', semantic=False, layer=0,)


    # TODO: Test (both - vox1):vox2 ratio? Is vox2 helping encode the same items vox1 is informing?



