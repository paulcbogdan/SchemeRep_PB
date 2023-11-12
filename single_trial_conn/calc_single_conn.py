import os

import numpy as np
from nilearn import image
from nilearn.glm.first_level import compute_regressor
from pickle_wrap import pickle_wrap

from atlas_utils import get_atlas
from fMRI_proc import within_run_to_nan
from modularity_testing import get_partition_matrix
from organize_bhv import get_trial_info, get_all_sns
from tqdm import tqdm

from permutation_test import Timer
from pathlib import Path
import gzip
import shutil

from stim import get_DNN_vecs, get_stim_RDM, get_semantic_vectors
from utils import tril_flat, stdize
import scipy.stats as stats

import matplotlib.pyplot as plt

def get_hrf():
    onset, amplitude, duration = 0.0, 1.0, 1.0
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    time_length = 20
    frame_times = np.linspace(0, time_length, time_length // 2)
    signal, _labels = compute_regressor(
        exp_condition,
        'spm',
        frame_times,
        con_id="main",
        oversampling=16,
    )
    return signal[:, 0]
    # plt.plot(range(len(signal)), signal)
    # print(len(signal))
    # plt.show()
# get_hrf()


def deconvolve_all_signals(regions_signal):
    regions_signal = regions_signal[:, 4:-4]
    regions_signal = regions_signal - np.mean(regions_signal, axis=1)[:, None]
    hrf = get_hrf()
    n_elements = regions_signal.shape[1]
    deconvolved = []
    for i in tqdm(range(regions_signal.shape[0])):
        rs = []
        deconvolved_region = []
        for t in range(regions_signal.shape[1]):
            if t + 1 == n_elements:
                rs.append(np.nan)
                continue
            t_post = n_elements - len(hrf) - t
            if t_post < 0:
                hrf = hrf[:t_post]
                hrf_t = np.hstack((np.zeros(t), hrf))
            else:
                hrf_t = np.hstack((np.zeros(t), hrf, np.zeros(t_post)))
            r, p = stats.pearsonr(regions_signal[i, :], hrf_t)
            rs.append(r)
            deconvolved_region.append(p)
        deconvolved_region = [0]*4 + deconvolved_region + [0]*4
        deconvolved.append(deconvolved_region)
        continue
        # print(regions_signal[i, :].shape)
        # print(hrf.shape)
        # print(hrf)
        # plt.plot(hrf)
        # plt.show()
        # quit()
        # test, _ = np.polydiv(regions_signal[i, :], hrf)

        # print(regions_signal[i, :])
        # test = signal.convolve(regions_signal[i, :], hrf)
        # test, _ = signal.deconvolve(test, hrf)


        # test, _ = signal.deconvolve(regions_signal[i, :], hrf)
        # test = signal.convolve(regions_signal[i, :], hrf)
        # print('deconvolved:')
        # print(test)
        # print(test.shape)
        # plt.plot(regions_signal[i, :])
        # plt.show()

        # re = signal.convolve(test, hrf)
        # print(re)

        # print(sd)
        # sd_regions = np.nanstd(regions_signal[i, :])
        # print(sd_regions)
        # regions_signal[i, :] = regions_signal[i, :] / sd_regions * sd
        # print(rs)
        # print('-')
        # print(regions_signal[i, :])
        # plt.plot(rs)
        # plt.plot(regions_signal[i, :])
        # # plt.ylim(min(regions_signal[i, :]), max(regions_signal[i, :]))
        # plt.show()
        # quit()
        # regions_signal[i, :] = signal.deconvolve(regions_signal[i, :], hrf)[0]
    deconvolved = np.array(deconvolved)
    return deconvolved
    # return regions_signal

def decompress(in_fp, out_fp):
    print(f'Decompressing: {in_fp}')
    with gzip.open(in_fp, 'r') as f_in, open(out_fp, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def get_FC_trial(regions_signal, row, name, TRs, atlas):
    onset_TR = row[f'{name}_onset_TR']
    trial_slice = slice(onset_TR, onset_TR + TRs)
    regions_trial_signal = regions_signal[:, trial_slice]

    # Could regress out across ROI (e.g., axis=0 here), although that
    #   would be most meaningful if regions are isolated (e.g., only occipital)

    mtx = np.corrcoef(regions_trial_signal)
    return mtx, regions_trial_signal

def get_mtx_sn_(sn='102', name='bl', combine_regions=False, TRs=3):
    atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    # masker = NiftiLabelsMasker(labels_img=atlas['maps'], standardize=True)
    # corr_measure = ConnectivityMeasure(kind="correlation")
    df_sn = get_trial_info(sn)

    name_remap = 'enc' if name in ['obj', 'scn'] else name

    dir_in = fr'dir_preproc/{sn}/{name_remap.lower()}'
    mtx_all = []
    regions_trials_signal = []
    for run in range(1, 4):
        df_run = df_sn[df_sn[f'{name}_run'] == run]

        t = Timer()
        fp_in = fr'{dir_in}/BOLD_run{run}.nii' # .nii is 35% time of .nii.gz
        if not Path(fp_in).exists():
            fp_in_gz = fp_in + '.gz'
            decompress(fp_in_gz, fp_in)

        img = image.load_img(fp_in)

        img_data = img.get_fdata()
        ROIs = atlas['ROIs']
        ROI_nums = atlas['ROI_nums']
        ROI_regions = atlas['ROI_regions']
        regions_signal = []
        for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
            atlas_roi = atlas['maps'].get_fdata() == ROI_num
            region_vecs = img_data[atlas_roi]
            region_signal = np.mean(region_vecs, axis=0)
            regions_signal.append(region_signal)

        regions_signal = np.array(regions_signal)
        regions_signal = deconvolve_all_signals(regions_signal)

        # print(f'Get data: {t.lap():.3f}')
        for idx, row in tqdm(df_run.iterrows()):
            mtx, regions_trial_signal = get_FC_trial(regions_signal, row, name,
                                                     TRs, atlas)
            mtx_all.append(mtx)
            regions_trials_signal.append(regions_trial_signal)
        # regions_trials_signal = np.array(regions_trials_signal)
        # print(regions_trials_signal.shape)
        # plt.imshow(regions_trials_signal)
        # plt.show()
    # quit()

    mtx_all = np.array(mtx_all)
    regions_trials_signal = list(np.array(x) for x in
                                 zip(*regions_trials_signal)) # (ROI, trial, 9p'0
    # voxel, TR)
    return mtx_all, regions_trials_signal

def get_mtx_sn(sn='102', name='bl', combine_regions=False, TRs=3,
               easy_override=False):
    print(f'test: {sn=}')
    mtx_all, bolds = pickle_wrap(None, get_mtx_sn_, kwargs={'sn': sn, 'name': name,
                                'combine_regions': combine_regions, 'TRs': TRs},
                          cache_dir='cache/mtx', easy_override=easy_override)
    return mtx_all, bolds

def do_connectivity_RSA(sn='102', name='bl', combine_regions=False, TRs=4,
                        semantic=False):


    df_sn = get_trial_info(sn)
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    RSM_stim_obj = get_stim_RDM(df_sn, d_vecs, obj_only=True)
    np.fill_diagonal(RSM_stim_obj, np.nan)

    mtx_all, _ = get_mtx_sn(sn, name, combine_regions, TRs, easy_override=True)
    # atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    # plot_connectivity(np.nanmean(mtx_all, axiss=0),
    #                   atlas['ticks'], atlas['tick_labels'],
    #                   atlas['tick_lows'], no_avg=True,
    #                   vmin=-0.4, vmax=0.4,
    #                   xlabel='', ylabel='',
    #                   title=f'Subject: {sn}, deconvolved trial TRs')

    atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    ROI_regions = atlas['ROI_regions']
    idxs_occ = [i for i, region in enumerate(ROI_regions) if
                ('EVC' in region) or ('LOC' in region) or ('sOcG' in region) or
                ('FuG' in region)]
    # idxs_occ = list(range(len(ROI_regions)))
    mtx_occ_all = get_partition_matrix(mtx_all, idxs_occ)
    flat_occ_all = tril_flat(mtx_occ_all)
    RSM_occ = np.corrcoef(flat_occ_all)
    np.fill_diagonal(RSM_occ, np.nan)

    flat_stim_obj = tril_flat(RSM_stim_obj)
    flat_roi = tril_flat(RSM_occ)
    nans = np.isnan(flat_stim_obj) | np.isnan(flat_roi)
    flat_stim_obj = flat_stim_obj[~nans]
    flat_roi = flat_roi[~nans]
    if np.sum(nans) > 0:
        print('Number of nans in RSMs: ', nans.sum())

    try:
        r, p = stats.pearsonr(flat_stim_obj, flat_roi)
    except:
        plt.imshow(RSM_occ)
        plt.show()
        quit()
    print(f'{r=:.3f}, {p=:.3f}')
    return r

def do_TR_by_TR_RSA(sn='102', name='bl', combine_regions=False, TRs=4):
    atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    ROI_regions = atlas['ROI_regions']
    idxs_occ = [i for i, region in enumerate(ROI_regions) if
                'EVC' in region or 'LOC' in region]

    df_sn = get_trial_info(sn)
    d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    RSM_stim_obj = get_stim_RDM(df_sn, d_vecs, obj_only=True)
    RSM_stim_obj = within_run_to_nan(RSM_stim_obj)
    np.fill_diagonal(RSM_stim_obj, np.nan)

    _, bolds = get_mtx_sn(sn, name, combine_regions, TRs, easy_override=True,)
    bolds_occ = [bolds[idx] for idx in idxs_occ]
    bolds_comb = np.concatenate(bolds_occ, axis=1)
    bolds_comb = bolds_comb.reshape((bolds_comb.shape[0], -1))
    RSM_roi = np.corrcoef(bolds_comb)
    flat_stim_obj = tril_flat(RSM_stim_obj)
    flat_roi = tril_flat(RSM_roi)
    nans = np.isnan(flat_stim_obj) | np.isnan(flat_roi)
    flat_stim_obj = flat_stim_obj[~nans]
    flat_roi = flat_roi[~nans]
    if nans > 0:
        print('Number of nans in RSMs: ', nans.sum())
    r, p = stats.pearsonr(flat_stim_obj, flat_roi)
    print(f'{r=:.3f}, {p=:.3f}')
    return r
    # print(bolds.shape)
    # bolds_occ.reshape((bolds_occ.shape[1], bolds_occ.shape[0], -1))
    # for roi in range(len(bolds_occ)):
    #     bolds_roi = bolds_occ[roi]
    #     bolds_roi = bolds_roi.reshape((bolds_roi.shape[0], -1))
    #     # bolds_roi = bolds_roi[:, :, 1]
    #     print(bolds_roi.shape)
    #     RSM_roi = np.corrcoef(bolds_roi)
    #     r, p = stats.pearsonr(tril_flat(RSM_stim_obj), tril_flat(RSM_roi))
    #     print(f'{r=:.3f}, {p=:.3f}')

def do_analysis_all():
    age2sn = get_all_sns(ret=False)
    sns = os.listdir(r'C:\PycharmProjects_C\SchemeRep\dir_preproc')
    print(sns)
    quit()
    rs = []
    for sn in sns[::]:
        if sn == '.DS_Store': continue
        if sn == '135': continue# or sn == '103': continue
        print(f'Do RSA conn: {sn}')
        try:
            r = do_connectivity_RSA(sn)
            # r = do_TR_by_TR_RSA(sn)
            # r = test_ERS(sn)
            rs.append(r)
            m = np.mean(rs)
            sd = np.std(rs)
            se = sd / np.sqrt(len(rs))
            t = m / se
            print(f'Group-level: {m=:.3f}, {sd=:.3f}, {se=:.3f}, {t=:.3f}')
        except FileNotFoundError:
            print(f'FileNotFoundError for: {sn}')

def test_ERS(sn, TRs=3, easy_override=False):
    bl_conn, _ = pickle_wrap(None, get_mtx_sn_, kwargs={'sn': sn, 'name': 'bl',
                                'combine_regions': False, 'TRs': TRs},
                          cache_dir='cache/mtx', easy_override=easy_override)

    atlas = get_atlas(combine_regions=False, bilateral=False)
    ROI_regions = atlas['ROI_regions']
    # idxs_occ = [i for i, region in enumerate(ROI_regions) if
    #             'EVC' in region or 'LOC' in region]
    idxs_occ = [i for i, region in enumerate(ROI_regions) if
                ('EVC' in region) or ('LOC' in region) or ('sOcG' in region) or
                ('FuG' in region)]

    bl_occ_all = get_partition_matrix(bl_conn, idxs_occ)
    bl_flat = tril_flat(bl_occ_all)

    enc_conn, _ = pickle_wrap(None, get_mtx_sn_, kwargs={'sn': sn, 'name': 'obj',
                                'combine_regions': False, 'TRs': TRs},
                          cache_dir='cache/mtx', easy_override=easy_override)
    enc_occ_all = get_partition_matrix(enc_conn, idxs_occ)
    enc_flat = tril_flat(enc_occ_all)

    vecs_enc_ = stdize(bl_flat, axis=1)
    vecs_ret_ = stdize(enc_flat, axis=1)
    vecs_enc_ = vecs_enc_[:, None, :]
    vecs_ret_ = vecs_ret_[None, :, :]
    vecs_prod = vecs_enc_ * vecs_ret_
    ers_mat = np.mean(vecs_prod, axis=2)
    ers_same = np.copy(np.diag(ers_mat))

    # ers = corr_last_dim(bl_flat, enc_flat)
    # print(ers.shape)
    # ers_same = np.diag(ers)
    # print(ers_same)

    diag = np.diag_indices_from(ers_mat)
    ers_mat[diag] = np.nan

    ers_else = np.nanmean(ers_mat, axis=1)

    n_nans = np.isnan(ers_else).sum()
    ers_else = ers_else[~np.isnan(ers_same)]
    ers_same = ers_same[~np.isnan(ers_same)]
    # print(ers_same)
    # quit()

    M_same = np.mean(ers_same)
    M_else = np.mean(ers_else)
    M_dif = M_same - M_else
    print(f'{M_same=:.3f}, {M_else=:.3f}, {M_dif=:.3f}, {n_nans=}')
    return M_dif


# RSA based on connectivity among just LOC ROIs (or LOC & EVC)

if __name__ == '__main__':
    # from functools import partial
    # get_mtx_sn_ = partial(get_mtx_sn, sn='102')
    # get_mtx_sn_ = lambda: get_mtx_sn(sn='103')
    # print(get_mtx_sn_.__defaults__)
    # quit()
    # pickle_wrap(None, get_mtx_sn, kwargs={'sn': '102'})
    # quit()
    # print(lambda: get_mtx_sn())
    # quit()
    # get_mtx_sn()
    # test_ERS('102')
    do_analysis_all()
    # do_connectivity_RSA('134')
    # do_TR_by_TR_RSA()
    # need to get TRs
    # load atlas
    # try visual RSA
