import os

import numpy as np
from nilearn import image
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from pickle_wrap import pickle_wrap

from atlas_utils import get_atlas
from fMRI_proc import get_all_stim_RDMs, within_run_to_nan
from modularity_testing import get_partition_matrix
from organize_bhv import get_trial_info, get_all_sns
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm

from permutation_test import Timer
from pathlib import Path
import gzip
import shutil

from stim import get_DNN_vecs, get_stim_RDM
from utils import tril_flat, stdize
import scipy.stats as stats


def decompress(in_fp, out_fp):
    print(f'Decompressing: {in_fp}')
    with gzip.open(in_fp, 'r') as f_in, open(out_fp, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def get_FC_trial(img_data, row, name, TRs, atlas):
    # img_trial = img.slicer[..., trial_slice]
    # time_series = masker.fit_transform(img_trial)
    # mtx = corr_measure.fit_transform([time_series])[0]
    onset_TR = row[f'{name}_onset_TR']
    trial_slice = slice(onset_TR, onset_TR + TRs)
    # t = Timer()
    img_trial_data = img_data[..., trial_slice]
    # print(f'Time slice: {t.lap():.3f}')
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    time_series = []
    region_vecs_all = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        region_vecs = img_trial_data[atlas_roi]
        time_series.append(region_vecs.mean(axis=0))
        region_vecs_all.append(region_vecs)
    time_series = np.array(time_series)
    time_series = stdize(time_series, axis=0)

    # Could regress out across ROI (e.g., axis=0 here), although that
    #   would be most meaningful if regions are isolated (e.g., only occipital)

    mtx = np.corrcoef(time_series)
    return mtx, region_vecs_all

def get_mtx_sn_(sn='102', name='bl', combine_regions=False, TRs=3):
    atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    # masker = NiftiLabelsMasker(labels_img=atlas['maps'], standardize=True)
    # corr_measure = ConnectivityMeasure(kind="correlation")
    df_sn = get_trial_info(sn)

    dir_in = fr'dir_preproc/{sn}/{name.lower()}'
    mtx_all = []
    bolds = []
    for run in range(1, 4):
        df_run = df_sn[df_sn[f'{name}_run'] == run]

        t = Timer()
        fp_in = fr'{dir_in}/BOLD_run{run}.nii' # .nii is 35% time of .nii.gz
        if not Path(fp_in).exists():
            fp_in_gz = fp_in + '.gz'
            decompress(fp_in_gz, fp_in)

        img = image.load_img(fp_in)
        # print(f'Load: {t.lap():.3f}')
        img_data = img.get_fdata()
        # print(f'Get data: {t.lap():.3f}')
        for idx, row in tqdm(df_run.iterrows()):
            mtx, bold = get_FC_trial(img_data, row, name, TRs, atlas)
            mtx_all.append(mtx)
            bolds.append(bold)

    mtx_all = np.array(mtx_all)
    bolds = list(np.array(x) for x in zip(*bolds)) # (ROI, trial, voxel, TR)
    return mtx_all, bolds

def get_mtx_sn(sn='102', name='bl', combine_regions=False, TRs=4,
               easy_override=False):
    print(f'test: {sn=}')
    mtx_all, bolds = pickle_wrap(None, get_mtx_sn_, kwargs={'sn': sn, 'name': name,
                                'combine_regions': combine_regions, 'TRs': TRs},
                          cache_dir='cache/mtx', easy_override=easy_override)
    return mtx_all, bolds

def do_connectivity_RSA(sn='102', name='bl', combine_regions=False, TRs=4):
    atlas = get_atlas(combine_regions=combine_regions, bilateral=False)
    ROI_regions = atlas['ROI_regions']
    idxs_occ = [i for i, region in enumerate(ROI_regions) if
                'EVC' in region or 'LOC' in region]

    df_sn = get_trial_info(sn)
    d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    RSM_stim_obj = get_stim_RDM(df_sn, d_vecs, obj_only=True)
    np.fill_diagonal(RSM_stim_obj, np.nan)

    mtx_all, _ = get_mtx_sn(sn, name, combine_regions, TRs, easy_override=True)
    mtx_occ_all = get_partition_matrix(mtx_all, idxs_occ)
    flat_occ_all = tril_flat(mtx_occ_all)
    RSM_occ = np.corrcoef(flat_occ_all)
    np.fill_diagonal(RSM_occ, np.nan)

    r, p = stats.pearsonr(tril_flat(RSM_stim_obj), tril_flat(RSM_occ))
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
    r, p = stats.pearsonr(tril_flat(RSM_stim_obj), tril_flat(RSM_roi))
    print(f'{r=:.3f}, {p=:.3f}')
    return r


    # print(bolds.shape)
    # bolds_occ.reshape((bolds_occ.shape[1], bolds_occ.shape[0], -1))
    for roi in range(len(bolds_occ)):
        bolds_roi = bolds_occ[roi]
        bolds_roi = bolds_roi.reshape((bolds_roi.shape[0], -1))
        # bolds_roi = bolds_roi[:, :, 1]
        print(bolds_roi.shape)
        RSM_roi = np.corrcoef(bolds_roi)
        r, p = stats.pearsonr(tril_flat(RSM_stim_obj), tril_flat(RSM_roi))
        print(f'{r=:.3f}, {p=:.3f}')

def do_connectivity_RSA_all():
    age2sn = get_all_sns(ret=False)
    sns = os.listdir(r'C:\PycharmProjects_C\SchemeRep\dir_preproc')
    rs = []
    for sn in sns[::-1]:
        if sn == '135' or sn == '103': continue
        print(f'Do RSA conn: {sn}')
        try:
            r = do_connectivity_RSA(sn)
            # r = do_TR_by_TR_RSA(sn)
            rs.append(r)
            m = np.mean(rs)
            sd = np.std(rs)
            se = sd / np.sqrt(len(rs))
            t = m / se
            print(f'2nd order: {m=:.3f}, {sd=:.3f}, {se=:.3f}, {t=:.3f}')
        except FileNotFoundError:
            print(f'FileNotFoundError for: {sn}')


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
    do_connectivity_RSA_all()
    # do_connectivity_RSA('135')
    # do_TR_by_TR_RSA()
    # need to get TRs
    # load atlas
    # try visual RSA
    pass