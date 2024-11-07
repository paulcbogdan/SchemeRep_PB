import os
from time import sleep

import numpy as np
import pandas as pd
from nilearn import masking, image
from nilearn.image import high_variance_confounds

from Study3.EEG_simultaneous_funcs import ROOT_EEG_FMRI
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
import scipy.stats as stats

def get_fMRI_ar(sn, sess, combine_regions=False):
    root_sn = fr'{ROOT_EEG_FMRI}\sub-{sn}\ses-{sess.split("_")[0]}'
    dir_func = fr'{root_sn}\func\sub-{sn}_ses-{sess}_bold\func_preproc'
    fp_fMRI = fr'{dir_func}\func_pp_filter_sm0.mni152.3mm.nii.gz'
    mask_img = masking.compute_brain_mask(fp_fMRI)

    if os.path.isfile(fp_fMRI):
        img = image.load_img(fp_fMRI)
    else:
        print(fr'Seemingly no file: {fp_fMRI=}')
        sleep(1)
        if os.path.isfile(fp_fMRI):
            print('\tFile found after 1 s pause')
            img = image.load_img(fp_fMRI)
        else:
            print('\tConfirmed no file')
            return None, None

    df_compcor = pd.DataFrame(high_variance_confounds(img, mask_img=mask_img,))
    dir_nuisance = fr'{root_sn}\func\sub-{sn}_ses-{sess}_bold\func_nuisance'
    fp_motion = fr'{dir_nuisance}\mc_1-6.txt'
    df_motion = pd.read_csv(fp_motion, sep=' ', header=None,
                            names=[f'motion_{i}' for i in range(6)])
    df_confounds = pd.concat([df_compcor, df_motion], axis=1)
    img = image.clean_img(img, confounds=df_confounds, high_pass=1 / 128,
                          standardize=False, t_r=2.1)

    data_fMRI = img.get_fdata()

    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    key2idxs = {'ATL': [], 'MFG': [], 'IPL': [], 'LOC': []}
    for i, region in enumerate(atlas['ROI_regions']):
        for key, l in key2idxs.items():
            if key in region:
                l.append(i)

    ar_fMRI = img_data2ar(data_fMRI, atlas)
    ar_fMRI = stats.zscore(ar_fMRI, axis=-1)
    return ar_fMRI, key2idxs



def get_fMRI_score_sn(sn, sess='01', combine_regions=False,):

    try:
        ar_fMRI, key2idxs = pickle_wrap(get_fMRI_ar,
                                        kwargs={'sn': sn, 'sess': sess,},
                                        easy_override=False, verbose=0,)
    except ValueError as e:
        print(f'Missing file ({sn}, {sess}): {e=}')
        return None

    if ar_fMRI is None:
        return None

    atlas = get_atlas(natview=True, combine_regions=combine_regions)

    ROI2idx = {ROI: [] for ROI in atlas['ROI_regions']}

    for i, region in enumerate(atlas['ROI_regions']):
        ROI2idx[region].append(i)

    key2idxs['MFG'] = ROI2idx['MFG'] + ROI2idx['IFG']
    key2idxs['IPL'] = ROI2idx['IPL']
    key2idxs['LOC'] = ROI2idx['LOC'] + ROI2idx['sOcG'] + ROI2idx['EVC']
    key2idxs['ATL'] = ROI2idx['ATL']

    conn_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]

    ATL_MFG = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['MFG'])].mean(axis=(0, 1))
    ATL_LOC = conn_fMRI[np.ix_(key2idxs['ATL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))
    MFG_IPL = conn_fMRI[np.ix_(key2idxs['MFG'],
                               key2idxs['IPL'])].mean(axis=(0, 1))
    IPL_LOC = conn_fMRI[np.ix_(key2idxs['IPL'],
                               key2idxs['LOC'])].mean(axis=(0, 1))

    ATL_MFG = stats.zscore(ATL_MFG, axis=-1)
    ATL_LOC = stats.zscore(ATL_LOC, axis=-1)
    MFG_IPL = stats.zscore(MFG_IPL, axis=-1)
    IPL_LOC = stats.zscore(IPL_LOC, axis=-1)

    fluc = np.abs(MFG_IPL + ATL_LOC - ATL_MFG - IPL_LOC)

    alt1_signed = MFG_IPL + ATL_LOC - ATL_MFG - IPL_LOC
    alt2_abs_sum = np.abs(MFG_IPL + ATL_LOC + ATL_MFG + IPL_LOC)
    alt3_sum = MFG_IPL + ATL_LOC + ATL_MFG + IPL_LOC
    return fluc, alt1_signed, alt2_abs_sum, alt3_sum

def img_data2ar(data, atlas):
    ar = []
    for j, (ROI, ROI_num, region) in enumerate(zip(atlas['ROIs'],
                                                   atlas['ROI_nums'],
                                                   atlas['ROI_regions']
                                                   )):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        region_vecs = data[atlas_roi]
        ts = np.nanmean(region_vecs, axis=0)
        ar.append(ts)
    return np.array(ar)
