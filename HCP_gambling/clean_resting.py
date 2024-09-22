import zipfile
import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import numpy as np
from nilearn import image
import scipy.stats as stats
from nilearn import plotting

import matplotlib.pyplot as plt

from HCP_gambling.preproc_gambling import load_motion
from HCP_gambling.HCP_vendor import get_sn_roi_ar
from atlas_utils import get_atlas
from networks.old.network_funcs import load_FC_for_Lifu
from networks.vendor_partitioning import get_vendor_partitions, do_regression
# from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
# from vendor_partitioning import get_vendor_partitions, do_regression
import time
from numba import jit, prange, njit
import seaborn as sns
from pathlib import Path

def clean_sn_rs(sn, lr, drive='G', reg_global=False, no_compcor=False):
    df_motion = load_motion(sn, lr, drive=drive, rs=True)
    dir_out = r'E:\HCP_RS_clean'

    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''

    fp_out = fr'{dir_out}\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(fp_out) and os.path.getsize(fp_out) > 10_000_000:
        print(f'Exists: {fp_out=}')
        return

    fp_rs = fr'{drive}:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\rfMRI_REST1_{lr}\rfMRI_REST1_{lr}.nii.gz'
    fp_mask = fr'{drive}:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\rfMRI_REST1_{lr}\brainmask_fs.2.nii.gz'
    print(f'Doing: {sn}, {lr}')
    t_st = time.time()
    img = image.load_img(fp_rs)
    print(f'\tLoaded rs: {time.time() - t_st:.2f} s')
    t_st = time.time()

    if not no_compcor:
        df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
        df_confounds = pd.concat([df_compcor, df_motion], axis=1)
    else:
        df_confounds = df_motion

    if reg_global:
        fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'
        img = image.load_img(img)
        mask = image.load_img(fp_mask)
        global_signal = img.get_fdata()[mask.get_fdata() > 0].mean(axis=0)
        df_confounds['global'] = global_signal
    print(f'\tLoaded confounds: {time.time() - t_st:.2f} s')

    t_st = time.time()
    img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                          standardize=True, t_r=.72, mask_img=fp_mask)
    print(f'\tCleaned rs: {time.time() - t_st:.2f} s')
    t_st = time.time()
    img.to_filename(fp_out)
    print(f'\tSaved ({time.time() - t_st:.2f} s): {fp_out=}')


def clean_sn_rs_all(easy_override=False, reg_global=False, no_compcor=False,
                    drive='G'):
    sns = os.listdir(fr'{drive}:\HCP_RS_unzipped')
    sns = list(sns)
    print(f'{len(sns)=}')
    sns = sorted(sns)
    # sns = {'203418'}

    # sns = {'150423', '171734', '119833', '127933', '128127',
    #            '127327', '105216', '105014', '203418', '203923',
    #            '204016', '201515', '201717', '201818', '202113'}

    sns = sorted(list(sns))

    bad_sns = []
    for sn in tqdm(sns, desc='Cleaning RS', position=0, leave=True):
        try:
            clean_sn_rs(sn, 'LR', drive=drive, reg_global=reg_global,
                        no_compcor=no_compcor)
        except Exception as e:
            print(f'Error: {sn=}, {e=}')
            bad_sns.append(sn)

    # for sn in tqdm(sns, desc='Cleaning RS', position=0, leave=True):
        try:
            clean_sn_rs(sn, 'RL', drive=drive, reg_global=reg_global,
                        no_compcor=no_compcor)
        except Exception as e:
            print(f'Error: {sn=}, {e=}')
            bad_sns.append(sn)


if __name__ == '__main__':
    clean_sn_rs_all()