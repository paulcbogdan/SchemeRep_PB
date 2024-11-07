import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import pandas as pd
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from nilearn import image

from Study1B.preprocess_Study1B import load_motion
import time
from pathlib import Path


def clean_sn_rs(sn, lr, drive='G', reg_global=False, no_compcor=False,
                out_drive='C'):
    df_motion = load_motion(sn, lr, drive=drive, rs=True)
    dir_out = fr'{out_drive}:\HCP_RS_clean'

    dir_out_C = fr'C:\HCP_RS_clean'
    dir_out_E = fr'E:\HCP_RS_clean'

    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''

    fp_out_C = fr'{dir_out_C}\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
    if os.path.exists(fp_out_C) and os.path.getsize(fp_out_C) > 10_000_000:
        print(f'Exists (C): {fp_out_C=}')
        return
    fp_out_E = fr'{dir_out_E}\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
    if os.path.exists(fp_out_E) and os.path.getsize(fp_out_E) > 10_000_000:
        print(f'Exists (E): {fp_out_E=}')
        return

    fp_out = fr'{dir_out}\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(fp_out) and os.path.getsize(fp_out) > 10_000_000:
        print(f'Exists: {fp_out=}')
        return

    fp_rs = fr'{drive}:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\rfMRI_REST1_{lr}\rfMRI_REST1_{lr}.nii.gz'
    fp_mask = fr'{drive}:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\rfMRI_REST1_{lr}\brainmask_fs.2.nii.gz'
    print(f'Doing: {sn}, {lr}, {fp_out=}')
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
    img = image.clean_img(img, confounds=df_confounds, high_pass=1 / 128,
                          standardize=True, t_r=.72, mask_img=fp_mask)
    print(f'\tCleaned rs: {time.time() - t_st:.2f} s')
    t_st = time.time()
    img.to_filename(fp_out)
    print(f'\tSaved ({time.time() - t_st:.2f} s): {fp_out=}')


def clean_sn_rs_all(reg_global=False, no_compcor=False,
                    drive='G', out_drive='C'):
    sns = os.listdir(fr'{drive}:\HCP_RS_unzipped')

    sns = sorted(list(sns))

    bad_sns = []
    for sn in tqdm(sns, desc='Cleaning RS', position=0, leave=True):

        try:
            clean_sn_rs(sn, 'LR', drive=drive, reg_global=reg_global,
                        no_compcor=no_compcor, out_drive=out_drive)
        except Exception as e:
            print(f'Error: {sn=}, {e=}')
            bad_sns.append(sn)

        try:
            clean_sn_rs(sn, 'RL', drive=drive, reg_global=reg_global,
                        no_compcor=no_compcor, out_drive=out_drive)
        except Exception as e:
            print(f'Error: {sn=}, {e=}')
            bad_sns.append(sn)


if __name__ == '__main__':
    clean_sn_rs_all()
