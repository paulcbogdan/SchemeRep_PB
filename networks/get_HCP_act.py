import time

from tqdm import tqdm
import os
import zipfile
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.image import high_variance_confounds

from Utils.atlas_funcs import get_atlas
from utils import stdize
from Utils.pickle_wrap_funcs import pickle_wrap


def get_HCP_task_sns():
    sns = ['100206', '100307', '100408', '100610', '101006', '101107',
           '101309', '101410', '101915', '102008', '102109', '102311',
           '102513', '102614', '102715', '102816', '103111', '103212',
           '103414', '103515', '103818', '104012', '104416', '104820',
           '105014', '105115', '105216', '105620', '105923', '106016',
           '106319', '106521', '106824', '107018', '107220', '107321',
           '107422', '107725', '108020', '108121', '108222', '108323',
           '108525', '108828', '109123', '109325', '109830', '110007',
           '110411', '110613']
    return sns

    in_dir = r'G:\HCP_G'
    # print(os.listdir(in_dir))
    sns = []
    for fn in os.listdir(in_dir):
        if fn[-4:] != '.zip':
            continue
        sn = fn.split('_')[0]
        sns.append(sn)

    # print(f'{len(sns)=}')
    # for i, sn in enumerate(sns):
    #     print(f'{i} | {sn}')

    sns = sns[:50]
    print(sns)
    return sns

def unzip_HCP(task='EMOTION'):
    in_dir = r'G:\HCP_G'
    sns = get_HCP_task_sns()
    for sn in tqdm(sns, desc=f'Unzipping: {task}'):
        fp_zip = f'{in_dir}/{sn}_3T_tfMRI_{task}_preproc.zip'
        dir_out = f'{in_dir}/{sn}_3T_tfMRI_{task}_preproc'
        if os.path.isdir(dir_out):
            continue
        with zipfile.ZipFile(fp_zip, 'r') as zip_ref:
            print(f'{sn} | {dir_out=}')
            zip_ref.extractall(dir_out)

def clean_HCP_task(fp, sn_dir, compcor=True):
    fp_clean = fp.replace('.nii', '_clean.nii')
    if compcor:
        fp_clean = fp_clean.replace('.nii', '_compcor.nii')
    if os.path.isfile(fp_clean):
        img = image.load_img(fp_clean)
        return img.get_fdata()

    img = image.load_img(fp)
    fp_mask = fr'{sn_dir}\brainmask_fs.2.nii.gz'
    assert os.path.isfile(fp_mask), f'No mask found: {sn_dir=}'

    fp_motion = fr'{sn_dir}\Movement_Regressors.txt'
    motion_ar = np.loadtxt(fp_motion)[:, :12]
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz",
                     "dtx", "dty", "dtz", "drx", "dry", "drz"]
    df_confounds = pd.DataFrame(motion_ar, columns=add_reg_names)
    df_confounds = df_confounds.reset_index(drop=True)

    if compcor:
        df_compcor = pd.DataFrame(high_variance_confounds(img,
                                                          percentile=2,
                                                          mask_img=fp_mask))
        df_confounds = pd.concat([df_confounds, df_compcor], axis=1)

    img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                          standardize=False, t_r=.72, mask_img=fp_mask)
    print(f'{fp_clean=}')
    img.to_filename(fp_clean)
    return img.get_fdata()

def load_HCP(task='EMOTION', lr_only=True, combine_regions=True):
    import gzip


    task2conds = {'EMOTION': ('fear', 'neut'),
                  'WM': ('2bk', '0bk')}
    cond0, cond1 = task2conds[task]

    task = task.upper()
    in_dir = r'G:\HCP_G'
    sns = get_HCP_task_sns()
    ars = []
    for sn in tqdm(sns, desc=f'Load: {task}'):
        if task == 'EMOTION':
            dir_sn = fr'{in_dir}/{sn}_3T_tfMRI_{task}_preproc/{sn}'
        elif task == 'WM':
            dir_sn = fr'{in_dir}/{sn}'
        else:
            raise ValueError
        assert os.path.isdir(dir_sn), f'No dir found: {dir_sn=}'
        ar_sn = []
        for lr in ['LR', 'RL']:
            if lr_only and lr == 'RL': continue
            dir_lr = fr'{dir_sn}/MNINonLinear/Results/tfMRI_{task}_{lr}'
            fp = fr'{dir_lr}/tfMRI_{task}_{lr}.nii'
            if os.path.isfile(fp) and os.path.getsize(fp) < 10:
                print(f'Delete too small: {fp=}')
                os.remove(fp)
            if not os.path.isfile(fp):
                fp_gz = f'{fp}.gz'
                assert os.path.isfile(fp_gz)
                with gzip.open(fp_gz, 'rb') as f_in:
                    with open(fp, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f'Unzipped: {fp_gz=}')
            ar = get_roi_ar_from_img(fp, dir_lr, cond0, cond1, combine_regions)
            # ar = pickle_wrap(get_roi_ar_from_img,
            #                  kwargs={'fp': fp, 'dir_lr': dir_lr,
            #                          'cond0': cond0, 'cond1': cond1,
            #                          'combine_regions': combine_regions},
            #                  easy_override=False)
            ar_sn.append(ar)
        ar = np.concatenate(ar_sn, axis=-1)
        ars.append(ar)
    # quit()
    ars = np.array(ars)
    return ars


def get_roi_ar_from_img(fp, dir_lr, cond0, cond1, combine_regions):
    atlas = get_atlas(combine_regions=combine_regions, HCP=True)
    try:
        data = clean_HCP_task(fp, dir_lr)
    except OSError as e:
        print(f'OSError: {fp=} ({e=})')
        time.sleep(10)
        data = clean_HCP_task(fp, dir_lr)
    fp_ev0 = f'{dir_lr}/EVs/{cond0}.txt'
    idxs0 = get_idxs(fp_ev0)
    # print(f'{idxs0=}')
    # print(data.shape)
    # quit()
    data = stdize(data, axis=-1)
    data0 = data[..., idxs0]
    ar0 = img_data2ar(data0, atlas)
    ar0_nan = np.full((3, *ar0.shape), np.nan)
    ar0_nan[0] = ar0
    fp_ev1 = f'{dir_lr}/EVs/{cond1}.txt'
    idxs1 = get_idxs(fp_ev1)
    data1 = data[..., idxs1]
    ar1 = img_data2ar(data1, atlas)
    ar1_nan = np.full((3, *ar1.shape), np.nan)
    ar1_nan[2] = ar1
    ar = np.concatenate([ar0_nan, ar1_nan], axis=-1)
    # print(f'{ar.shape=}')
    # print(ar[:, 0, :])
    # quit()

    return ar

# def img_data2ar(data, atlas):
#     ar = []
#     for j, (ROI, ROI_num, region) in enumerate(zip(atlas['ROIs'],
#                                                    atlas['ROI_nums'],
#                                                    atlas['ROI_regions']
#                                                    )):
#         atlas_roi = atlas['maps'].get_fdata() == ROI_num
#         region_vecs = data[atlas_roi]
#         ts = np.nanmean(region_vecs, axis=0)
#         ar.append(ts)
#     return np.array(ar)

def get_idxs(fp_ev):
    if '2bk.txt' in fp_ev or '0bk.txt' in fp_ev:
        df = pd.read_csv(fp_ev, sep=' ', names=['onset', 'duration', 'amplitude'])
    else:
        df = pd.read_csv(fp_ev, sep='\t', names=['onset', 'duration', 'amplitude'])
    idxs = []
    for onset, duration in zip(df['onset'], df['duration']):
        onset -= 9.36 # https://www.mail-archive.com/hcp-users@humanconnectome.org/msg00621.html
        idx_st = onset // .72
        idx_end = (onset + duration) // .72
        idxs.extend(np.arange(idx_st, idx_end))
    idxs = np.array(idxs).astype(int)
    return idxs

if __name__ == '__main__':
    kwargs = {'task': 'WM',
              'lr_only': True,
              'combine_regions': True}
    ar = pickle_wrap(load_HCP, kwargs=kwargs)
    print(f'{ar.shape=}')

    # get_HCP_task_sns()
    # unzip_HCP()