import os
from functools import partial

from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nilearn import image
from nilearn.image import high_variance_confounds
from scipy import linalg, stats as stats
from tqdm import tqdm

from analyze_rs import load_act_conn, get_rs_vendor_df
from atlas_utils import get_atlas
from fluctuations import get_df_networks, partial_corr_df
from old.modularity import get_modules, get_partition_matrix
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from utils import pickle_wrap, stdize, f2str
from vendor_lmers import get_module_cross_trialwise_z, get_module_trialwise_z
from vendor_partitioning import get_vendor_partitions

HCP_ROOT = r'F:\HCP_Preprocessing\HCP\HCP_WM_data'
HCP_CACHE = r'E:\PycharmProjects_E\SchemeRep\cache\HCP_nii'
from time import time

# def get_LSS_imgs():

def load_motion(sn, lr):
    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_in = fr'{subj_dir}\tfMRI_WM_{lr}\tfMRI_WM_{lr}.nii'
    fp_motion = fr'{subj_dir}\tfMRI_WM_{lr}\Movement_Regressors.txt'
    motion = np.loadtxt(fp_motion)[:, :6]
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    df = pd.DataFrame(motion, columns=add_reg_names)
    return df




def get_sn_HCP(sn, lr, easy_override=False):
    fp_clean = fr'{HCP_CACHE}\{sn}_{lr}_clean.nii' # .nii are smaller than .pkl
    if os.path.isfile(fp_clean) and not easy_override:
        img = image.load_img(fp_clean)
        data = img.get_fdata()
        return data

    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_in = fr'{subj_dir}\tfMRI_WM_{lr}\tfMRI_WM_{lr}.nii'

    # t_st = time()
    img = image.load_img(fp_in)
    # print(f'Load in {time() - t_st:.2f} s')
    # t_st = time()
    img.get_fdata()
    # print(f'Test get_fdata: {time() - t_st:.2f} s')
    # t_st = time()
    confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))

    motion_confounds = load_motion(sn, lr)
    confounds = pd.concat([confounds, motion_confounds], axis=1)
    # print(f'Confound in {time() - t_st:.2f} s')
    img = image.clean_img(img, confounds=confounds)

    # print(f'Clean in {time() - t_st:.2f} s')

    # fp_test = r'E:\PycharmProjects_E\SchemeRep\test.nii'
    img.to_filename(fp_clean)


    # print(f'{img.affine=}')
    # print(f'{img.shape=}')
    data = img.get_fdata()


    return data

def chop_data(data):
    LR_onsets = {'0bk_body': 36.119,
                 '0bk_faces': 221.844,
                 '0bk_places': 250.06,
                 '0bk_tools': 107.45,
                 '2bk_body': 150.526,
                 '2bk_faces': 79.222,
                 '2bk_places': 178.609,
                 '2bk_tools': 7.997,}
    TR = 0.72
    onsets = [LR_onsets['2bk_body'], LR_onsets['2bk_faces'],
              LR_onsets['2bk_places'], LR_onsets['2bk_tools']]
    onsets = [onset + 5 for onset in onsets]
    closes = [onset + 21 for onset in onsets]
    onsets = [int(onset / TR) for onset in onsets]
    closes = [int(close / TR) for close in closes]

    data_new = []

    for onset, close in zip(onsets, closes):
        data_new.append(data[..., onset:close])
    data_new = np.concatenate(data_new, axis=-1)
    return data_new



def load_HCP_act(N=50, lr_only=True, do_chop=True):
    sns = os.listdir(HCP_ROOT)
    sns = sns[:N]
    sns = sns[::-1]
    sn_roi_act = []
    atlas = get_atlas(HCP=True)
    # print(f'{atlas["maps"].shape=}')
    # print(atlas['maps'].affine)
    # quit()
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    for sn in tqdm(sns, desc='Loading fMRI'):
        if lr_only:
            data = get_sn_HCP(sn, 'LR', easy_override=False)
            if do_chop:
                data = chop_data(data)
            # data = pickle_wrap(get_sn_HCP, kwargs={'sn': sn,
            #                                        'lr': 'LR'},
            #                    easy_override=False)
            print(f'{sn=}, {data.shape=}')
        else:
            raise NotImplementedError
        ar = []
        for j, (ROI, ROI_num, region) in enumerate(
                zip(ROIs, ROI_nums, ROI_regions)):
            # a = atlas['maps'].get_fdata()
            atlas_roi = atlas['maps'].get_fdata() == ROI_num
            region_vecs = data[atlas_roi]
            ts = np.nanmean(region_vecs, axis=0)
            ar.append(ts)
        ar = np.array(ar)
        # print(f'{sn} | {ar.shape=}')
        sn_roi_act.append(ar)
    sn_roi_act = np.array(sn_roi_act)
    return sn_roi_act, sns

def get_HCP_df(N=10):
    f = partial(load_HCP_act, N=N, do_chop=True)
    # sn_roi_act, sns, conn_trials = load_act_conn(True, f=f)
    df, networks = get_df_networks(f=f)

    print(df['FC_all'])

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    partial_corr_df(df, networks, cov=['FC_all', ])
    print('-'*10)
    partial_corr_df(df, networks, cov=['FC_all', 'dd', 'vv', 'dv_ant', 'dv_pos'])



if __name__ == '__main__':
    get_HCP_df()
    quit()
    # get_HCP_df()
    load_HCP_act()
