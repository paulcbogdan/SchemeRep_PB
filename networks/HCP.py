import os

from LSS import get_LSS_img, load_motion
from utils import HCP_ROOT, HCP_CACHE

os.chdir(r'E:\PycharmProjects_E\SchemeRep')
from functools import partial

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from atlas_utils import get_atlas
from fluctuations import get_df_networks, partial_corr_df



def get_sn_HCP(sn, lr, easy_override=False, LSS=False, LSA=False,
               clean_confounds=False):
    if LSS:
        return get_LSS_img(sn, lr, easy_override=easy_override, lsa=LSA)

    fp_clean = fr'{HCP_CACHE}\{sn}_{lr}_{LSA}_{clean_confounds}.nii' # .nii are smaller than .pkl
    if os.path.isfile(fp_clean) and not easy_override:
        img = image.load_img(fp_clean)
        data = img.get_fdata()
        return data
    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_in = fr'{subj_dir}\tfMRI_WM_{lr}\tfMRI_WM_{lr}.nii'
    img = image.load_img(fp_in)
    img = image.index_img(img, slice(4, None))

    # img.get_fdata()
    # confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))
    # motion_confounds = load_motion(sn, lr)
    # confounds = pd.concat([confounds, motion_confounds], axis=1)
    # img = image.clean_img(img, confounds=confounds)

    dir_mask = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in\masks'
    fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
    fp_mask = fr'{dir_mask}\{fn_mask}'


    df_confounds = load_motion(sn, 'lr')  # TODO: compcor not used previously
    df_confounds = df_confounds.iloc[4:].reset_index(drop=True)

    df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
    df_confounds = pd.concat([df_confounds, df_compcor], axis=1)

    img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                          standardize=False, t_r=.72, mask_img=fp_mask)

    img.to_filename(fp_clean)
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

def apply_HCP_mask(data, sn, lr):
    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_mask = fr'{subj_dir}\tfMRI_WM_{lr}\brainmask_fs.2.nii.gz'
    mask = image.load_img(fp_mask)
    mask_data = mask.get_fdata()
    data[mask_data < 0.5, :] = np.nan
    return data


def load_HCP_act(N=50, lr_only=True, do_chop=False, LSS=True, mask=True,
                 LSA=True, clean_confounds=True):
    sns = os.listdir(HCP_ROOT)
    sns = sns[:N]
    # sns = sns[::-1]
    # sns = sns[:5]
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
            data = get_sn_HCP(sn, 'LR', easy_override=False, LSS=LSS,
                              LSA=LSA, clean_confounds=clean_confounds)

            if do_chop:
                data = chop_data(data)
            # if mask and not LSS:
            #     apply_HCP_mask(data, sn, 'LR')
            # print(data)
            # quit()
            # plt.imshow(data[40, :, :, 0])
            # plt.show()
            # quit()
            # data = pickle_wrap(get_sn_HCP, kwargs={'sn': sn,
            #                                        'lr': 'LR'},
            #                    easy_override=False)
            # print(f'{sn=}, {data.shape=}')
        else:
            raise NotImplementedError

        ar = []
        for j, (ROI, ROI_num, region) in enumerate(
                zip(ROIs, ROI_nums, ROI_regions)):
            # a = atlas['maps'].get_fdata()
            atlas_roi = atlas['maps'].get_fdata() == ROI_num
            region_vecs = data[atlas_roi]
            # print(region_vecs)
            ts = np.nanmean(region_vecs, axis=0)
            ar.append(ts)
        ar = np.array(ar)
        # quit()
        # print(f'{sn} | {ar.shape=}')
        sn_roi_act.append(ar)
    sn_roi_act = np.array(sn_roi_act)
    return sn_roi_act, sns

def get_HCP_df(N=15):
    f = partial(load_HCP_act, N=N, do_chop=False, clean_confounds=False,
                LSS=False, mask=False, LSA=False)
    # sn_roi_act, sns, conn_trials = load_act_conn(True, f=f)
    df, networks = get_df_networks(f=f, zscore=True)

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)
    # print(df[networks[:3]])

    # print(df[networks].corr())
    # quit()

    partial_corr_df(df, networks, cov=['FC_all', 'ad_no', 'pd_no',
                                       'av_no', 'pv_no'])
    print('-'*10)
    partial_corr_df(df, networks, cov=['FC_all', 'dd', 'vv', 'dv_ant', 'dv_pos'])



if __name__ == '__main__':
    get_HCP_df()