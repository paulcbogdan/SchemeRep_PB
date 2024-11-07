import os
import time

from old_Apr6.LSS import get_LSS_img, load_motion
from utils import HCP_ROOT, HCP_CACHE, HCP_RS_ROOT, pickle_wrap, stdize

os.chdir(r'H:\PycharmProjects_H\SchemeRep')
from functools import partial

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from atlas_utils import get_atlas
from Study2A.rs_funcs import get_df_networks, partial_corr_df


def get_sn_HCP(sn, lr, easy_override=False, LSS=False, LSA=False,
               clean_confounds=False, RS=True, compcor=True):
    # print(f'get_sn_HCP: {sn=}')

    if LSS:
        return get_LSS_img(sn, lr, easy_override=easy_override, lsa=LSA)

    RS_str = f'_RS' if RS else ''
    cc_str = '_noCC' if not compcor else ''
    fp_clean = (fr'{HCP_CACHE}\{sn}_{lr}_{LSA}_{clean_confounds}'
                fr'{RS_str}{cc_str}.nii') # .nii are smaller than .pkl
    if os.path.isfile(fp_clean) and not easy_override:
        print(f'exists: {fp_clean=}')
        # return None
        img = image.load_img(fp_clean)
        data = img.get_fdata()
        return data
    if RS:
        subj_dir = fr'{HCP_RS_ROOT}\{sn}\MNINonLinear\Results\rfMRI_REST1_LR'
        fp_in = fr'{subj_dir}\rfMRI_REST1_LR.nii.gz'
    else:
        subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
        fp_in = fr'{subj_dir}\tfMRI_WM_{lr}\tfMRI_WM_{lr}.nii'
    img = image.load_img(fp_in)
    # print(img.shape)
    # quit()
    # img = image.index_img(img, slice(4, None))


    # img.get_fdata()
    # confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))
    # motion_confounds = load_motion(sn, lr)
    # confounds = pd.concat([confounds, motion_confounds], axis=1)
    # img = image.clean_img(img, confounds=confounds)

    if clean_confounds:
        fp_mask = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results\tfMRI_WM_LR\brainmask_fs.2.nii.gz'
        mask_exist = os.path.isfile(fp_mask)
        print(f'{mask_exist=} | {fp_mask=}')
        df_confounds = load_motion(sn, 'lr', rs=RS)  # TODO: compcor not used previously
        df_confounds = df_confounds.reset_index(drop=True)
        # print(f'num nans: {pd.isna(df_confounds).sum()=}')
        if compcor:
            df_compcor = pd.DataFrame(high_variance_confounds(img,
                                                              percentile=2,
                                                              mask_img=fp_mask))
            df_confounds = pd.concat([df_confounds, df_compcor], axis=1)
        # print(f'num nans: {pd.isna(df_compcor).sum()=}')

        img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                              standardize=False, t_r=.72, mask_img=fp_mask)
        # quit()

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


def load_HCP_act(RS=True, N=50, lr_only=True, LSS=False,
                 LSA=False, clean_confounds=True, YA_only=None,
                 compcor=True, combine_regions=False,
                 GSR=True):
    sns = os.listdir(HCP_ROOT)
    if RS:
        sns_ = []
        for sn in sns:
            dir_candidate = fr'E:\HCP_RS\{sn}'
            if os.path.isdir(dir_candidate):
                sns_.append(sn)
        sns = sns_

    sns = sns[:N]
    sns = sns[::-1]
    print(f'{sns=}')
    print(len(sns))
    # sns = sns[:5]
    sn_roi_act = []
    # combine_regions = False
    # print(f'toast: {combine_regions=}') # combine_regions=combine_regions,
    atlas = get_atlas(combine_regions=combine_regions, HCP=True)

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    for sn in tqdm(sns, desc='Loading HCP fMRI'):
        if lr_only:
            try:
                data = get_sn_HCP(sn, 'LR', easy_override=False, LSS=LSS,
                                  LSA=LSA, clean_confounds=clean_confounds,
                                  RS=RS, compcor=compcor)
            except ValueError as e:
                print(f'AKKH??: {e}')
                time.sleep(5)
                continue
            except Exception as e:
                print(f'ERROR: {sn=}, {e=}')
                # continue
                time.sleep(5)
                try:
                    data = get_sn_HCP(sn, 'LR', easy_override=False, LSS=LSS,
                                      LSA=LSA, clean_confounds=clean_confounds,
                                      compcor=compcor)
                except:
                    print(f'SKYUIIIIIP: {sn=}, {e=}')
                    continue
        else:
            raise NotImplementedError
        # continue
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
        print(f'{sn} | {ar.shape=}')
        sn_roi_act.append(ar)
    sn_roi_act = np.array(sn_roi_act)

    if GSR:
        sn_roi_act = stdize(sn_roi_act, axis=2)
        sn_roi_act = stdize(sn_roi_act, axis=1)
    # conn_trials = sn_roi_act[..., None, :] * \
    #               sn_roi_act[..., None, :, :]

    return sn_roi_act, sns

def get_HCP_df(N=25):
    f = partial(load_HCP_act, N=N,
                RS=True, clean_confounds=True,
                compcor=True, GSR=False)

    # sn_roi_act, sns, conn_trials = load_act_conn(True, f=f)
    df, networks = pickle_wrap(get_df_networks,
                               kwargs={'f': f, 'zscore': False,
                                       'anat_ver': 2},
                               easy_override=True)

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)
    # print(df[networks[:3]])

    # print(df[networks].corr())
    # quit()

    # networks = ['dv_ant', 'dv_pos']

    #

    # partial_corr_df(df, networks,
    #                 cov=['FC_all', 'ad_else', 'pd_else', 'av_else', 'pv_else'])

    print('------')
    # 'dv_ant_else', 'dv_pos_else'
    # partial_corr_df(df, networks,
    #                 cov=['FC_all', 'dd_no', 'vv_no', 'dv_ant_no', 'dv_pos_no'])
    networks = ['pd_else', 'ad_else', 'av_else', 'pv_else']
    networks = ['dd', 'vv', 'dv_ant', 'dv_pos', 'dpva', 'vpda']
    # networks = ['dp', 'da', 'vp', 'va']
    # print(df[networks].std())
    pre_n = len(df)
    # for col in networks:
    #     df[f'{col}_z'] = stats.zscore(df[col], nan_policy='omit')
    #     df = df[df[f'{col}_z'].abs() < 5]
    #     n_dif = pre_n - len(df)
    #     pre_n = len(df)
    #     print(f'{len(df)=}, dropped: {n_dif}')
    # print(df[networks].mean())
    #
    # plt.scatter(df['dd'], df['vpda'])
    # plt.show()

    # 'FC_all',
    # 'pd_else', 'pv_else', 'ad_else', 'av_else', 'FC_all'

    partial_corr_df(df, networks,
                    cov=['FC_all',
                         'pd_no', 'ad_no', 'av_no', 'pv_no'])

    # networks = ['dd', 'vv', 'dv_ant', 'dv_pos']
    # partial_corr_df(df, networks,
                    # cov=['FC_all', 'dv_ant', 'dv_pos'])
                         # 'dd_else', 'vv_else', ])

    # partial_corr_df(df, networks,
    #                 cov=['FC_all', 'pd_no', 'ad_no', 'av_no', 'pv_no'])
    # 'ad_no', 'pd_no', 'av_no', 'pv_no'
    # print('-'*10)
    # partial_corr_df(df, networks, cov=['FC_all', 'dd', 'vv', 'dv_ant',
    #                                    'dv_pos'])



if __name__ == '__main__':
    get_HCP_df()