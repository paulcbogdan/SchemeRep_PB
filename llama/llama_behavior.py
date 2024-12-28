from Utils.atlas_funcs import get_atlas
from connRSA.conn_regress import do_regr_RSA_sn
from llama.model_settings import get_explore_llama, get_base_kw
from llama.numba_regr_test import pairwise_interaction_t_values_proper
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy import stats

from organize_bhv import get_trial_info


# disable SettingWithCopyWarning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def get_base_kw_predicting(sn, region, local, big_voxelwise):
    kw = {'fp': 'obj7_fMRI', 'sn': sn}
    if local:
        kw['ROI_focus'] = f'{region}_M_corr'
    elif big_voxelwise:
        kw['ROI_focus'] = f'{region}_BOLD_cmb'
    else:
        kw['ROI_focus'] = f'{region}_BOLD'

    kw['regress_row'] = True
    kw['trial_similarity'] = 'corr'
    kw['second_order'] = 'spear'
    kw['RDM_method'] = 'within_nan'
    kw['stdize_by_run'] = False
    return kw

def sn_attn_enc(sn, region='subcort', local=True,
                big_voxelwise=False, do_acc=True
                ):
    # region = 'PFC'
    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model,
                                   attn=True, normalize=True,
                                   do_prod=False, do_M=False,
                                   include_scn=True,
                                   st=8, end=20,
                                   last_only=False
                                   )

    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   include_scn=True,
                                   do_prod=False)
    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)

    # kw['semantic'] = 'inc'
    kw['semantic'] = model_attn
    kw['ROIs_ctrl'] = []
    # ['inc']# model_item] # ['cortical_M_corr']#model_item]

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                        easy_override=False, dir_branches=100)

    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]
    df_sn['IRAFs'] = IRAFs

    # df_sn = df_sn[df_sn['inc'] < 3]

    if do_acc:
        df_sn['inc_m'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
        df_sn['enc_acc'] = -(df_sn['inc_m'] - df_sn['per_inc']).abs()
    else:
        df_sn['enc_acc'] = df_sn['per_inc'].astype(float)

    # print(df_sn['enc_acc'].value_counts())

    df_sn = df_sn.dropna(subset=['IRAFs', 'enc_acc'])
    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['enc_acc'])
    # print(f'{r=:.3f}')
    return r

def sn_attn_enc_Tha(sn, region='subcort', local=True,
                     big_voxelwise=False, do_acc=True
                    ):
    region_ = 'PFC'
    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model,
                                   attn=True, normalize=True,
                                   do_prod=False, do_M=False,
                                   include_scn=True,
                                   st=8, end=20,
                                   last_only=False
                                   )

    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   include_scn=True,
                                   do_prod=False)
    kw = get_base_kw_predicting(sn, region_, local, big_voxelwise)

    kw['semantic'] = model_attn

    kw['ROIs_ctrl'] = []#model_item]
    #, 'cortical_M_corr']
    # ['inc']# model_item] # ['cortical_M_corr']#model_item]

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                        easy_override=False, dir_branches=100)

    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]
    df_sn['IRAFs'] = IRAFs

    if do_acc:
        df_sn['inc_m'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
        df_sn['enc_acc'] = -(df_sn['inc_m'] - df_sn['per_inc']).abs()
    else:
        df_sn['enc_acc'] = df_sn['per_inc'].astype(float)


    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)
    kw['semantic'] = model_attn
    kw['ROIs_ctrl'] = []#model_item]
    IRAFs_str = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                            easy_override=False, dir_branches=100)
    if len(kw['ROIs_ctrl']):
        IRAFs_str = IRAFs_str[:, 0]

    y = np.array(df_sn['enc_acc'].to_list())

    df_sn = df_sn.dropna(subset=['IRAFs', 'enc_acc'],)
    assert len(df_sn) > 10, f'{len(df_sn)=}'

    X = np.array([IRAFs, IRAFs_str]).T
    nan_y = np.isnan(y)
    nan_X = np.any(np.isnan(X), axis=1)
    nans = nan_y | nan_X
    y = y[~nans]
    X = X[~nans]
    X = stats.zscore(X, axis=0)
    t = pairwise_interaction_t_values_proper(y, X)
    return t[0, 1]


def sn_item_dm(sn, region='Str', local=True, big_voxelwise=False):
    local = True
    big_voxelwise = False
    activation_model = 'meta-llama/Llama-3.2-3b'
    model = get_explore_llama(activation_model,
                              attn=True, normalize=True,
                              do_M=False, st=4, end=16,)

    model = get_explore_llama(activation_model,
                              attn=False, normalize=True,
                              do_M='obj_solo', st=6, end=22)

    # model = get_explore_llama(activation_model,
    #                           attn=True, normalize=True,
    #                           include_scn=True,
    #                           do_prod=False)

    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)
    kw['fp'] = 'obj7_fMRI'
    kw['semantic'] = model
    kw['ROIs_ctrl'] = []#'cortical_M_corr']#model_solo]

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                        verbose=-1, easy_override=False,
                        dir_branches=100)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]

    df_sn = get_trial_info(sn)
    df_sn['is_YA'] = df_sn['sn'].apply(lambda x: str(x)[0] == '1')

    df_sn.sort_values(by=f'obj_trial', inplace=True)
    df_sn['IRAFs'] = IRAFs

    # df_sn = df_sn[df_sn['inc'] == 3]

    df_sn['mem_key'] = df_sn['vis_hit'] & df_sn['con_hit']
    df_sn['mem_key'] = df_sn['vis_hit'].astype(float)

    df_sn = df_sn.dropna(subset=['IRAFs', 'mem_key'])
    assert len(df_sn) > 10, f'{len(df_sn)=}'

    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['mem_key'])
    # r = np.nanmean(df_sn['IRAFs'])
    is_YA = df_sn['is_YA'].iloc[0]

    # if is_YA:
    #     r = 0
    #     r = -r
    return r


def test_sn_llama_mem():
    fps = ['obj7_fMRI']
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]

    REGIONS = get_atlas(combine_regions=True,
                        combine_bilateral=True)['ROIs']
    REGIONS = ['cortical'] + REGIONS
    for REGION in REGIONS:
        # print(REGION)
        efs = []
        for sn in sns:
            # ef = sn_attn_enc_Tha(sn, region=REGION)
            ef = sn_item_dm(sn, region=REGION)
            if np.isnan(ef): continue
            if ef is None: continue
            efs.append(ef)
        t, p = stats.ttest_1samp(efs, 0, axis=0)
        N = np.sum(~np.isnan(efs), axis=0)
        print(f'{REGION} | t[{N - 1}] = {t:.2f}, {p=:.3f}')



if __name__ == '__main__':
    test_sn_llama_mem()