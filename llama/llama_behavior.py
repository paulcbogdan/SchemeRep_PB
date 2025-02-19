from Utils.atlas_funcs import get_atlas
from marinate import marinate
from connRSA.conn_regress import do_regr_RSA_sn
from llama.model_settings import get_explore_llama, get_base_kw
from llama.numba_regr_test import pairwise_interaction_t_values_proper
from marinate.pkld import pkld
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

def sn_attn_enc(sn, region='PFC', local=True,
                big_voxelwise=False, do_acc=False,
                control_item=False, attn=True,
                normalize=True
                ):
    # region = 'PFC'
    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model,
                                   attn=attn, normalize=normalize,
                                   do_prod=False, do_M=False,
                                   include_scn=True,
                                   st=8, end=20,
                                   last_only=False
                                   )
    # print(model_attn)
    # quit()

    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   include_scn=True,
                                   do_prod=False,
                                   do_M='obj_solo',
                                   )
    # print(model_item)
    # quit()
    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)

    # kw['semantic'] = 'inc'
    kw['semantic'] = model_attn
    kw['ROIs_ctrl'] = [model_item] if control_item else []
    # ['inc']# model_item] # ['cortical_M_corr']#model_item]


    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                        easy_override=False, dir_branches=100)

    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]
    df_sn['IRAFs'] = IRAFs

    l = df_sn['IRAFs'].to_list()
    print(f'{sn}')
    print(f'{region}')
    print(f'{l=}')
    quit()

    # df_sn = df_sn[df_sn['inc'] < 3]

    if do_acc:
        df_sn['inc_m'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
        df_sn['enc_acc'] = -(df_sn['inc_m'] - df_sn['per_inc']).abs()
    else:
        df_sn['enc_acc'] = df_sn['per_inc'].astype(float)

    df_sn = df_sn.dropna(subset=['IRAFs', 'enc_acc'])
    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['enc_acc'])
    return r

def sn_attn_enc_Tha(sn, region, FC_target='tha_str', local=True,
                    big_voxelwise=False, do_acc=True,
                    control_item=False,
                    st=8, end=20, attn=True,
                    normalize=True,
                    local_target=True
                    ):


    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model,
                                   attn=attn, normalize=normalize,
                                   do_prod=False, do_M=False,
                                   include_scn=True,
                                   st=st, end=end,
                                   last_only=False
                                   )

    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   include_scn=True,
                                   do_prod=False,
                                   do_M='obj_solo',
                                   st=st, end=end
                                   # st=8, end=20,
                                   )


    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)

    kw['semantic'] = model_attn

    kw['ROIs_ctrl'] = [model_item] if control_item else []

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                        easy_override=False, dir_branches=100)
    # print(IRAFs)
    # quit()

    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]
    df_sn['IRAFs'] = IRAFs

    if do_acc:
        df_sn['inc_m'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
        df_sn['enc_acc'] = -(df_sn['inc_m'] - df_sn['per_inc']).abs()
        # enc_nans = df_sn['enc_acc'].isna()
        # df_sn['enc_acc'] = (df_sn['enc_acc'] > -1).astype(float)
        # df_sn.loc[enc_nans, 'enc_acc'] = np.nan
    else:
        df_sn['enc_acc'] = df_sn['per_inc'].astype(float)

    kw = get_base_kw_predicting(sn, FC_target, local_target,
                                big_voxelwise)
    kw['semantic'] = model_attn
    kw['ROIs_ctrl'] = [model_item] if control_item else []

    IRAFs_str = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                            easy_override=False, dir_branches=100)

    if len(kw['ROIs_ctrl']):
        IRAFs_str = IRAFs_str[:, 0]

    # r, p = stats.spearmanr(IRAFs, IRAFs_str, nan_policy='omit')
    # return r

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


@pkld(overwrite=False)
def sn_item_dm(sn, region='Str', local=True, big_voxelwise=False,
               fp_neuro='obj7_fMRI'):
    activation_model = 'meta-llama/Llama-3.2-3b'

    model = get_explore_llama(activation_model,
                              attn=False, normalize=True,
                              do_M='obj_solo', st=6, end=22)

    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)
    kw['fp'] = fp_neuro
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
    return r

@pkld(overwrite=False)
def run_sn_mem(region, local=False, big_voxelwise=False,
               fp_neuro='obj7_fMRI', get_age_ef=False):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    efs = []
    for sn in sns:
        try:
            ef = sn_item_dm(sn, region=region, local=local,
                            big_voxelwise=big_voxelwise,
                            fp_neuro=fp_neuro)
        except FileNotFoundError:
            efs.append(np.nan)
            continue
        efs.append(ef)
    if get_age_ef:
        efs_YA = [ef for sn, ef in zip(sns, efs) if str(sn)[0] == '1']
        efs_OA = [ef for sn, ef in zip(sns, efs) if str(sn)[0] == '2']
        t, p = stats.ttest_ind(efs_YA, efs_OA, equal_var=False)
        return t, p, efs
    else:
        t, p = stats.ttest_1samp(efs, 0, axis=0, nan_policy='omit')
        return t, p, efs

@pkld(overwrite=False)
def run_sn_attn_enc(region, control_item=False, do_acc=True,
                    FC=None, local=False, attn=True,
                    normalize=True, local_target=True):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    efs = []

    # print(1 + 'str')


    for sn in sns:


        try:
            if FC:
                ef = sn_attn_enc_Tha(sn, region, FC_target=FC, local=local,
                                     control_item=control_item,
                                     do_acc=do_acc, attn=attn,
                                     normalize=normalize,
                                     local_target=local_target)
            else:
                ef = sn_attn_enc(sn, region=region, local=local,
                                 control_item=control_item,
                                 do_acc=do_acc, attn=attn,
                                 normalize=normalize)
        except FileNotFoundError as e:
            # print(f'{e=}')
            # quit()
            efs.append(np.nan)
            continue
        efs.append(ef)
    t, p = stats.ttest_1samp(efs, 0, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(efs), axis=0)
    return t, efs

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
            # ef = sn_attn_enc(sn, region=REGION)
            try:
                ef = sn_attn_enc(sn, region=REGION)
                # ef = sn_attn_enc_Tha(sn, region=REGION,
                #                      )
            except np.linalg.LinAlgError:
                pass
            # ef = sn_item_dm(sn, region=REGION)
            if np.isnan(ef): continue
            if ef is None: continue
            efs.append(ef)
        t, p = stats.ttest_1samp(efs, 0, axis=0)
        N = np.sum(~np.isnan(efs), axis=0)
        print(f'{REGION} | t[{N - 1}] = {t:.2f}, {p=:.3f}')



if __name__ == '__main__':
    test_sn_llama_mem()