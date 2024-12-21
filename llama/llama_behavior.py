
from connRSA.conn_regress import do_regr_RSA_sn
from llama.model_settings import get_explore_llama, get_base_kw
from llama.numba_regr_test import pairwise_interaction_t_values_proper
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy import stats

from organize_bhv import get_trial_info

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

def sn_attn_encoding(sn, region='PFC', local=True,
                     big_voxelwise=False,
                     str_interaction=True):
    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model,
                                   attn=True, normalize=True,
                                   do_prod=False, do_M=False)

    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   do_prod=False)
    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)
    kw['semantic'] = model_attn
    kw['ROIs_ctrl'] = []

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                        easy_override=False, dir_branches=100)




    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]
    df_sn['IRAFs'] = IRAFs

    df_sn['inc'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
    df_sn['enc_acc'] = (df_sn['inc'] - df_sn['per_inc']).abs()

    if str_interaction:
        kw = get_base_kw_predicting(sn, 'subcort', local, big_voxelwise)
        kw['semantic'] = model_attn
        kw['ROIs_ctrl'] = []
        IRAFs_str = pickle_wrap(do_regr_RSA_sn, kwargs=kw, verbose=-1,
                                easy_override=False, dir_branches=100)
        y = np.array(df_sn['enc_acc'].to_list())

        df_sn.dropna(subset=['IRAFs', 'enc_acc'], inplace=True)
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

    df_sn.dropna(subset=['IRAFs', 'enc_acc'], inplace=True)
    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['enc_acc'])
    print(f'{r=:.3f}')
    return r

def sn_item_dm(sn, region='Str', local=True, big_voxelwise=False):
    activation_model = 'meta-llama/Llama-3.2-3b'
    model = get_explore_llama(activation_model,
                              attn=False, normalize=True)

    kw = get_base_kw_predicting(sn, region, local, big_voxelwise)
    kw['semantic'] = model
    kw['ROIs_ctrl'] = []

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                        verbose=-1, easy_override=False,
                        dir_branches=100)
    if len(kw['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]

    df_sn = get_trial_info(sn)
    df_sn.sort_values(by=f'obj_trial', inplace=True)
    df_sn['IRAFs'] = IRAFs

    # df_sn['mem_key'] = df_sn['vis_hit'] & df_sn['con_hit']
    df_sn['mem_key'] = df_sn['con_hit'].astype(float)

    df_sn.dropna(subset=['IRAFs', 'mem_key'], inplace=True)
    assert len(df_sn) > 10, f'{len(df_sn)=}'

    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['mem_key'])
    print(f'{r=:.3f}')
    return r


def get_sn_llama_mem(sn, fps, mem_key='vis_hit'):
    # model = get_explore_llama('meta-llama/Llama-3.3-70b-Instruct',
    #                           attn=True)
    activation_model = 'meta-llama/Llama-3.2-3b'
    # activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    model = get_explore_llama(activation_model,
                              attn=True, normalize=True)
    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True,
                                   do_prod=True)

    # model_item = get_explore_llama(activation_model,
    #                                attn=True, normalize=True, st=0)
    # model = get_explore_llama(activation_model,
    #                           attn=False, normalize=True,
    #                           do_prod=True, st=0)
    # model = get_explore_BERT('BERT')
    # model = get_explore_BERT('simCSE')

    # model = True
    kwargs = {'semantic': model,
              'fp': None,
              'trial_similarity': 'corr',
              'second_order': 'spear',
              'RDM_method': 'within_nan',
              'stdize_by_run': False,
              }
    kwargs['sn'] = sn
    kwargs['fp'] = fps[0]

    kwargs['ROI_focus'] = f'ITL_BOLD_cmb'
    kwargs['ROI_focus'] = f'PFC_M_corr' # MAYBE for obj7_fMRI
    # kwargs['ROI_focus'] = f'ITL_BOLD_cmb'
    # kwargs['ROI_focus'] = f'ITL_M_corr'
    # kwargs['ROI_focus'] = f'cortical_M_corr'

    kwargs['ROIs_ctrl'] = []
    # kwargs['ROIs_ctrl'] = [model_item]
    kwargs['regress_row'] = True

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                        verbose=-1,
                        easy_override=False, dir_branches=100)
    if len(kwargs['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]

    df_sn = get_trial_info(sn)
    sess = (kwargs['fp'].split('_')[0].replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    df_sn['IRAFs'] = IRAFs

    # df_sn['mem_key'] = df_sn['vis_hit'] & df_sn['con_hit']
    df_sn['mem_key'] = df_sn['vis_hit'].astype(float)

    # df_sn['inc'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
    # df_sn['enc_accuracy'] = (df_sn['inc'] - df_sn['per_inc']).abs()
    # df_sn['mem_key'] = df_sn['enc_accuracy'].astype(float)

    # df_sn['per_inc'] = df_sn['per_inc'].map({1: 1, 2: 2, 3: 2, 4: 3})
    # df_sn['enc_accuracy'] = (df_sn['inc'] - df_sn['per_inc']).abs()
    # df_sn['mem_key'] = df_sn['enc_accuracy'].astype(float)


    df_sn.dropna(subset=['IRAFs', 'mem_key'], inplace=True)
    assert len(df_sn) > 10, f'{len(df_sn)=}'

    r, p = stats.pearsonr(df_sn['IRAFs'], df_sn['mem_key'])
    print(f'{r=:.3f}')
    return r


def test_sn_llama_mem():
    fps = ['obj7_fMRI']
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]

    efs = []
    for sn in sns:
        ef = sn_attn_encoding(sn)
        # ef = sn_item_dm(sn)
        if ef is None: continue
        efs.append(ef)
    t, p = stats.ttest_1samp(efs, 0, axis=0)
    N = np.sum(~np.isnan(efs), axis=0)
    print(f't[{N - 1}] = {t:.2f}, {p=:.3f}')



if __name__ == '__main__':
    test_sn_llama_mem()