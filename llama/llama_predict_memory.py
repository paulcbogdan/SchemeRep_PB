from numba.cuda.libdevice import atan2

from connRSA.conn_regress import do_regr_RSA_sn
from connRSA_finalizing.plot_bars_explore import get_explore_llama, get_explore_BERT
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy import stats

from organize_bhv import get_trial_info


def get_sn_llama_mem(sn, fps, mem_key='vis_hit'):
    # model = get_explore_llama('meta-llama/Llama-3.3-70b-Instruct',
    #                           attn=True)
    activation_model = 'meta-llama/Llama-3.2-3b'
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    model = get_explore_llama(activation_model,
                              attn=True, normalize=True)
    model_item = get_explore_llama(activation_model,
                                   attn=False, normalize=True)
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
    kwargs['ROIs_ctrl'] = [model_item]
    kwargs['regress_row'] = True

    if (kwargs['regress_row'] and len(kwargs['ROIs_ctrl']) > 0 and
            sn in ['132', '224', '234']):
        return None
    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                        verbose=-1,
                        easy_override=False, dir_branches=100)
    if len(kwargs['ROIs_ctrl']):
        IRAFs = IRAFs[:, 0]

    df_sn = get_trial_info(sn)
    sess = (kwargs['fp'].split('_')[0].replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    df_sn['IRAFs'] = IRAFs
    # print(df_sn['con_resp'].value_counts(dropna=True))

    # df_sn['mem_key'] = df_sn['vis_hit'] & df_sn['con_hit']
    df_sn['mem_key'] = df_sn['vis_hit'].astype(float)

    # function that maps 1-3 to 1-4
    df_sn['inc'] = df_sn['inc'].map({1: 1, 2: 2.5, 3: 4})
    df_sn['enc_accuracy'] = (df_sn['inc'] - df_sn['per_inc']).abs()
    df_sn['mem_key'] = df_sn['enc_accuracy'].astype(float)

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
        ef = get_sn_llama_mem(sn, fps)
        if ef is None: continue
        efs.append(ef)
    t, p = stats.ttest_1samp(efs, 0, axis=0)
    N = np.sum(~np.isnan(efs), axis=0)
    print(f't[{N - 1}] = {t:.2f}, {p=:.3f}')



if __name__ == '__main__':
    test_sn_llama_mem()