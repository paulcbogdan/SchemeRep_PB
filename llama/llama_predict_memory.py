from connRSA.conn_regress import do_regr_RSA_sn
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np
from scipy import stats

from organize_bhv import get_trial_info


def get_sn_llama_mem(sn, fps, mem_key='vis_hit'):
    # activation_model = 'meta-llama/Llama-3.2-3b'
    # activation_model = 'meta-llama/Llama-3.1-70b'
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'

    model_l = []
    for llama_layer in range(1, 80):
        model = ('llama', 'gate_proj_in', llama_layer,
                 'scn', activation_model, True)
        model_l.append(model)
    model = model_l

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
    # kwargs['ROI_focus'] = f'cortical_BOLD_cmb'
    kwargs['ROI_focus'] = f'ITL_BOLD_cmb'
    # kwargs['ROI_focus'] = f'ITL_M_corr'

    kwargs['ROIs_ctrl'] = []
    kwargs['regress_row'] = True

    IRAFs = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                        verbose=-1,
                        easy_override=True, dir_branches=100)

    df_sn = get_trial_info(sn)
    sess = (kwargs['fp'].split('_')[0].replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    df_sn['IRAFs'] = IRAFs
    # print(df_sn['con_resp'].value_counts(dropna=True))

    # df_sn['mem_key'] = df_sn['vis_hit'] & df_sn['con_hit']
    df_sn['mem_key'] = df_sn['vis_hit']
    # df_sn['mem_key'] = df_sn['con_resp']


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
        efs.append(ef)
    t, p = stats.ttest_1samp(efs, 0, axis=0)
    N = np.sum(~np.isnan(efs), axis=0)
    print(f't[{N - 1}] = {t:.2f}, {p=:.3f}')



if __name__ == '__main__':
    test_sn_llama_mem()