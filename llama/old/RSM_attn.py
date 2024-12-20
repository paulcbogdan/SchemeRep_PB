from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.conn_regress import do_regr_RSA_sn
from org_sns import get_sns
import numpy as np
import scipy.stats as stats

def do_RSM_attn(region='PFC'):

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]

    fps = ['bl7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    fps = ['obj7_fMRI']

    # model = 'meta-llama/Llama-3.2-3b'
    model = 'meta-llama/Llama-3.1-70b'

    attn_l = []
    for layer in range(80):
        attn_l.append(('llama', 'attn_weights', layer, 'obj',
                      model, True))
        attn_l.append(('llama', 'attn_weights', layer, 'scn',
                      model, True))


    item_l = []

    for layer in range(80):
        item_l.append(('llama', 'gate_proj_in', layer, 'obj',
                     model, True))
        item_l.append(('llama', 'gate_proj_in', layer, 'scn',
                     model, True))

    ctrl = item_l
    focus = attn_l


    betas_local = np.full((len(sns), len(fps)), np.nan)
    betas_dist = np.full((len(sns), len(fps)), np.nan)
    betas_dif = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kw = {'semantic': focus, 'trial_similarity': 'corr',
                  'second_order': 'spear', 'RDM_method': 'within_nan',
                  'stdize_by_run': False, 'regress_row': False,
                  }

            kw['sn'] = sn
            kw['fp'] = fp
            kw['return_dif'] = True
            kw['ROI_focus'] = f'{region}_M_corr'
            kw['ROIs_ctrl'] = [ctrl]

            beta1, beta2, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                            verbose=-1, easy_override=False,
                                            dir_branches=100)
            betas_local[i, j] = beta1
            betas_dist[i, j] = beta2
            betas_dif[i, j] = dif

    betas_local *= 1_000
    betas_dist *= 1_000

    betas_local = np.nanmean(betas_local, axis=1)
    betas_dist = np.nanmean(betas_dist, axis=1)

    t_beta, p_beta = stats.ttest_rel(betas_local, betas_dist)
    print(f'Local vs. distributed, beta, t = {t_beta:.2f}, p = {p_beta:.3f}')
    t_local_beta_1samp, p_local_beta_1samp = stats.ttest_1samp(betas_local, 0)
    print(f'\tLocal 1-samp, beta: t = {t_local_beta_1samp:.2f}, p = {p_local_beta_1samp:.3f}')
    t_dist_beta_1samp, p_dist_beta_1samp = stats.ttest_1samp(betas_dist, 0)
    print(f'\tDist 1-samp, beta: t = {t_dist_beta_1samp:.2f}, p = {p_dist_beta_1samp:.3f}')


if __name__ == '__main__':
    do_RSM_attn()
