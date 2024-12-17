from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from fMRI_proc import within_run_to_nan
import numpy as np

from llama.get_obj_scn_vecs import get_sn_fp_llama_RSM_l, get_sn_fp_llama_RSM
from org_sns import get_sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def regr_fMRI_on_many_stim(sn, region, stim_l, fp, trial_similarity='corr',
                           second_order='spear', RDM_method='within_nan',
                           stdize_by_run=False, big_voxelwise=True):

    ROI_focus = f'{region}_M_corr'

    dir_in = fr'C:/PycharmProjects/SchemeRep/cache/conn_RSA/ars/RSA'
    dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method}_{stdize_by_run}')
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus}.npy'
    with open(fp_focus, 'rb') as f:
        RSM_focus = np.load(f)
        if RDM_method == 'within_nan':
            RSM_focus = within_run_to_nan(RSM_focus)

    flat_focus = RSM_focus[np.tril_indices_from(RSM_focus, k=-1)]

    RSM_ctrl_l = []
    flat_ctrl_l = []

    for ROI_ctrl in stim_l:
        if isinstance(ROI_ctrl, tuple):
            assert ROI_ctrl[0] == 'llama'
            RSM_ctrl = get_sn_fp_llama_RSM(sn, fp, ROI_ctrl)
        elif isinstance(ROI_ctrl, list):
            assert ROI_ctrl[0][0] == 'llama'
            RSM_ctrl = get_sn_fp_llama_RSM_l(sn, fp, ROI_ctrl)
        else:
            raise ValueError
        RSM_ctrl_l.append(RSM_ctrl)
        flat_ctrl = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
        flat_ctrl_l.append(flat_ctrl)
        flat_itr = np.array([1] * len(flat_ctrl))
    # print(f'{len(flat_ctrl_l)=}')
    # quit()

    flat_ctrls = np.array(flat_ctrl_l).T
    nan_cols = (np.isnan(flat_ctrls) &
                ~np.isnan(flat_focus)[:, None]).any(axis=0)
    flat_ctrls = flat_ctrls[:, ~nan_cols]
    X = np.hstack([flat_itr[:, None], flat_ctrls])
    goods = ~(np.isnan(flat_focus) | np.any(np.isnan(X), axis=1))
    # print(goods.shape)
    # quit()
    flat_focus = flat_focus[goods]
    X = X[goods]

    cnt_nan = np.sum(np.isnan(X))
    # print(f'{cnt_nan=}')
    solution, residuals, rank, s = np.linalg.lstsq(X, flat_focus,
                                                   rcond=None)
    return solution[1:]

def regr_fMRI_on_many_stim_all_sn(region, stim_l, separate=False,
                                  regr_M=True):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']

    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]
    # fps = ['obj7_fMRI']
    # fps = ['bl7_fMRI', 'con7_fMRI', 'vis7_fMRI']

    betas = np.full((len(sns), len(fps), len(stim_l)), np.nan)

    kw = {'region': region, 'stim_l': stim_l}
    for i, sn in enumerate(tqdm(sns, desc=f'({region})')):
        for j, fp in enumerate(fps):
            kw['sn'] = sn
            kw['fp'] = fp
            if separate:
                for k in range(len(stim_l)):
                    if regr_M:
                        kw['stim_l'] = [stim_l[k],
                                        stim_l]

                                        # [x for x in stim_l if x != stim_l[k]]]
                    else:
                        kw['stim_l'] = [stim_l[k]]
                    betas[i, j, k] = pickle_wrap(regr_fMRI_on_many_stim, kwargs=kw,
                                                 easy_override=False, verbose=-1)[0]
            else:
                betas_sf = pickle_wrap(regr_fMRI_on_many_stim, kwargs=kw,
                                       easy_override=False, verbose=-1)
                betas[i, j, :] = betas_sf

    betas_sns = np.nanmean(betas, axis=1)
    print(f'{region}:')
    layers = []
    vals = []
    for i in range(len(stim_l)):
        layer = stim_l[i][2]
        if separate and False:
            t, p = stats.ttest_1samp(betas_sns[:, i] -
                                     np.nanmean(betas_sns, axis=1), 0)
        else:
            t, p = stats.ttest_1samp(betas_sns[:, i], 0)
        n_non_nan = np.sum(~np.isnan(betas_sns[:, i]))
        print(f'\t{layer}: t[{n_non_nan-1}] = {t:.2f}')
        layers.append(layer)
        vals.append(t)
    plt.plot(layers, vals)
    plt.title(region)
    plt.show()


def run_all_sn():
    stim_base = ('llama', 'gate_proj_in',
                 None, 'scn',
                 'meta-llama/Llama-3.3-70b-Instruct',
                 True)

    # stim_base = ('llama', 'attn_weights',
    #              None, 'scn',
    #              'meta-llama/Llama-3.1-70b',
    #              True)

    stim_l = []
    layers = [8, 16, 24, 40, 58, 78]
    layers = [16, 32, 48, 64]
    # layers = [16, 64]
    layers = list(range(80))

    for layer in layers:
        stim = (stim_base[0], stim_base[1], layer, stim_base[3],
                stim_base[4], stim_base[5])
        stim_l.append(stim)

    target_ROIs = get_atlas(combine_regions=True,
                            combine_bilateral=True)['tick_labels']
    # target_ROIs = target_ROIs[4::5]
    target_ROIs = ['ITL', 'PFC', 'Parietal', 'Occipital']

    for region in target_ROIs:
        regr_fMRI_on_many_stim_all_sn(region, stim_l, separate=True)

    # regr_fMRI_on_many_stim_all_sn('ITL', stim_l)
    # regr_fMRI_on_many_stim_all_sn('Occipital', stim_l)
    # regr_fMRI_on_many_stim_all_sn('PFC', stim_l)
    # regr_fMRI_on_many_stim_all_sn('Parietal', stim_l)



if __name__ == '__main__':
    run_all_sn()