import numpy as np


from org_sns import get_sns
from organize_bhv import get_trial_info
import utils

from stim import get_semantic_vectors, get_DNN_vecs
from functools import cache
import matplotlib.pyplot as plt

def within_run_to_nan2(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        RDM_[low:high, low:high] = np.nan
    # TODO: Fix, this won't work properly except for on encoding!!
    return RDM_

@cache
def get_sn_fp_stim_RSM(sn, fp, semantic, dnn_layer=None, dist='corr',
                       within_to_nan=True):
    if semantic:
        d_vecs = get_semantic_vectors(normalize=True)
    else:
        d_vecs = get_DNN_vecs(DNN_layer=dnn_layer, PCA=True, PCA_obj=True)

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    vecs = [d_vecs[obj] for obj in df_sn['obj']]
    if dist == 'corr':
        RSM = np.corrcoef(vecs)
    elif dist == 'spear':
        raise ValueError
    if within_to_nan:
        RSM = within_run_to_nan2(RSM)
    return RSM

def make_all_stim_RSMs():
    sns = get_sns('all')['healthy']
    successful_sns = []
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    for sn in sns:
        for fp in ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']:
            RSM = utils.pickle_wrap(get_sn_fp_stim_RSM, None,
                                    kwargs={'sn': sn, 'fp': fp, 'semantic': True,
                                            'dnn_layer': None})
            print(f'done: {sn}, semantic ')
            for dnn_layer in range(2, 36, 2):
                RSM = utils.pickle_wrap(get_sn_fp_stim_RSM, None,
                                        kwargs={'sn': sn, 'fp': fp, 'semantic': False,
                                                'dnn_layer': dnn_layer})
                print(f'done: {sn}, DNN {dnn_layer}')
            # for semantic in [True, False]:
            #     for dnn_layer in range(1, 9):
            #         try:
            #             get_sn_fp_stim_RSM(sn, fp, semantic, dnn_layer)
            #             successful_sns.append(sn)
            #         except Exception as e:
            #             print(f'{sn} failed: {e}')

if __name__ == '__main__':
    # fp_test = r'C:\PycharmProjects\SchemeRep\cache\conn_RSA\ars\RSA\obj7_fMRI_corr_spear_within_nan_False\102_stim_False.npy'
    # RSM = np.load(fp_test)
    # RSM_new = get_sn_fp_stim_RSM(102, 'obj7_fMRI', False, 2)
    # plt.imshow(RSM)
    # plt.show()
    # plt.imshow(RSM_new)
    # plt.show()
    make_all_stim_RSMs()
