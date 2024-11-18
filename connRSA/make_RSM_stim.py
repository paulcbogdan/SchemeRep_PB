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
def get_layer2DNN_map():
    map = {0: 0, 1: 2, 2: 5, 3: 7, 4: 10,
           5: 12, 6: 14, 7: 17, 8: 19, 9: 21,
           10: 24, 11: 26, 12: 28,

           # classification layers below
           13: 31, 14: 34, 15: 37,
           -1: -1}
    return map

@cache
def get_sn_fp_stim_RSM(sn, fp, semantic, layer=None, dist='corr',
                       within_to_nan=True, ):
    if semantic:
        d_vecs = get_semantic_vectors(normalize=True)
    else:
        dnn_layer = get_layer2DNN_map()[layer]
        d_vecs = get_DNN_vecs(DNN_layer=dnn_layer, PCA=True, PCA_obj=True,
                              )

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
                                            'layer': None})
            print(f'done: {sn}, semantic ')
            for layer in range(0, 16):
                RSM = utils.pickle_wrap(get_sn_fp_stim_RSM, None,
                                        kwargs={'sn': sn, 'fp': fp, 'semantic': False,
                                                'layer': layer, },
                                        easy_override=True)
                print(f'done: {sn}, layer {layer}')
            # for semantic in [True, False]:
            #     for dnn_layer in range(1, 9):
            #         try:
            #             get_sn_fp_stim_RSM(sn, fp, semantic, dnn_layer)
            #             successful_sns.append(sn)
            #         except Exception as e:
            #             print(f'{sn} failed: {e}')

if __name__ == '__main__':
    fp_test = r'C:\PycharmProjects\SchemeRep\cache\conn_RSA\ars\RSA\obj7_fMRI_corr_spear_within_nan_False\102_stim_False.npy'
    # RSM = np.load(fp_test)
    # RSM_2 = get_sn_fp_stim_RSM(102, 'obj7_fMRI', False, 15)
    # plt.imshow(RSM)
    # plt.show()
    #
    # RSM_last = get_sn_fp_stim_RSM(102, 'obj7_fMRI', False, -1)
    # plt.imshow(RSM)
    # plt.show()
    # quit()
    # plt.imshow(RSM_new)
    # plt.show()
    make_all_stim_RSMs()
