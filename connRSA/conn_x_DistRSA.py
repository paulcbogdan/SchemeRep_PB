from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from Utils.atlas_funcs import get_atlas
from connRSA_finalizing.plot_Fig5_conn import run_IC_analysis, plot_FC_mat
from connRSA_finalizing.plot_Fig34_main_bars import plot_DistRep_bars

import utils
# from Utils.pickle_wrap_funcs import pickle_wrap

# suppress RuntimeWarning: All-NaN slice


def plot_conn_vs_DistRSA(ERS=True, regress_FC=False, RSA_feat=False):
    # corr = run_IC_analysis(ERS=ERS, regress_FC=regress_FC, get_M=True,
    #                        rsa_feat=False)
    corr = utils.pickle_wrap(run_IC_analysis, easy_override=False,
                             kwargs={'ERS': ERS, 'regress_FC': regress_FC,
                                     'get_M': True, 'rsa_feat': RSA_feat})


    corrs_FC, sns_FC = (
        utils.pickle_wrap(plot_FC_mat, None, easy_override=False))

    atlas = get_atlas()

    region2score_IC = defaultdict(list)
    region2score_FC = defaultdict(list)

    out = np.zeros(atlas['maps'].shape)
    for i, (ROI, region_i) in enumerate(zip(atlas['ROIs'],
                                            atlas['ROI_regions'])):
        for j, (ROI_j, region_j) in enumerate(zip(atlas['ROIs'],
                                                atlas['ROI_regions'])):
            if region_i == region_j:
                if i % 2 != j % 2: continue
                region2score_IC[region_i].append(
                    np.nanmedian(corr[:, i, j], axis=0))
                region2score_FC[region_i].append(
                    np.nanmedian(corrs_FC[:, i, j], axis=0))

    region2dist_RSA = {}
    for region in region2score_IC.keys():
        betas = utils.pickle_wrap(plot_DistRep_bars,
                                  easy_override=False,
                                  kwargs={'region': region,
                                          'big_voxelwise': False,
                                          'get_betas': True})
        vis_local = betas[(False, 'Local')]
        vis_dist = betas[(False, 'Distributed')]

        sem_local = betas[(True, 'Local')]
        sem_dist = betas[(True, 'Distributed')]

        # v = sem_dist - sem_local
        v = sem_dist - sem_local + vis_dist - vis_local

        t, p = stats.ttest_1samp(v, 0, nan_policy='omit')
        region2dist_RSA[region] = t
        # print(vis_local.shape)

        # print(list(betas))
        # print(betas.shape)
        # quit()

    v_IC = []
    v_FC = []
    v_RSA = []
    for region, l_IC in region2score_IC.items():
        IC_gm = np.nanmedian(l_IC)
        # print(IC_gm)
        # quit()
        v_IC.append(IC_gm)
        # l_FC = region2score_FC[region]
        RSA = region2dist_RSA[region]
        # FC_gm = np.nanmedian(l_FC)
        # print(f'{region}, IC={IC_gm:.4f}, FC={FC_gm:.4f}')
        print(f'{region}, IC={IC_gm:.4f}, RSA={RSA:.2f}')
        v_RSA.append(RSA)

        # v_IC.append(IC_gm / FC_gm)
        # v_FC.append(FC_gm / FC_gm)

    r, p = stats.spearmanr(v_IC, v_RSA)
    plt.scatter(v_IC, v_RSA)
    plt.title(f'IC vs. RSA, r={r:.2f}, p={p:.2f}')
    plt.show()

    v_IC = stats.zscore(v_IC)
    # v_RSA = stats.zscore(v_RSA)
    plt.plot(v_IC, color='g', label='IC')
    plt.plot(v_RSA, color='r', label='RSA')
    plt.xticks(range(len(atlas['tick_labels'])), atlas['tick_labels'],
               rotation=90)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    plot_conn_vs_DistRSA()