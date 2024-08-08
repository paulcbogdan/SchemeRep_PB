import pandas as pd

from atlas_utils import get_atlas
from load_more import load_a
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import partial_corr_df
from utils import pickle_wrap, stdize
from vendor_partitioning import do_regression, get_vendor_partitions
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

def load_rs():
    sn_roi_act, sns, conn_trials = load_a(fp='rs_medium', norm_std=True)
    return conn_trials

def load_task(combine_regions=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
    atlas = get_atlas(combine_regions=combine_regions)
    bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
             if roi in bad_rois]
    sn_inc_conn[..., bad_j, :] = np.nan
    sn_inc_conn[..., :, bad_j] = np.nan
    z_both = do_regression(sn_inc_conn, flip=False, nans=True)
    partitions_VD, _ = get_main_partitions(z_both)


    z_both = do_regression(sn_inc_conn, flip=True, nans=True)
    partitions_PA, _ = get_main_partitions(z_both)

    partitions_VD = [set(partitions_VD[0]), set(partitions_VD[1])]
    partitions_PA = [set(partitions_PA[0]), set(partitions_PA[1])]

    quad0 = partitions_VD[0].intersection(partitions_PA[0])
    quad1 = partitions_VD[0].intersection(partitions_PA[1])
    quad2 = partitions_VD[1].intersection(partitions_PA[0])
    quad3 = partitions_VD[1].intersection(partitions_PA[1])

    quads = [list(quad0), list(quad1), list(quad2), list(quad3)]

    # print(f'{len(quad0)=}')
    # print(f'{len(quad1)=}')
    # print(f'{len(quad2)=}')
    # print(f'{len(quad3)=}')
    #
    # print(f'{len(partitions_VD[0])=}')
    # print(f'{len(partitions_VD[1])=}')
    # print(f'{len(partitions_PA[0])=}')
    # print(f'{len(partitions_PA[1])=}')

    non_used_nodes = (set(range(246)) - partitions_VD[0] -  partitions_VD[1] -
                      partitions_PA[0] - partitions_PA[1])
    non_used_quad = (set(range(246)) - quad0 - quad1 - quad2 - quad3)

    partitions_VD = [list(partitions_VD[0]), list(partitions_VD[1])]
    partitions_PA = [list(partitions_PA[0]), list(partitions_PA[1])]

    return partitions_VD, partitions_PA, list(non_used_quad), quads

    # print(sn_inc_conn.shape)
    # quit()

    # sn_inc_conn[:, :, 93, :] = np.nan
    # sn_inc_conn[:, :, :, 93] = np.nan

    # sn_inc_conn =

    # for i in range(246):
    #     num_nan = np.sum(np.all(np.isnan(sn_inc_activity[:, :, i, :]),
    #                             axis=(1, 2)), axis=0)
    #     print(i, ':', num_nan)
    # quit()




    # print(sn_inc_conn[:, :, 93, 0])
    # quit()

    # plt.imshow(z_both)
    # plt.show()
    # quit()
    # print(z_both[0, 1])

    # atlas = get_atlas()

    # for k, p in enumerate(partitions):
    #     # if len(p) < 10: continue
    #     if k == 3: break
    #     mat = np.full((246, 246), 0)
    #
    #     for i in p:
    #         for j in p:
    #             mat[i, j] = 1
    #     plot_connectivity(mat, atlas=atlas, vmin=0, vmax=2, minimal=False)
    #
    # # print(partitions)
    # quit()



    # get_vendor_partitions(age='healthy', flip=True, plot=True,
    #                   scrub=True, easy_override=True, thr=THRESHOLD,
    #                   combine_regions=False, regress=REGRESS)

def calc_corr(combine_regions=False):
    conn_trials = load_rs()
    partitions_VD, partitions_PA, non_used_nodes, quads = (
        load_task(combine_regions=combine_regions))



    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_ver=3,
                              combine_regions=combine_regions)

    partitions_VD = [p_d_ant + p_d_pos, p_v_ant + p_v_pos]
    partitions_PA = [p_d_ant + p_v_ant, p_d_pos + p_v_pos]

    non_used_nodes = (set(range(246)) -
                      set(partitions_VD[0]) - set(partitions_VD[1]) -
                      set(partitions_PA[0]) - set(partitions_PA[1]))
    non_used_nodes = list(non_used_nodes)

    # VD0 = conn_trials[:, *np.ix_(partitions_VD[0], partitions_VD[0]), :]
    VD0 = conn_trials[:, *np.ix_(p_d_ant, p_d_pos), :]

    # VD0 = conn_trials[:, *np.ix_(quads[0], quads[1])]
    VD0 = np.nanmean(VD0, axis=(1, 2))
    # VD1 = conn_trials[:, *np.ix_(partitions_VD[1], partitions_VD[1]), :]
    VD1 = conn_trials[:, *np.ix_(p_v_ant, p_v_pos), :]
    # VD1 = conn_trials[:, *np.ix_(quads[2], quads[3])]
    VD1 = np.nanmean(VD1, axis=(1, 2))
    # PA0 = conn_trials[:, *np.ix_(partitions_PA[0], partitions_PA[0]), :]
    PA0 = conn_trials[:, *np.ix_(p_v_ant, p_d_ant), :]
    # PA0 = conn_trials[:, *np.ix_(quads[0], quads[2])]
    PA0 = np.nanmean(PA0, axis=(1, 2))
    # PA1 = conn_trials[:, *np.ix_(partitions_PA[1], partitions_PA[1]), :]
    PA1 = conn_trials[:, *np.ix_(p_v_pos, p_d_pos), :]
    # PA1 = conn_trials[:, *np.ix_(quads[1], quads[3])]
    PA1 = np.nanmean(PA1, axis=(1, 2))



    glob = np.nanmean(conn_trials, axis=(1, 2))
    # nde = conn_trials[:, *np.ix_(non_used_nodes, non_used_nodes), :]
    # nde = np.nanmean(nde, axis=(1, 2))
    #
    # # nde0 = conn_trials[:, *np.ix_(quads[0], non_used_nodes), :]
    # nde0 = conn_trials[:, *np.ix_(partitions_VD[0], non_used_nodes), :]
    nde0 = conn_trials[:, *np.ix_(p_d_ant, list(set(range(246)) )), :]
    nde0 = np.nanmean(nde0, axis=(1, 2))
    # nde1 = conn_trials[:, *np.ix_(partitions_VD[1], non_used_nodes), :]
    nde1 = conn_trials[:, *np.ix_(p_v_ant, non_used_nodes), :]
    nde1 = np.nanmean(nde1, axis=(1, 2))
    # nde2 = conn_trials[:, *np.ix_(partitions_PA[0], non_used_nodes), :]
    nde2 = conn_trials[:, *np.ix_(p_d_pos, non_used_nodes), :]
    nde2 = np.nanmean(nde2, axis=(1, 2))
    # nde3 = conn_trials[:, *np.ix_(partitions_PA[1], non_used_nodes), :]
    nde3 = conn_trials[:, *np.ix_(p_v_pos, non_used_nodes), :]
    nde3 = np.nanmean(nde3, axis=(1, 2))
    #
    df = pd.DataFrame({'VD0': VD0.flatten(), 'VD1': VD1.flatten(),
                       'PA0': PA0.flatten(), 'PA1': PA1.flatten(),
                       'global': glob.flatten(),
                       # 'NDE': nde.flatten(),
                       'NDE0': nde0.flatten(), 'NDE1': nde1.flatten(),
                       'NDE2': nde2.flatten(), 'NDE3': nde3.flatten()})

    df['dd_vv'] = df['VD0'] + df['VD1']
    df['dv_dv'] = df['PA0'] + df['PA1']

    # all_nodes = np.concatenate([quads[0], quads[1], quads[2], quads[3], ])

    # for quad in [quads[0], quads[1], quads[2], quads[3]]:
    # # for quad in [partitions_VD[0], partitions_VD[1],
    # #              partitions_PA[0], partitions_PA[1]]:
    #     atlas = get_atlas(combine_regions=combine_regions)
    #     coords = [atlas['coords'][i] for i in quad]
    #     plotting.plot_markers([1] * len(coords), coords)
    #     plt.show()
    # quit()

    # plt.scatter(df['dd_vv'], df['dv_dv'])
    # plt.show()

    # 'VD0', 'VD1', 'PA0', 'PA1'
    partial_corr_df(df, ['dd_vv', 'dv_dv'],
                        ['NDE0', 'NDE1', 'NDE2', 'NDE3', ])
    # 'NDE0', 'NDE1', 'NDE2', 'NDE3'
    # ['global']

    # ['dd_vv', 'dv_dv'], #

if __name__ == '__main__':
    calc_corr()


