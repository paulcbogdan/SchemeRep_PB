import numpy as np
from connsearch.report import plot_ROI_scores
from matplotlib import pyplot as plt

from atlas_utils import get_atlas
from conn_utils import get_BNA_ROIs
from network_clf import generic_prep
from utils import stdize


def run_HC_schaef(threshold=0.95):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'schaefer',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    partitions, sn_inc_activity_sch, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)
    # print(sn_inc_activity_sch.shape)
    # n_nans = np.sum(np.isnan(sn_inc_activity_sch[:, 0]))
    # n_non_nans = np.sum(~np.isnan(sn_inc_activity_sch[:, 0]))
    # print(f'{n_nans=}, {n_non_nans=}')
    # quit()

    kwargs1 = kwargs.copy()
    kwargs1['atlas_name'] = 'BNA'
    _, sn_inc_activity_bna, _, _, _ = generic_prep(kwargs1,
                                                   threshold=threshold)
    print(sn_inc_activity_bna.shape)
    # BNA = get_atlas()
    ROIs = get_BNA_ROIs()
    idxs_hc = [i for i, ROI in enumerate(ROIs) if 'Hipp' in ROI]
    sn_inc_activity_hc = sn_inc_activity_bna[:, :, idxs_hc, :]
    # idxs_phg = [i for i, ROI in enumerate(ROIs) if 'PhG' in ROI]

    all_conns = []
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
        sn_inc_activity_sch_age = sn_inc_activity_sch[age_idxs]
        age_conns = []
        for i in range(sn_inc_activity_hc_age.shape[0]):
            act_hc = sn_inc_activity_hc_age[i]
            act_sch = sn_inc_activity_sch_age[i]
            # print(act_hc.shape)
            # print(f'{act_sch.shape=}')
            # n_nans = np.sum(np.isnan(act_sch[1, :, :]))
            # n_non_nans = np.sum(~np.isnan(act_sch[1, :, :]))
            # print(f'{n_nans=}, {n_non_nans=}')
            # quit()
            sn_conns = []
            for cond in range(act_hc.shape[0]):
                act_hc_cond = act_hc[cond]
                act_hc_cond = stdize(act_hc_cond, axis=1, nans=True)
                act_sch_cond = act_sch[cond]

                act_sch_cond = stdize(act_sch_cond, axis=1, nans=True)
                conn = act_hc_cond[None, ...] * act_sch_cond[:, None, :]
                conn = np.nanmean(conn, axis=-1)
                sn_conns.append(conn)
                # print(conn)
                # quit()
            age_conns.append(sn_conns)
        age_conns = np.array(age_conns)
        age_conns = np.nanmean(age_conns, axis=-1)
        all_conns.append(age_conns)

        dif_conn = age_conns[:, 0] - age_conns[:, 1]
        dif_M = np.nanmean(dif_conn, axis=0)
        dif_std = np.nanstd(dif_conn, axis=0)
        dif_N = np.sum(~np.isnan(dif_conn), axis=0)
        dif_se = dif_std / np.sqrt(dif_N)
        dif_t = dif_M / dif_se
        # dif_t = dif_M

        atlas = get_atlas(schaefer=True)
        coords = atlas['coords']
        plot_ROI_scores(dif_t, coords, fp_out=fp, show=True,
                        title=f'age = {age}')


    itr_t = run_two_sample_on_2D(all_conns[0][:, 0] - all_conns[0][:, 1],
                                 all_conns[1][:, 0] - all_conns[1][:, 1])
    plot_ROI_scores(itr_t, coords, fp_out=fp, show=True,
                    title=f'age x inc')
    # quit()
    itr_t = run_two_sample_on_2D(all_conns[0][:, 0] + all_conns[0][:, 1],
                                 all_conns[1][:, 0] + all_conns[1][:, 1])
    plot_ROI_scores(itr_t, coords, fp_out=fp, show=True,
                    title=f'age eff')



def run_two_sample_on_2D(ar0, ar1):
    YA_M = np.nanmean(ar0, axis=0)
    OA_M = np.nanmean(ar1, axis=0)
    YA_sd = np.nanstd(ar0, axis=0)
    OA_sd = np.nanstd(ar1, axis=0)
    YA_N = np.sum(~np.isnan(ar0), axis=0)
    OA_N = np.sum(~np.isnan(ar1), axis=0)

    both_sd = ((YA_sd ** 2) * (YA_N - 1) +
               (OA_sd ** 2) * (OA_N - 1)) / \
              (YA_N + OA_N - 2)
    both_se = np.sqrt(both_sd * (1 / YA_N + 1 / OA_N))
    t = (YA_M - OA_M) / both_se
    t[np.isnan(t)] = 0
    return t

if __name__ == '__main__':
    run_HC_schaef()







