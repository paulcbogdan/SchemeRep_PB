from HCP_gambling.HCP_rs_x_task_corr import get_sn_roi_ar_std, find_overlapping_sns
from networks.old_Apr6.fluctuations import partial_corr_df
from networks.sn_anat_fluc import get_quads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from old.plot_gen import plot_connectivity


def get_corr_matrix(rs, pda, pdp, pva, pvp, p_no):
    cols_order = ['Lda_Ldp', 'Rda_Rdp',
                  'Lva_Lvp', 'Rva_Rvp',
                  'Ldp_Lvp', 'Rdp_Rvp',
                  'Lda_Lva', 'Rda_Rva',
                  'Ldp_Rdp', 'Lvp_Rvp',
                  'Lda_Rda', 'Lva_Rva', ]

    lda_ldp = rs[*np.ix_(pda[::2], pdp[::2]), :].mean(axis=(0, 1))
    rda_rdp = rs[*np.ix_(pda[1::2], pdp[1::2]), :].mean(axis=(0, 1))
    lva_lvp = rs[*np.ix_(pva[::2], pvp[::2]), :].mean(axis=(0, 1))
    rva_rvp = rs[*np.ix_(pva[1::2], pvp[1::2]), :].mean(axis=(0, 1))

    ldp_lvp = rs[*np.ix_(pdp[::2], pvp[::2]), :].mean(axis=(0, 1))
    rdp_rvp = rs[*np.ix_(pdp[1::2], pvp[1::2]), :].mean(axis=(0, 1))
    lda_lva = rs[*np.ix_(pda[::2], pva[::2]), :].mean(axis=(0, 1))
    rda_rva = rs[*np.ix_(pda[1::2], pva[1::2]), :].mean(axis=(0, 1))

    ldp_rdp = rs[*np.ix_(pdp[::2], pdp[1::2]), :].mean(axis=(0, 1))
    lvp_rvp = rs[*np.ix_(pvp[::2], pvp[1::2]), :].mean(axis=(0, 1))
    lda_rda = rs[*np.ix_(pda[::2], pda[1::2]), :].mean(axis=(0, 1))
    lva_rva = rs[*np.ix_(pva[::2], pva[1::2]), :].mean(axis=(0, 1))

    cols_ctrl = ['pd_no_L', 'ad_no_L', 'av_no_L', 'pv_no_L',
                 'pd_no_R', 'ad_no_R', 'av_no_R', 'pv_no_R',]

    pd_no_L = rs[*np.ix_(pdp[::2], p_no), :].mean(axis=(0, 1))
    pv_no_L = rs[*np.ix_(pvp[::2], p_no), :].mean(axis=(0, 1))
    ad_no_L = rs[*np.ix_(pda[::2], p_no), :].mean(axis=(0, 1))
    av_no_L = rs[*np.ix_(pva[::2], p_no), :].mean(axis=(0, 1))

    pd_no_R = rs[*np.ix_(pdp[1::2], p_no), :].mean(axis=(0, 1))
    pv_no_R = rs[*np.ix_(pvp[1::2], p_no), :].mean(axis=(0, 1))
    ad_no_R = rs[*np.ix_(pda[1::2], p_no), :].mean(axis=(0, 1))
    av_no_R = rs[*np.ix_(pva[1::2], p_no), :].mean(axis=(0, 1))

    cols = cols_order + cols_ctrl

    df = pd.DataFrame(np.array([lda_ldp, rda_rdp, lva_lvp, rva_rvp,
                                ldp_lvp, rdp_rvp, lda_lva, rda_rva,
                                ldp_rdp, lvp_rvp, lda_rda, lva_rva,
                                pd_no_L, pv_no_L, ad_no_L, av_no_L,
                                pd_no_R, pv_no_R, ad_no_R, av_no_R,
                                ]).T, columns=cols)

    corr = partial_corr_df(df.copy(), cols_order,
                           cov=['pd_no_L', 'ad_no_L', 'av_no_L', 'pv_no_L',
                                'pd_no_R', 'ad_no_R', 'av_no_R', 'pv_no_R',
                                ])
    return corr, cols_order

def analyze_rs_sn(sn, lr, pda, pdp, pva, pvp, p_no):
    kw = {'lr': lr, 'combine_regions': False,
          'bilateral': False, 'reg_global': False,
          'no_compcor': False, 'rs': True}

    ar = get_sn_roi_ar_std(sn=sn, **kw)
    rs = ar[:, None, :] * ar[None, :, :]
    mat, cols_order = get_corr_matrix(rs, pda, pdp, pva, pvp, p_no)
    return mat, cols_order


def replicate_vendor():
    task_reg_global = True
    task_no_compcor = True
    rs_reg_global = False
    rs_no_compcor = False

    sns = find_overlapping_sns()
    sns = sorted(list(sns))

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(False, all_roi=False, anat_ver=3,
                  combine_regions=False))

    mats_all = []
    for sn in sns[:4]:
        sn_mats = []
        for lr in ['LR', 'RL']:
            mat, cols_order = analyze_rs_sn(sn, lr, p_d_ant,
                                            p_d_pos, p_v_ant,
                                            p_v_pos, p_no)
            sn_mats.append(mat)
            break
        sn_mat = np.nanmean(sn_mats, axis=0)
        mats_all.append(sn_mat)
    corr = np.nanmean(mats_all, axis=0)

    bool_ar = np.zeros(corr.shape)
    # new_corr = []
    for i in range(corr.shape[0]):
        row = corr[i]
        median = np.nanmedian(row)
        bool_ar[i, row > median - .0001] += 1
        bool_ar[row > median - .0001, i] += 1
    bool_ar[bool_ar > 1.5] = 1
    corr = bool_ar

    tick_lows = np.arange(0, len(cols_order))
    ticks = tick_lows
    tick_labels = cols_order
    plot_connectivity(corr, ticks, tick_labels, tick_lows,
                      title=None, no_avg=True, #vmin=-0.15, vmax=0.15,
                      minimal=False)
    # plt.imshow(mat)
    # plt.show()
    quit()

if __name__ == '__main__':
    replicate_vendor()
